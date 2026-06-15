import torch
import torch.nn.functional as F
import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# ==========================================
# CẤU HÌNH TRÒ
# ==========================================
STUDENT_PATH = "./LLaDA-1.5B-Drafter-Clean"
VOCAB_SIZE = 128256 # Kích thước từ điển gốc của LLaDA/Llama-3

print("Đang tải Tokenizer và Drafter 1.5B lên GPU...")
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

lora_config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"], 
    lora_dropout=0.05
)
student = get_peft_model(student, lora_config)
student.config.use_cache = False
# Đã xóa gradient_checkpointing vì dư dả RAM

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)

# ==========================================
# VÒNG LẶP HỌC QUA BÍ KÍP (OFFLINE)
# ==========================================
print("\nBắt đầu Offline Distillation (VRAM sử dụng: ~5GB)")
student.train()

# Quét tất cả các file đáp án đã lưu
batch_files = sorted(glob.glob("./offline_logits/batch_*.pt"))
progress_bar = tqdm(batch_files, desc="Đang hấp thụ")

for step, file_path in enumerate(progress_bar):
    # Đọc đề bài và đáp án Thầy để lại
    batch_data = torch.load(file_path, weights_only=True)
    input_ids = batch_data["input_ids"].to("cuda")
    topk_values = batch_data["topk_values"].to("cuda")
    topk_indices = batch_data["topk_indices"].to("cuda")
    
    # 1. PHỤC DỰNG LẠI BẢNG ĐIỂM CỦA THẦY
    # Tạo một ma trận rỗng toàn điểm âm vô cực (-10000.0)
    batch_size, seq_len, _ = topk_indices.shape
    teacher_logits = torch.full((batch_size, seq_len, VOCAB_SIZE), -10000.0, device="cuda", dtype=torch.bfloat16)
    
    # Đổ 100 điểm cao nhất vào đúng vị trí từ vựng của nó
    teacher_logits.scatter_(-1, topk_indices, topk_values)
    
    # 2. TRÒ LÀM BÀI
    student_outputs = student(input_ids=input_ids)
    student_logits = student_outputs.logits
    
    # 3. CHẤM ĐIỂM BẰNG KL-DIVERGENCE
    T = 2.0
    loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)
    
    # Lan truyền ngược
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Dọn rác Tensor tức thì
    current_loss = loss.item()
    progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
    del student_outputs, student_logits, teacher_logits, loss, topk_values, topk_indices, input_ids
    
    # Lưu Checkpoint mỗi 50 file
    if step > 0 and step % 50 == 0:
        student.save_pretrained(f"./checkpoints/offline_step_{step}")

print("\nHấp thụ bí kíp hoàn tất!")
merged_model = student.merge_and_unload()
merged_model.save_pretrained("./LLaDA-1.5B-Offline-Pro")
tokenizer.save_pretrained("./LLaDA-1.5B-Offline-Pro")
print("Con Drafter tối thượng đã ra lò tại ./LLaDA-1.5B-Offline-Pro")
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import gc

# ==========================================
# 1. CẤU HÌNH AN TOÀN CHO L4 24GB
# ==========================================
TEACHER_ID = "GSAI-ML/LLaDA-8B-Instruct"
STUDENT_PATH = "./LLaDA-1.5B-Drafter-Clean"
BATCH_SIZE = 1                  # Giữ Batch 1 để tiết kiệm VRAM tuyệt đối
ACCUMULATION_STEPS = 16         # Tích lũy 16 vòng (tương đương Batch 16)
MASKING_RATIO = 0.30            
MAX_LENGTH = 128                

# ==========================================
# 2. TẢI CẢ THẦY VÀ TRÒ DƯỚI DẠNG NGUYÊN BẢN (BFLOAT16)
# ==========================================
print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
mask_id = tokenizer.mask_token_id

# KHÔNG DÙNG BITSANDBYTES! Tải thẳng bf16 siêu sắc nét
print("Đang tải Thầy 8B (~15GB VRAM)...")
teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
teacher.eval() # Khóa Thầy

print("Đang tải Trò 1.5B (~3GB VRAM)...")
student = AutoModelForCausalLM.from_pretrained(
    STUDENT_PATH, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)

# Lắp LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"], 
    lora_dropout=0.05
)
student = get_peft_model(student, lora_config)

# TUYỆT ĐỐI KHÔNG GỌI student.resize_token_embeddings() Ở ĐÂY NỮA!

# Bật khiên chống nổ RAM cho quá trình học
student.config.use_cache = False
if hasattr(student, "gradient_checkpointing_enable"):
    student.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

# ==========================================
# 3. DATA LOADER ĐỤC LỖ
# ==========================================
print("Chuẩn bị Dữ liệu Wikipedia...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset['text'] if len(t.strip()) > 50][:4000]

def collate_fn(batch):
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    rand_matrix = torch.rand(input_ids.shape)
    mask_condition = (rand_matrix < MASKING_RATIO) & (attention_mask == 1)
    
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_condition] = mask_id
    
    return {
        "input_ids": masked_input_ids.to("cuda"),
        "attention_mask": attention_mask.to("cuda")
    }

dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 4. VÒNG LẶP CHƯNG CẤT (ĐỈNH CAO)
# ==========================================
print("\nBắt đầu Chưng cất Logits (Soft Labels) | Không nén!")
os.makedirs("./checkpoints", exist_ok=True)
student.train()
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Đang chưng cất")

for step, batch in progress_bar:
    # --- PRO-TRICK: TÍNH THẦY XONG XÓA LUÔN ĐỂ CỨU RAM ---
    with torch.no_grad():
        teacher_outputs = teacher(**batch)
        teacher_logits = teacher_outputs.logits.detach() # Tách hẳn khỏi đồ thị
    del teacher_outputs 
    # ----------------------------------------------------

    # Trò làm bài
    student_outputs = student(**batch)
    student_logits = student_outputs.logits
    
    # Ép Trò học bảng điểm của Thầy (KL-Divergence)
    T = 2.0
    loss = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction='batchmean'
    ) * (T * T)
    
    loss = loss / ACCUMULATION_STEPS
    loss.backward()
    
    # Dọn rác trung gian ngay lập tức
    current_loss = loss.item() * ACCUMULATION_STEPS
    del student_outputs, student_logits, teacher_logits, loss
    
    # Cập nhật LoRA
    if (step + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        real_step = (step + 1) // ACCUMULATION_STEPS
        
        if real_step % 100 == 0:
            student.save_pretrained(f"./checkpoints/drafter_step_{real_step}")

print("\nChưng cất hoàn tất!")
merged_model = student.merge_and_unload()
merged_model.save_pretrained("./LLaDA-1.5B-KD-Drafter")
tokenizer.save_pretrained("./LLaDA-1.5B-KD-Drafter")
print("Bản sao hoàn hảo đã ra lò tại ./LLaDA-1.5B-KD-Drafter")
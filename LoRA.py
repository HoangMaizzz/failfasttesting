import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# 1. CẤU HÌNH SIÊU THAM SỐ (Dành cho L4 24GB)
# ==========================================
TEACHER_ID = "GSAI-ML/LLaDA-8B-Instruct"
STUDENT_PATH = "./LLaDA-1.5B-Drafter-Clean"
BATCH_SIZE = 4                  # L4 dư sức gánh Batch 4
ACCUMULATION_STEPS = 4          # Tích lũy 4 vòng -> Effective Batch = 16
MASKING_RATIO = 0.30            # Đục lỗ ngẫu nhiên 30% số chữ
MAX_LENGTH = 256                # Chiều dài câu
TEMPERATURE = 2.0               # Độ mềm của Loss

# ==========================================
# 2. TẢI MÔ HÌNH VÀ BỘ NHỚ
# ==========================================
print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)

# Xử lý các token đặc biệt còn thiếu cho mô hình nền Llama
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.mask_token is None:
    tokenizer.add_special_tokens({'mask_token': '[MASK]'})

mask_id = tokenizer.mask_token_id

print("Đang tải Thầy 8B (4-bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
teacher = AutoModelForCausalLM.from_pretrained(TEACHER_ID, quantization_config=bnb_config, trust_remote_code=True, device_map={"": 0})
teacher.resize_token_embeddings(len(tokenizer)) # Thay đổi kích thước embedding để chứa token mới
teacher.eval() # Thầy chỉ đứng nhìn, không học

print("Đang tải Trò 1.5B (bf16) lên GPU...")
student = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True)
student.resize_token_embeddings(len(tokenizer)) # Đồng bộ kích thước embedding cho Trò

# Lắp LoRA cho Trò
lora_config = LoraConfig(
    r=32, # Tăng rank lên 32 cho L4 học cho sâu
    lora_alpha=64, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj"], 
    lora_dropout=0.05
)
student = get_peft_model(student, lora_config)
optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)

# ==========================================
# 3. CHUẨN BỊ DỮ LIỆU ĐA LUỒNG (DATALOADER)
# ==========================================
print("Chuẩn bị Dữ liệu (Wikipedia)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset['text'] if len(t.strip()) > 50][:4000] # Lấy 4000 câu

def collate_fn(batch):
    # Mã hóa cả cụm
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # THUẬT TOÁN ĐỤC LỖ NGẪU NHIÊN (Chỉ đục những chữ có nghĩa, bỏ qua Padding)
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
# 4. VÒNG LẶP HUẤN LUYỆN PRO
# ==========================================
print(f"\nBắt đầu huấn luyện | Batch: {BATCH_SIZE} | Accumulation: {ACCUMULATION_STEPS} | Mask: 30%")

# Tạo thư mục chứa các bản lưu tạm nếu chưa có
os.makedirs("./checkpoints", exist_ok=True)

student.train()
optimizer.zero_grad()

progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Đang trị liệu")

for step, batch in progress_bar:
    # 1. Thầy tính điểm chuẩn
    with torch.no_grad():
        teacher_outputs = teacher(**batch)
        teacher_logits = teacher_outputs.logits

    # 2. Trò đoán thử
    student_outputs = student(**batch)
    student_logits = student_outputs.logits
    
    # 3. Tính độ lệch (Loss)
    loss = F.kl_div(
        F.log_softmax(student_logits / TEMPERATURE, dim=-1),
        F.softmax(teacher_logits / TEMPERATURE, dim=-1),
        reduction='batchmean'
    ) * (TEMPERATURE ** 2)
    
    loss = loss / ACCUMULATION_STEPS
    loss.backward()

    # 4. Cập nhật trọng số
    if (step + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        real_step = (step + 1) // ACCUMULATION_STEPS
        current_loss = loss.item() * ACCUMULATION_STEPS
        
        # Cập nhật hàm Loss trực tiếp lên thanh tiến trình
        progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
        
        # ----------------------------------------------------
        # BẢO HIỂM CHECKPOINT: Cứ 100 bước lưu lại 1 lần
        # ----------------------------------------------------
        if real_step > 0 and real_step % 100 == 0:
            ckpt_path = f"./checkpoints/drafter_step_{real_step}"
            student.save_pretrained(ckpt_path)
            # In ra một dòng nhỏ để an tâm
            progress_bar.write(f"Đã lưu an toàn Checkpoint tại bước {real_step}")

print("\nHoàn tất khóa huấn luyện!")
# Gộp trọng số và lưu bản Final
merged_model = student.merge_and_unload()
merged_model.save_pretrained("./LLaDA-1.5B-Pro-Drafter")
tokenizer.save_pretrained("./LLaDA-1.5B-Pro-Drafter")
print("Mô hình Pro Final đã lưu tại ./LLaDA-1.5B-Pro-Drafter")
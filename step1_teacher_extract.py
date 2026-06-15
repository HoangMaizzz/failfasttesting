import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# CẤU HÌNH THẦY
# ==========================================
TEACHER_ID = "GSAI-ML/LLaDA-8B-Instruct"
BATCH_SIZE = 16
MAX_LENGTH = 128
MASKING_RATIO = 0.30

os.makedirs("./offline_logits", exist_ok=True)

print("Đang tải Thầy 8B lên toàn bộ GPU L4...")
tokenizer = AutoTokenizer.from_pretrained(TEACHER_ID, trust_remote_code=True)
mask_id = tokenizer.mask_token_id

teacher = AutoModelForCausalLM.from_pretrained(
    TEACHER_ID, 
    torch_dtype=torch.bfloat16, 
    device_map="auto", 
    trust_remote_code=True
)
teacher.eval() # Khóa cứng Thầy

# ==========================================
# CHUẨN BỊ DỮ LIỆU ĐỤC LỖ CHUẨN XÁC
# ==========================================
print("Tải dữ liệu Wikipedia...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset['text'] if len(t.strip()) > 50][:4000]

def collate_fn(batch):
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Đục lỗ
    rand_matrix = torch.rand(input_ids.shape)
    mask_condition = (rand_matrix < MASKING_RATIO) & (attention_mask == 1)
    
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_condition] = mask_id
    
    return masked_input_ids, attention_mask

dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==========================================
# CHẠY VÀ LƯU TOP-K LOGITS
# ==========================================
print("Bắt đầu vắt kiệt tri thức của Thầy ra ổ cứng...")
for step, (masked_inputs, attn_mask) in enumerate(tqdm(dataloader)):
    masked_inputs = masked_inputs.to("cuda")
    
    with torch.no_grad():
        outputs = teacher(input_ids=masked_inputs)
        logits = outputs.logits
        
        # TUYỆT KỸ ÉP DUNG LƯỢNG: Chỉ lấy 100 từ có xác suất cao nhất
        topk_values, topk_indices = torch.topk(logits, k=100, dim=-1)
        
    # Nén lại thành một dictionary và chuyển sang CPU để lưu ra ổ cứng
    batch_data = {
        "input_ids": masked_inputs.cpu(),       # Đề bài đã đục lỗ (Trò bắt buộc phải giải đúng đề này)
        "topk_values": topk_values.cpu(),       # Điểm số của Top 100
        "topk_indices": topk_indices.cpu(),     # ID của Top 100
    }
    
    # Lưu file tensor (.pt)
    torch.save(batch_data, f"./offline_logits/batch_{step:04d}.pt")

print("Hoàn tất! Bạn có thể xóa Thầy khỏi bộ nhớ.")
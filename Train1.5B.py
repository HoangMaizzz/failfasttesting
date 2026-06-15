import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ==========================================
# 1. CẤU HÌNH (Dành cho L4 thảnh thơi)
# ==========================================
STUDENT_PATH = "./LLaDA-1.5B-Drafter-Clean"
BATCH_SIZE = 8                  # RAM dư dả, kéo Batch lên 8 chạy cho xé gió
MASKING_RATIO = 0.30            
MAX_LENGTH = 128                

# ==========================================
# 2. CHỈ TẢI MỖI CON TRÒ (Chiếm vỏn vẹn ~3GB)
# ==========================================
print("Đang tải Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
mask_id = tokenizer.mask_token_id

print("Đang tải Trò 1.5B lên GPU...")
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
student.config.use_cache = False
if hasattr(student, "gradient_checkpointing_enable"):
    student.gradient_checkpointing_enable()

optimizer = torch.optim.AdamW(student.parameters(), lr=2e-4)

# ==========================================
# 3. CHUẨN BỊ DỮ LIỆU ĐỤC LỖ TỰ ĐỘNG
# ==========================================
print("Đang tải sách giáo khoa Wikipedia...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [t for t in dataset['text'] if len(t.strip()) > 50][:4000]

def collate_fn(batch):
    inputs = tokenizer(batch, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    # Tạo nhãn cứng (Ground Truth) để đối chiếu
    labels = input_ids.clone()
    
    # Đục lỗ 30%
    rand_matrix = torch.rand(input_ids.shape)
    mask_condition = (rand_matrix < MASKING_RATIO) & (inputs["attention_mask"] == 1)
    
    masked_input_ids = input_ids.clone()
    masked_input_ids[mask_condition] = mask_id
    
    # Tuyệt chiêu: Bỏ qua không chấm điểm những chữ không bị đục lỗ (-100 là mã bỏ qua của PyTorch)
    labels[~mask_condition] = -100
    
    return {
        "input_ids": masked_input_ids.to("cuda"),
        "labels": labels.to("cuda")
    }

dataloader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

# ==========================================
# 4. VÒNG LẶP TỰ CHỮA LÀNH
# ==========================================
print(f"\nBắt đầu tự chữa lành | Batch: {BATCH_SIZE} | Mask: 30%")
os.makedirs("./checkpoints", exist_ok=True)
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

student.train()
progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Đang trị liệu")

for step, batch in progress_bar:
    optimizer.zero_grad()
    
    # Trò tự làm bài với các khoảng trống [MASK]
    outputs = student(input_ids=batch["input_ids"])
    logits = outputs.logits
    
    # So sánh trực tiếp chữ Trò điền với đáp án thật trong sách giáo khoa
    loss = loss_fct(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))
    
    loss.backward()
    optimizer.step()
    
    # Cập nhật hiển thị và dọn rác
    current_loss = loss.item()
    progress_bar.set_postfix({"Loss": f"{current_loss:.4f}"})
    del outputs, logits, loss
    
    # Lưu Checkpoint mỗi 100 bước
    if step > 0 and step % 100 == 0:
        student.save_pretrained(f"./checkpoints/drafter_step_{step}")

print("\nChữa lành hoàn tất!")
merged_model = student.merge_and_unload()
merged_model.save_pretrained("./LLaDA-1.5B-Healed-Drafter")
tokenizer.save_pretrained("./LLaDA-1.5B-Healed-Drafter")
print("Mô hình đã phục hồi nhân phẩm thành công tại ./LLaDA-1.5B-Healed-Drafter")
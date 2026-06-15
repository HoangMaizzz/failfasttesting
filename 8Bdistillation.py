import torch
from transformers import AutoModelForCausalLM, AutoConfig

teacher_id = "GSAI-ML/LLaDA-8B-Instruct"

print("1. Đưa Thầy 8B lên bàn mổ (Loading Teacher)...")
# Phải load Thầy trước để an toàn không bị ảnh hưởng bởi đoạn Hack phía sau
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_id, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("2. Lên danh sách các Lớp cần giữ lại...")
layers_to_keep = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31]
num_kept_layers = len(layers_to_keep)

print("3. Khởi tạo một Thể xác 1.5B trống rỗng (Loading empty Student)...")
student_config = AutoConfig.from_pretrained(teacher_id, trust_remote_code=True)

# --- THE MAGIC HACK V2: THÁO Ổ KHÓA ---
config_class = type(student_config) # Lấy class gốc LLaDAConfig

# Nếu tác giả khóa biến này bằng @property, ta xóa lệnh khóa đó đi
if hasattr(config_class, 'num_hidden_layers'):
    delattr(config_class, 'num_hidden_layers')

# Bây giờ cửa đã mở, gán 12 lớp thoải mái vào chính chủ LLaDAConfig
student_config.num_hidden_layers = num_kept_layers

# Quét luôn các biến dự phòng (nếu có)
if hasattr(student_config, "n_layers"):
    student_config.n_layers = num_kept_layers
if hasattr(student_config, "num_layers"):
    student_config.num_layers = num_kept_layers
# ---------------------------------------

# Khởi tạo Trò (Lúc này Hugging Face sẽ duyệt cái rụp vì đúng chuẩn LLaDAConfig)
student_model = AutoModelForCausalLM.from_config(
    student_config, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("4. Bắt đầu phẫu thuật cấy ghép (Weight Transfer)...")
state_dict_student = student_model.state_dict()
state_dict_teacher = teacher_model.state_dict()

# Quét qua toàn bộ các tế bào của ông Thầy
for key in state_dict_teacher.keys():
    # Giữ NGUYÊN VẸN Não bộ (Embedding) và Thanh quản (LM Head + Final Norm)
    if "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
        state_dict_student[key] = state_dict_teacher[key]

    # Chọn lọc các Lớp Transformer (Khúc ruột)
    elif "model.layers" in key:
        layer_idx = int(key.split(".")[2])
        if layer_idx in layers_to_keep:
            new_idx = layers_to_keep.index(layer_idx)
            new_key = key.replace(f"model.layers.{layer_idx}", f"model.layers.{new_idx}")
            state_dict_student[new_key] = state_dict_teacher[key]

print("5. Khâu vết mổ (Loading injected weights)...")
student_model.load_state_dict(state_dict_student)

print("6. Lưu thành quả Drafter 1.5B (Saving model)...")
student_model.save_pretrained("./LLaDA-1.5B-Drafter-Init")
student_config.save_pretrained("./LLaDA-1.5B-Drafter-Init")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)
tokenizer.save_pretrained("./LLaDA-1.5B-Drafter-Init")

print("Hoàn tất! Con Drafter 1.5B của bạn đã ra lò.")
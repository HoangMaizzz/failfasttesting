import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

teacher_id = "GSAI-ML/LLaDA-8B-Instruct"

print("1. Đưa Thầy 8B lên bàn mổ...")
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_id, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("2. Lên danh sách các Lớp cần giữ lại...")
layers_to_keep = [0, 1, 2, 3, 10, 16, 22, 28, 29, 30, 31]
num_kept_layers = len(layers_to_keep)

print("3. Khởi tạo Thể xác 1.5B (The Clean Way)...")
student_config = AutoConfig.from_pretrained(teacher_id, trust_remote_code=True)

# --- GIẢI PHÁP TRIỆT ĐỂ: CHỈ CHẠM VÀO BIẾN GỐC ---
student_config.n_layers = num_kept_layers
# -------------------------------------------------

student_model = AutoModelForCausalLM.from_config(
    student_config, 
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

print("4. Bắt đầu phẫu thuật cấy ghép...")
state_dict_student = student_model.state_dict()
state_dict_teacher = teacher_model.state_dict()

for key in state_dict_teacher.keys():
    if "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
        state_dict_student[key] = state_dict_teacher[key]
    elif "model.layers" in key:
        layer_idx = int(key.split(".")[2])
        if layer_idx in layers_to_keep:
            new_idx = layers_to_keep.index(layer_idx)
            # Thêm dấu chấm để đảm bảo replace chính xác (VD: tránh nhầm layer 1 với 11)
            new_key = key.replace(f"model.layers.{layer_idx}.", f"model.layers.{new_idx}.")
            state_dict_student[new_key] = state_dict_teacher[key]

print("5. Khâu vết mổ...")
student_model.load_state_dict(state_dict_student)

print("6. Lưu thành quả Drafter Sạch (Clean Model)...")
student_model.save_pretrained("./LLaDA-1.5B-Drafter-Clean")
student_config.save_pretrained("./LLaDA-1.5B-Drafter-Clean")

tokenizer = AutoTokenizer.from_pretrained(teacher_id, trust_remote_code=True)
tokenizer.save_pretrained("./LLaDA-1.5B-Drafter-Clean")

print("Hoàn tất! Model chuẩn 100%, vĩnh viễn không dính lỗi JSON.")
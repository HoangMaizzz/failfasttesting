import torch
from transformers import AutoModelForCausalLM, AutoConfig

print("1. Đưa Thầy 8B lên bàn mổ (Loading Teacher)...")
# Giả sử đường dẫn tới LLaDA 8B. Sử dụng bfloat16 để tiết kiệm RAM.
teacher_id = "GSAI-ML/LLaDA-8B" 
teacher_model = AutoModelForCausalLM.from_pretrained(teacher_id, torch_dtype=torch.bfloat16)

print("2. Lên danh sách các Lớp cần giữ lại...")
# Giữ lại 12 lớp trải đều từ 0 đến 31
layers_to_keep = [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 31]
num_kept_layers = len(layers_to_keep)

print("3. Khởi tạo một Thể xác 1.5B trống rỗng (Loading empty Student)...")
# Sao chép Config của Thầy, nhưng ép cấu hình số lớp xuống còn 12
student_config = AutoConfig.from_pretrained(teacher_id)
student_config.num_hidden_layers = num_kept_layers

# Khởi tạo mô hình Trò (Lúc này trọng số bên trong đang là Random/Rác)
student_model = AutoModelForCausalLM.from_config(student_config, torch_dtype=torch.bfloat16)

print("4. Bắt đầu phẫu thuật cấy ghép (Weight Transfer)...")
state_dict_student = student_model.state_dict()
state_dict_teacher = teacher_model.state_dict()

# Quét qua toàn bộ các tế bào (tham số) của ông Thầy
for key in state_dict_teacher.keys():
    
    # a. Giữ NGUYÊN VẸN Não bộ (Embedding) và Thanh quản (LM Head + Final Norm)
    if "embed_tokens" in key or "lm_head" in key or "model.norm" in key:
        state_dict_student[key] = state_dict_teacher[key]

    # b. Chọn lọc các Lớp Transformer (Khúc ruột)
    elif "model.layers" in key:
        # Tên key có dạng: 'model.layers.5.self_attn.q_proj.weight'
        # Ta cần bóc số '5' ra để kiểm tra
        layer_idx = int(key.split(".")[2])
        
        if layer_idx in layers_to_keep:
            # Map index của Thầy sang index của Trò (Trò chỉ có từ 0 -> 11)
            # Ví dụ: Lớp 5 của Thầy sẽ trở thành Lớp 2 của Trò
            new_idx = layers_to_keep.index(layer_idx)
            new_key = key.replace(f"model.layers.{layer_idx}", f"model.layers.{new_idx}")
            
            # Bơm máu (trọng số) vào cơ thể Trò
            state_dict_student[new_key] = state_dict_teacher[key]

print("5. Khâu vết mổ (Loading injected weights)...")
student_model.load_state_dict(state_dict_student)
import torch
from quick_model_loader.model_loader import QuickModelLoader, pipeline_storage_transfer
from quick_model_loader.meta import TensorsMeta, ModelMeta
from quick_model_loader.allocator import Device
import io
import time
from safetensors import safe_open

print("Start init quick model loader...")
model_loader = QuickModelLoader(
    cuda_storage_sizes=[16 * 1024 * 1024 * 1024],
    pinned_buffer_size=16 * 1024 * 1024 * 1024,
    models_storage_size=48 * 1024 * 1024 * 1024,
)

print("Start cache models...")

llama2_7b_path = "/home/models/Llama-2-7b-hf"

qwen_7b_path = "/home/models/Qwen-7B-Chat"

print("Cache Llama2 7b model...")
model_loader.cache_model(llama2_7b_path)
print("Cache Qwen 7b model...")
model_loader.cache_model(qwen_7b_path)

print("Start loading Llama2 7b model to gpu...")
model_loader.load_model(llama2_7b_path)
# for name, weight in model_loader.loaded_model.to_tensors():
#     if name == "model.layers.15.self_attn.v_proj.weight":
#         print(weight)
#     pass

# with safe_open(llama2_7b_root_dir + "/model-00001-of-00002.safetensors", "pt") as f:
#     print(f.get_tensor("model.layers.15.self_attn.v_proj.weight"))

print("Start switch Qwen 7b model to gpu...")
start_time = time.time()
model_loader.load_model(qwen_7b_path)
for name, _ in model_loader.loaded_model.to_tensors():
    pass
end_time = time.time()
elapsed = end_time - start_time
print(elapsed)


start_time = time.time()
weights_cuda = []
for name, weight in model_loader.model_to_tensors(llama2_7b_path, "cuda"):
    weights_cuda.append(weight)

end_time = time.time()
elapsed = end_time - start_time
print(elapsed)

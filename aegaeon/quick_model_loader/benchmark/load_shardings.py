import torch
import time
from quick_model_loader.model_loader import QuickModelLoader
from quick_model_loader.meta import CheckPointConfig, ParallelType
from safetensors.torch import save_file


_GB = 1024 * 1024 * 1024

print("Start init quick model loader...")
model_loader = QuickModelLoader(
    cuda_storage_sizes=[48 * _GB, 48 * _GB, 48 * _GB, 48 * _GB],
    pinned_buffer_size=16 * _GB,
    models_storage_size=192 * _GB,
)

print("Start cache models with checkpoints...")
tp_size = 4
model_path = "/home/models/Qwen2-72B"
checkpoint_config = CheckPointConfig("vllm", ParallelType.TP, tp_size)
print("Cache model...")
model_loader.cache_model(model_path, checkpoint_config)

sharding_content_cpu = model_loader.cached_models[model_path].sharding_contents[1]
start_time = time.time()
shared_list = []
for name, weight in sharding_content_cpu.to_tensors():
    weight.share_memory_()
    shared_list.append(shared_list)
end_time = time.time()
print(end_time - start_time)


print("Load model...")
start_time = time.time()
model_loader.load_model(model_path, checkpoint_config)
end_time = time.time()
print(end_time - start_time)
# save_file({"storage": tensor}, "sharding1.safetensors")

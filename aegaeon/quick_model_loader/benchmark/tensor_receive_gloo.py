import torch
import torch.distributed as dist
import time

dist.init_process_group(
    backend="gloo", init_method="tcp://127.0.0.1:12345", rank=1, world_size=2
)

tensor_size = 14 * 1024 * 1024 * 1024

received_tensor = torch.zeros(tensor_size, dtype=torch.uint8)
start_time = time.time()
dist.recv(tensor=received_tensor, src=0)
end_time = time.time()
elapsed = end_time - start_time
print(elapsed)
bandwidth = tensor_size / elapsed / (1024 * 1024 * 1024)
print(f"Bandwidth: {bandwidth} GB/s")
print(received_tensor)

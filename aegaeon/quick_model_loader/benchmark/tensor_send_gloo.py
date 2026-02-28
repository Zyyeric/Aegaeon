import torch
import torch.distributed as dist

dist.init_process_group(
    backend="gloo", init_method="tcp://0.0.0.0:12345", rank=0, world_size=2
)

tensor_size = 14 * 1024 * 1024 * 1024

tensor = torch.ones(tensor_size, dtype=torch.uint8)
dist.send(tensor=tensor, dst=1)

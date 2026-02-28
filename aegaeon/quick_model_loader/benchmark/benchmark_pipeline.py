import torch
import time
from quick_model_loader.model_loader import pipeline_storage_transfer


if __name__ == "__main__":
    cuda_storage = torch.zeros(
        8 * 1024 * 1024 * 1024, dtype=torch.uint8, device="cuda"
    ).untyped_storage()

    pinned_buffer = torch.zeros(
        8 * 1024 * 1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=True
    ).untyped_storage()

    cpu_storage = torch.zeros(
        8 * 1024 * 1024 * 1024, dtype=torch.uint8, device="cpu", pin_memory=False
    ).untyped_storage()

    start_time = time.time()
    pipeline_storage_transfer(cuda_storage, pinned_buffer, cpu_storage)
    end_time = time.time()
    elpased = end_time - start_time
    print(elpased)

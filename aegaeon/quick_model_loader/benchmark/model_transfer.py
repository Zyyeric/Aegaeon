import torch
import torch.multiprocessing as mp
import time
import gc
from safetensors.torch import save_file

from quick_model_loader.model_loader import QuickModelLoader
from quick_model_loader.meta import CheckPointConfig, ParallelType
from quick_model_loader.handle import StorageHandle, ShardingHandle

_GB = 1024 * 1024 * 1024


def child_process(queue):
    for i in range(4):
        sharding_handle: ShardingHandle = queue.get()
        sharding_content = sharding_handle.to_sharding_content()
        tensors_dict = {}
        for name, weight in sharding_content.to_tensors():
            tensors_dict[name] = weight
        names = sorted(tensors_dict.keys())
        with open(f"slave{i}_tensors2.txt", "w") as file:
            for name in names:
                print(name, tensors_dict[name], file=file)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    queue = mp.Queue()

    print("Start init quick model loader...")
    model_loader = QuickModelLoader(
        cuda_storage_sizes=[16 * _GB, 16 * _GB, 16 * _GB, 16 * _GB],
        pinned_buffer_size=16 * _GB,
        models_storage_size=48 * _GB,
    )

    tp_size = 4
    qwen_7b_path = "/home/models/Qwen-7B-Chat"
    checkpoint_config = CheckPointConfig("vllm", ParallelType.TP, tp_size)

    print("Cache Qwen 7b model...")
    model_loader.cache_model(qwen_7b_path, checkpoint_config)

    print("Load Qwen 7b model...")
    model_loader.load_model(qwen_7b_path, checkpoint_config)

    p = mp.Process(target=child_process, args=(queue,))
    p.start()

    for i in range(4):
        sharding_content = model_loader.loaded_model.sharding_contents[i]
        tensors_dict = {}
        for name, weight in sharding_content.to_tensors():
            tensors_dict[name] = weight
        names = sorted(tensors_dict.keys())
        with open(f"master{i}_tensors2.txt", "w") as file:
            for name in names:
                print(name, tensors_dict[name], file=file)
        sharding_handle = ShardingHandle(sharding_content)
        queue.put(sharding_handle)

    p.join()

    pass

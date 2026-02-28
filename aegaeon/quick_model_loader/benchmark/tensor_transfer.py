import torch
import cupy
import time
import torch.multiprocessing as mp

from quick_model_loader.model_loader import QuickModelLoader
from quick_model_loader.meta import CheckPointConfig, ParallelType


class TensorHandle:
    def __init__(self, tensor: torch.Tensor) -> None:
        assert "cuda" in tensor.device.type

        tensor_uint8 = tensor.view(dtype=torch.uint8)
        cupy_array = cupy.asarray(tensor_uint8)

        self.handle = cupy.cuda.runtime.ipcGetMemHandle(cupy_array.data.ptr)
        self.cupy_shape = cupy_array.shape
        self.cupy_dtype = cupy_array.dtype
        self.cupy_nbytes = cupy_array.nbytes

        self.tensor_dtype = tensor.dtype
        self.tensor_shape = tensor.shape
        self.tensor_device = tensor.device

    def to_tensor(self) -> torch.Tensor:
        start_time = time.time()
        shared_data = cupy.cuda.runtime.ipcOpenMemHandle(self.handle)
        end_time = time.time()
        print(end_time - start_time)
        memory = cupy.cuda.UnownedMemory(shared_data, size=self.cupy_nbytes, owner=None)
        memory_ptr = cupy.cuda.MemoryPointer(memory, offset=0)
        shared_cupy_array = cupy.ndarray(
            shape=self.cupy_shape, dtype=self.cupy_dtype, memptr=memory_ptr
        )

        tensor_unint8 = torch.as_tensor(
            shared_cupy_array, dtype=torch.uint8, device=self.tensor_device
        )
        tensor = tensor_unint8.view(dtype=self.tensor_dtype).reshape(self.tensor_shape)
        return tensor


def child_process(queue):
    tensor0_handle = queue.get()
    tensor0 = tensor0_handle.to_tensor()
    print(tensor0)

    tensor1_handle = queue.get()
    tensor1 = tensor1_handle.to_tensor()
    print(tensor1)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    queue = mp.Queue()

    tensor0 = torch.rand((3, 3), device="cuda:0", dtype=torch.float16)
    print(tensor0)
    tensor0_handle = TensorHandle(tensor0)

    tensor1 = torch.rand((3, 3), device="cuda:1", dtype=torch.float16)
    print(tensor1)
    tensor1_handle = TensorHandle(tensor1)

    p = mp.Process(target=child_process, args=(queue,))
    p.start()

    queue.put(tensor0_handle)
    queue.put(tensor1_handle)
    p.join()

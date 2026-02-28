import cupy as cp
import torch
import torch.multiprocessing as mp


def child_process(queue):
    import cupy as cp  # 在子进程中重新导入cupy，以确保正确的上下文

    # 从父进程接收IPC句柄
    handle, shape, dtype, nbytes = queue.get()

    # 从句柄打开共享内存
    shared_data = cp.cuda.runtime.ipcOpenMemHandle(handle)
    mem = cp.cuda.UnownedMemory(shared_data, size=nbytes, owner=None)
    memptr = cp.cuda.MemoryPointer(mem, offset=0)
    shared_tensor = cp.ndarray(shape=shape, dtype=dtype, memptr=memptr)
    out = torch.as_tensor(shared_tensor, dtype=torch.uint8, device="cuda")
    print(out.shape)
    x = out.view(dtype=torch.bfloat16, shape=(3, 3))

    # 读取共享数据并进行一些操作
    print("Shared tensor in child process:", x)

    # 修改共享数据
    x += 10
    print("Modified tensor in child process:", x)
    print(x.device)


if __name__ == "__main__":
    mp.set_start_method("spawn")  # 使用spawn来创建子进程
    queue = mp.Queue()

    # 创建一个CuPy数组
    x = torch.ones((3, 3), dtype=torch.bfloat16, device="cuda")
    print(x.data_ptr())
    y = x.view(dtype=torch.uint8)
    print(y.data_ptr())
    print(y.device)
    cp_array = cp.asarray(y)

    p = mp.Process(target=child_process, args=(queue,))
    p.start()

    handle = cp.cuda.runtime.ipcGetMemHandle(cp_array.data.ptr)
    queue.put((handle, cp_array.shape, cp_array.dtype, cp_array.nbytes))

    # 等待子进程执行结束
    p.join()

    # 查看被修改后的数组
    print("Modified tensor in parent process:", x)

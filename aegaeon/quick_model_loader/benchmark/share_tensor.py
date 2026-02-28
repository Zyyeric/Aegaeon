import torch
import torch.multiprocessing as mp


def worker(tensor, index, value):
    # Each worker modifies the shared tensor
    tensor[index] = value
    print(f"Process {index}: Tensor after update: {tensor}")


if __name__ == "__main__":
    # Create a tensor and move it to shared memory
    shared_tensor = torch.zeros(4)  # Create a tensor with 4 elements
    shared_tensor.share_memory_()  # Move the tensor to shared memory

    # Create a list of processes
    processes = []
    for i in range(4):
        # Spawn a new process that accesses the shared tensor
        p = mp.Process(target=worker, args=(shared_tensor, i, i + 1))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("Final tensor in shared memory:", shared_tensor)

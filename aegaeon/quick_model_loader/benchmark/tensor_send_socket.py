import torch
import os
import ctypes
import psutil
import socket

storage_size = 14 * 1024 * 1024 * 1024


def print_memory_usage():
    print(
        "Memory Usage:%.4f GB"
        % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024)
    )


tensor = torch.ones(storage_size, dtype=torch.uint8)
tensor_storage = tensor.untyped_storage()

# Create a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(("33.254.161.84", 12345))  # Replace with the actual receiver's IP and port

ptr = tensor_storage.data_ptr()

data_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char * storage_size))
sock.sendall(data_ptr.contents.raw)
sock.close()

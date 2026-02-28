import socket
import torch
import time

storage_size = 14 * 1024 * 1024 * 1024
chunk_size = 1024 * 1024

tensor_storage = torch.zeros(storage_size, dtype=torch.uint8).untyped_storage()

# Create a socket and listen for the incoming connection
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("0.0.0.0", 12345))  # Replace with actual receiver IP and port
sock.listen(1)
print("Listen...")
conn, addr = sock.accept()
print(f"Accept connection from {addr}")

start_time = time.time()
received_size = 0
while received_size < storage_size:
    received_data = conn.recv(min(chunk_size, storage_size - received_size))
    if not received_data:
        break
    received_storage = torch.UntypedStorage.from_buffer(
        received_data, dtype=torch.uint8
    )
    tensor_storage[received_size : received_size + len(received_data)].copy_(
        received_storage
    )
    received_size += len(received_data)
end_time = time.time()
elapsed = end_time - start_time
print(elapsed)
bandwidth = storage_size / elapsed / (1024 * 1024 * 1024)
print(f"Bandwidth: {bandwidth} GB/s")

tensor = (
    torch.tensor([], dtype=torch.uint8, device=tensor_storage.device)
    .set_(tensor_storage)
    .view(dtype=torch.uint8)
)
print(tensor)
print(tensor.shape)

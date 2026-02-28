import torch
from safetensors.torch import save_file


def create_test_model0():
    tensor0 = torch.tensor([0], dtype=torch.float32)
    tensor1 = torch.tensor([1, 1], dtype=torch.float32)

    tensor2 = torch.tensor([[2, 2], [2, 2]], dtype=torch.float16)
    tensor3 = torch.tensor([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=torch.float16)

    tensors_dict_01 = {"tensor0": tensor0, "tensor1": tensor1}
    tensors_dict_23 = {"tensor2": tensor2, "tensor3": tensor3}

    save_file(tensors_dict_01, "model0/model-00001-of-00002.safetensors")
    save_file(tensors_dict_23, "model0/model-00002-of-00002.safetensors")


def create_test_model1():
    tensor0 = torch.tensor([[2, 2], [2, 2]], dtype=torch.float16)
    tensor1 = torch.tensor([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=torch.float16)

    tensor2 = torch.tensor([0], dtype=torch.float32)
    tensor3 = torch.tensor([1, 1], dtype=torch.float32)

    tensors_dict_01 = {"tensor0": tensor0, "tensor1": tensor1}
    tensors_dict_23 = {"tensor2": tensor2, "tensor3": tensor3}

    save_file(tensors_dict_01, "model1/model-00001-of-00002.safetensors")
    save_file(tensors_dict_23, "model1/model-00002-of-00002.safetensors")


def create_test_model2():
    tensor0 = torch.tensor(list(range(16)), dtype=torch.float32).reshape((4, 4))
    tensor1 = torch.tensor(list(range(9)), dtype=torch.float16).reshape((3, 3))

    tensor2 = torch.tensor(list(range(8)), dtype=torch.float32).reshape((2, 2, 2))
    tensor3 = torch.tensor(list(range(27)), dtype=torch.float16).reshape((3, 3, 3))

    tensors_dict_01 = {"tensor0": tensor0, "tensor1": tensor1}
    tensors_dict_23 = {"tensor2": tensor2, "tensor3": tensor3}

    save_file(tensors_dict_01, "model2/model-00001-of-00002.safetensors")
    save_file(tensors_dict_23, "model2/model-00002-of-00002.safetensors")


def create_test_model3():
    tensor0 = torch.rand((1), dtype=torch.float32)
    tensor1 = torch.rand((1, 1), dtype=torch.float32)

    tensor2 = torch.rand((2, 2), dtype=torch.float32)
    tensor3 = torch.rand((3, 3), dtype=torch.float16)

    tensor4 = torch.rand((4, 4), dtype=torch.float32)
    tensor5 = torch.rand((5, 5), dtype=torch.float32)

    tensors_dict_01 = {"tensor0": tensor0, "tensor1": tensor1}
    tensors_dict_23 = {"tensor2": tensor2, "tensor3": tensor3}
    tensors_dict_45 = {"tensor4": tensor4, "tensor5": tensor5}

    save_file(tensors_dict_01, "model3/model-00001-of-00003.safetensors")
    save_file(tensors_dict_23, "model3/model-00002-of-00003.safetensors")
    save_file(tensors_dict_45, "model3/model-00003-of-00003.safetensors")


if __name__ == "__main__":
    create_test_model0()
    create_test_model1()
    create_test_model2()
    create_test_model3()

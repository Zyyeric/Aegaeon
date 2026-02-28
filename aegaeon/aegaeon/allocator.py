from typing import Union, Optional, Tuple
import torch

from aegaeon.utils import VRAM_ALIGN, MB, DTYPE_SIZE, prod


def _aligned(addr: int) -> int:
    aligned = addr + VRAM_ALIGN - 1
    return aligned - aligned % VRAM_ALIGN


class Allocator:
    """
    A simplistic GPU memory allocator for managing tensor allocations
    without actually involving CUDA memory management.
    """

    storage: torch.UntypedStorage
    device: torch.device

    def __init__(self, storage: torch.UntypedStorage) -> None:
        assert (
            storage.device.type == "cuda"
        ), "only CUDA devices are supported for Aegaeon Allocator"

        self.device = storage.device
        self.storage = storage
        self.top = 0
        self.bottom = []
        self._size = storage.nbytes()

    @classmethod
    def with_size(
        cls,
        size: int,
        device: torch.device,
    ):
        storage = torch.empty(size, dtype=torch.uint8, device=device).untyped_storage()
        return cls(storage)

    def can_allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        aligned: bool = True,
    ) -> bool:
        alloc_size = prod(shape) * DTYPE_SIZE[dtype]
        start = _aligned(self.top) if aligned else self.top
        end = start + alloc_size
        if end > self.size:
            return False
        return True

    def allocate(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        raw: bool = False,
        aligned: bool = True,
    ) -> Union[torch.Tensor, torch.UntypedStorage]:
        alloc_size = prod(shape) * DTYPE_SIZE[dtype]
        start = _aligned(self.top) if aligned else self.top
        end = start + alloc_size
        if end > self._size:
            raise ValueError(
                f"Insufficient memory. "
                f"Tried to allocate {alloc_size/MB:.2f} MB, "
                f"available {(self._size - start)/MB:.2f} MB."
            )

        storage = self.storage[start:end]
        self.top = end

        if raw:
            return storage
        tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(storage)
            .view(dtype=dtype)
            .reshape(shape)
        )
        return tensor

    def push(self):
        self.bottom.append(self.top)

    def pop(self):
        self.top = self.bottom.pop()

    def clear(self):
        self.top = 0

    def size(self) -> int:
        return self._size

    def allocated_size(self) -> int:
        return self.top

    def free_size(self) -> int:
        return self._size - self.top


_alloc: Optional[Allocator] = None


def initialize_alloc(size: int, device: torch.device):
    global _alloc
    assert _alloc is None, "initialize_alloc should be invoked only once"
    _alloc = Allocator.with_size(size, device)


def get_alloc() -> Allocator:
    global _alloc
    assert _alloc is not None, "initialize_alloc has not been invoked"
    return _alloc

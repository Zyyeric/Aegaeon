import contextlib
import os
from typing import Iterator

import torch


def enabled() -> bool:
    return os.environ.get("AEGAEON_ENABLE_NVTX", "0") == "1"


@contextlib.contextmanager
def range_(message: str) -> Iterator[None]:
    if not enabled() or not torch.cuda.is_available():
        yield
        return
    torch.cuda.nvtx.range_push(message)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


from typing import Optional

from .loader import BaseLoader, DefaultLoader, QuickLoader
from .handle import ModelHandle
from .cache import QuickCache


def get_model_loader(quick_loader: Optional[QuickLoader]) -> BaseLoader:
    """Get a model loader based on the load format."""
    if quick_loader is not None:
        return quick_loader
    return DefaultLoader()


__all__ = [
    "QuickLoader",
    "DefaultLoader",
    "get_model_loader",
    "ModelHandle",
    "QuickCache",
]

import torch
import os
import glob
import ctypes
import sys
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Generator, Type, Dict, TYPE_CHECKING
from vllm.model_executor.model_loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
)
import contextlib

from aegaeon.models import ModelRegistry
from aegaeon.utils import CudaRTLibrary

if TYPE_CHECKING:
    from aegaeon.config import ModelConfig

from .meta import (
    TensorsContent,
    ModelContent,
    CheckPointConfig,
    create_tensor_from_storage,
)
from .allocator import (
    SliceInfo,
)

from aegaeon.config import ModelConfig, QuickLoaderConfig
from aegaeon.logger import init_logger
from aegaeon.allocator import Allocator, get_alloc
from aegaeon.utils import GB

logger = init_logger(__name__)


class BaseLoader(ABC):
    """Base class for model loaders."""

    @abstractmethod
    def load_model(
        self,
        *,
        model_config: "ModelConfig",
        device: int,
        **extra_model_kwargs,
    ) -> torch.nn.Module:
        """Load a model with the given configurations."""
        ...


class DefaultLoader(BaseLoader):
    """Model loader that can load different file types from disk."""

    def __init__(self):
        pass

    def _prepare_weights(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Tuple[str, List[str], bool]:
        """Prepare weights for the model."""

        is_local = os.path.isdir(model_name_or_path)
        use_safetensors = False
        # Some quantized models use .pt files for storing the weights.
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if not is_local:
            raise ValueError(f"Non-local model: {model_name_or_path}")
        else:
            hf_folder = model_name_or_path

        hf_weights_files: List[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            # For models like Mistral-7B-Instruct-v0.3
            # there are both sharded safetensors files and a consolidated
            # safetensors file. Using both breaks.
            # Here, we download the `model.safetensors.index.json` and filter
            # any files not found in the index.
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files, hf_folder
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self, model_name_or_path: str, revision: Optional[str], fall_back_to_pt: bool
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get an iterator for the model weights based on the load format."""
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            model_name_or_path, revision, fall_back_to_pt
        )
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(hf_weights_files)
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files)

        return weights_iterator

    def load_model(
        self,
        *args,
        model_config: ModelConfig,
        device: torch.device,
        **extra_model_kwargs,
    ) -> torch.nn.Module:
        with set_default_torch_dtype(model_config.torch_dtype):
            with device:
                model_class = get_model_architecture(model_config)[0]
                model = model_class(config=model_config.hf_config, **extra_model_kwargs)
            model.load_weights(
                self._get_weights_iterator(
                    model_config.model.path(),
                    None,  # revision
                    fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
                ),
            )

            for _, module in model.named_modules():
                quant_method = getattr(module, "quant_method", None)
                if quant_method is not None:
                    quant_method.process_weights_after_loading(module)
                # FIXME: Remove this after Mixtral is updated
                # to use quant_method.
                if hasattr(module, "process_weights_after_loading"):
                    module.process_weights_after_loading()
        return model.eval()


class ManagedParameter(torch.nn.Parameter):
    """Parameter weights that are set later at copying."""

    def __new__(cls, data=None, requires_grad=False):
        alloc = get_alloc()
        slice_data = alloc.allocate(data.shape, data.dtype)
        obj = super(ManagedParameter, cls).__new__(cls, slice_data, requires_grad)
        return obj

    # def copy_(self, src: torch.Tensor, non_blocking: bool = False) -> torch.Tensor:
    #     return self.set_(src.untyped_storage()).view(src.dtype).reshape(src.shape)


class QuickLoader(BaseLoader):
    """Model loader that uses snapshots from QuickCache."""

    def __init__(
        self,
        cudart: CudaRTLibrary,
        config: QuickLoaderConfig,
    ):
        if not os.getenv("FAST_SWITCH") == "1":
            raise ValueError("QuickLoader should not be initialized.")
        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA decices avaliable!")

        # Monkey-patch `nn.Parameter` as `ManagedParameter`
        torch.nn.Parameter = ManagedParameter
        torch.nn.parameter.Parameter = ManagedParameter
        for module in sys.modules.values():
            if hasattr(module, "Parameter"):
                setattr(module, "Parameter", ManagedParameter)

        model_cache_filename = config.model_cache_filename
        if not os.path.isfile(model_cache_filename):
            raise ValueError(f"File {config.model_cache_filename} not found")

        model_cache = torch.from_file(
            model_cache_filename,
            shared=True,
            size=config.model_cache_size,
            dtype=torch.uint8,
        )

        cudart.cudaHostRegister(
            ctypes.c_void_p(model_cache.data_ptr()), model_cache.nbytes
        )
        assert model_cache.is_shared() and model_cache.is_pinned()

        model_cache[:] = model_cache[:]

        self.storage = model_cache.untyped_storage()
        self.model_cache_start = model_cache.data_ptr()
        self.model_contents: Dict[str, ModelContent] = {}
        for model_name, handle in config.model_cache_snapshot.items():
            self.model_contents[model_name] = handle.to_model(self.model_cache_start)

        # self.pinned_allocator = PinnedAllocator(config.pinned_buffer_size)
        torch.cuda.synchronize()

    def load_model(
        self,
        *, 
        model_config: ModelConfig,
        device: torch.device,
        **extra_model_kwargs,
    ) -> torch.nn.Module:
        with set_default_torch_dtype(model_config.torch_dtype):
            with device:
                model_class = get_model_architecture(model_config)[0]
                model = model_class(config=model_config.hf_config, **extra_model_kwargs)
                params_dict = dict(model.named_parameters())

                for name, param in params_dict.items():
                    param.detach_()

                tensors_generator = self.load_model_to_tensors(
                    model_config.model.path()
                )
                model.load_weights(tensors_generator)

                # TODO: the following is only possible with patched vLLM+QuickLoader,
                # which can save weights that are loader-friendly.
                
                # for name, weight in tensors_generator:
                #     param = params_dict[name]
                #     if param.dtype != weight.dtype:
                #         param.set_(source=weight.view(param.dtype))
                #     else:
                #         param.set_(source=weight)

                torch.cuda.default_stream(device).synchronize()
        return model.eval()

    def load_model_to_tensors(
        self, model_name: str, checkpoint_config: Optional[CheckPointConfig] = None
    ) -> Generator[Tuple[str, torch.Tensor], None, None]:
        if not model_name in self.model_contents:
            raise NotImplementedError(f"{model_name} should be cached")
        cpu_model_content = self.model_contents[model_name]
        if checkpoint_config is None:
            checkpoint_config = cpu_model_content.checkpoint_config

        alloc = get_alloc()
        for sharding_content in cpu_model_content.sharding_contents:
            for tensor_content in sharding_content.tensors_contents:
                alloc.push()
                loaded_tensor_content = self.load_tensors_content(
                    alloc, tensor_content, sharding_content.id
                )

                tensors_storage = loaded_tensor_content.tensors_storage
                tensor_info_map = (
                    loaded_tensor_content.tensors_meta.aligned_tensor_info_map
                )
                for name, tensor_info in tensor_info_map.items():
                    tensor: torch.Tensor = create_tensor_from_storage(
                        tensors_storage, tensor_info
                    )
                    yield name, tensor
                alloc.pop()

    def load_tensors_content(
        self,
        alloc: Allocator,
        tensors_content: TensorsContent,
        sharding_id: int,
    ) -> TensorsContent:
        tensors_meta = tensors_content.tensors_meta
        size = tensors_meta.get_storage_size()
        slice_storage = alloc.allocate((size,), torch.uint8, raw=True)
        slice_info = SliceInfo.with_addr_size(slice_storage.data_ptr(), size)

        loaded_tensors_content = TensorsContent(
            slice_storage, tensors_meta, slice_info, is_global_storage=False
        )

        slice_storage.copy_(tensors_content.tensors_storage, non_blocking=True)

        # pinned_size = int(self.pinned_allocator.get_size() / parallelism)
        # pinned_offset = sharding_id * pinned_size
        # pinned_buffer = self.pinned_allocator.allocate_pinned_buffer(
        #     pinned_size, pinned_offset
        # )

        # offset = 0
        # while offset < size:
        #     bound = min(offset + pinned_size, size)
        #     copy_size = bound - offset

        #     dst = (
        #         torch.tensor([], device=loaded_tensors_content.tensors_storage.device, dtype=torch.uint8)
        #             .set_(loaded_tensors_content.tensors_storage[offset:bound])
        #     )
        #     buf = (
        #         torch.tensor([], device=pinned_buffer.device, dtype=torch.uint8)
        #             .set_(pinned_buffer[:copy_size])
        #     )
        #     src = (
        #         torch.tensor([], device=tensors_content.tensors_storage.device, dtype=torch.uint8)
        #             .set_(tensors_content.tensors_storage[offset:bound])
        #     )

        #     cuts_num = copy_size // (64 * 1024 * 1024)
        #     pipeline_storage_transfer(
        #         dst, buf, src, create_cuts(copy_size, cuts_num))
        #     offset = bound

        return loaded_tensors_content


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(
    model_config: ModelConfig,
) -> Tuple[Type[torch.nn.Module], str]:
    architectures = getattr(model_config.hf_config, "architectures", [])
    for arch in architectures:
        model_cls = ModelRegistry.load_model_cls(arch)
        if model_cls is not None:
            return (model_cls, arch)
    raise ValueError(
        f"Model architectures {architectures} are not supported for now. "
        f"Supported architectures: {ModelRegistry.get_supported_archs()}"
    )


# def pipeline_storage_transfer(
#     stream: torch.cuda.Stream,
#     cuda_storage: torch.UntypedStorage,
#     pinned_buffer: torch.UntypedStorage,
#     cpu_storage: torch.UntypedStorage,
# ):
#     if not (
#         cuda_storage.nbytes() == pinned_buffer.nbytes()
#         and pinned_buffer.nbytes() == cpu_storage.nbytes()
#     ):
#         raise MemoryError(
#             f"Pipeline transferring requires all storages and buffer have the same size!"
#             f"(cuda: {cuda_storage.nbytes()}, pinned: {pinned_buffer.nbytes()}, cpu: {cpu_storage.nbytes()})"
#         )
#     size = cuda_storage.nbytes()
#     cuts_num = size // (64 * 1024 * 1024)
#     cuts = create_cuts(size, cuts_num)

#     for start, end in cuts:
#         # src = np.frombuffer(((ctypes.c_uint8) * (end-start)).from_address(cpu_storage.data_ptr()+start), dtype=np.uint8)
#         # dst = np.frombuffer(((ctypes.c_uint8) * (end-start)).from_address(pinned_buffer.data_ptr()+start), dtype=np.uint8)
#         # np.copyto(dst, src)
#         pinned_buffer[start:end].copy_(cpu_storage[start:end])

#         cuda_storage[start:end].copy_(pinned_buffer[start:end], non_blocking=True)
#     stream.synchronize()


def create_cuts(size: int, cuts_num: int) -> List[Tuple[int, int]]:
    cuts = []
    mod = size % cuts_num
    for i in range(cuts_num):
        if i == 0:
            start = 0
        else:
            start = cuts[i - 1][1]
        if i == cuts_num - 1:
            end = size
        else:
            end = start + size // cuts_num
            if i < mod:
                end += 1
        cuts.append((start, end))
    return cuts

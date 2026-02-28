import torch
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from functools import cached_property
from dataclasses import dataclass, field

from aegaeon.utils import DeviceType, VRAM_ALIGN, prod

if TYPE_CHECKING:
    from aegaeon.models import ModelType
    from aegaeon.loader import ModelHandle

# Fixed block size for Aegaeon
BLOCK_SIZE: int = 16


@dataclass
class NodeConfig:
    """Configuration for the setup of Aegaeon on a physical node."""

    # Node ID
    # This is a logic resource for ray to place a Controller and its Workers
    # all on the same physical node.
    node_id: str

    # Number of prefill engines on the node
    num_prefill_engines: int
    # Number of decode engines on the node
    num_decode_engines: int
    # TODO: parallel config for engines goes here..

    # Policies
    prefill_disp_policy: str = "fcfs-avgload"
    prefill_sched_policy: str = "uni"

    # CPU cache related
    # Number of slabs for the CPU cache
    cpu_num_slabs: int = 128
    # Size of a slab for the CPU cache
    cpu_slab_size_bytes: int = 1024**3

    # Model cache related
    model_cache_size: int = 128 * (1024**3)
    cached_models: List["ModelType"] = field(default_factory=list)


class ParallelConfig:
    """Configuration for the distributed execution.

    Args:
        tensor_parallel_size: number of tensor parallel groups.
        tensor_parallel_rank: rank in the tensor parallel group.
        pipeline_parallel_size: number of pipeline parallel groups.
        pipeline_parallel_rank: rank in the pipeline parallel group.
    """

    def __init__(
        self,
        tensor_parallel_size: int = 1,
        tensor_parallel_rank: int = 0,
        pipeline_parallel_size: int = 1,
        pipeline_parallel_rank: int = 0,
    ) -> None:
        assert pipeline_parallel_size == 1, f"pipeline parallelism is not supported yet"
        self.tensor_parallel_size = tensor_parallel_size
        self.tensor_parallel_rank = tensor_parallel_rank
        self.pipeline_parallel_size = pipeline_parallel_size
        self.pipeline_parallel_rank = pipeline_parallel_rank

        self.world_size = pipeline_parallel_size * tensor_parallel_size
        self.use_parallel = self.world_size > 1

    def to_list(self) -> List[int]:
        return [
            self.tensor_parallel_size,
            self.tensor_parallel_rank,
            self.pipeline_parallel_size,
            self.pipeline_parallel_rank,
        ]

    def is_last_stage(self) -> bool:
        return self.pipeline_parallel_rank == self.pipeline_parallel_size - 1


class ModelConfig:
    """Configuration for the model.

    Args:
        model: Model type.
    """

    def __init__(
        self,
        model: "ModelType",
    ):
        from vllm.transformers_utils.config import get_config, get_hf_text_config
        self.model = model
        self.hf_config = get_config(model=model.path(), trust_remote_code=True)
        self.hf_text_config = get_hf_text_config(self.hf_config)

    def __repr__(self):
        return f"ModelConfig(model={self.model})"

    @cached_property
    def dtype_size(self) -> int:
        return 2

    @cached_property
    def torch_dtype(self) -> torch.dtype:
        return torch.bfloat16
        # return torch.float16

    @cached_property
    def vocab_size(self) -> int:
        return self.hf_text_config.vocab_size

    @cached_property
    def hidden_size(self) -> int:
        return self.hf_text_config.hidden_size

    @cached_property
    def head_size(self) -> int:
        if (
            hasattr(self.hf_text_config, "model_type")
            and self.hf_text_config.model_type == "deepseek_v2"
        ):
            # FlashAttention supports only head_size 32, 64, 128, 256,
            # we need to pad head_size 192 to 256
            return 256
        if hasattr(self.hf_text_config, "head_dim"):
            return self.hf_text_config.head_dim
        # FIXME(woosuk): This may not be true for all models.
        return (
            self.hf_text_config.hidden_size // self.hf_text_config.num_attention_heads
        )

    @cached_property
    def total_num_kv_heads(self) -> int:
        # For GPTBigCode & Falcon:
        # NOTE: for falcon, when new_decoder_architecture is True, the
        # multi_query flag is ignored and we use n_head_kv for the number of
        # KV heads.
        falcon_model_types = ["falcon", "RefinedWeb", "RefinedWebModel"]
        new_decoder_arch_falcon = (
            self.hf_config.model_type in falcon_model_types
            and getattr(self.hf_config, "new_decoder_architecture", False)
        )
        if not new_decoder_arch_falcon and getattr(
            self.hf_text_config, "multi_query", False
        ):
            # Multi-query attention, only one KV head.
            # Currently, tensor parallelism is not supported in this case.
            return 1

        # For DBRX and MPT
        if self.hf_config.model_type == "mpt":
            if "kv_n_heads" in self.hf_config.attn_config:
                return self.hf_config.attn_config["kv_n_heads"]
            return self.hf_config.num_attention_heads
        if self.hf_config.model_type == "dbrx":
            return getattr(
                self.hf_config.attn_config,
                "kv_n_heads",
                self.hf_config.num_attention_heads,
            )

        attributes = [
            # For Falcon:
            "n_head_kv",
            "num_kv_heads",
            # For LLaMA-2:
            "num_key_value_heads",
            # For ChatGLM:
            "multi_query_group_num",
        ]
        for attr in attributes:
            num_kv_heads = getattr(self.hf_text_config, attr, None)
            if num_kv_heads is not None:
                return num_kv_heads

        # For non-grouped-query attention models, the number of KV heads is
        # equal to the number of attention heads.
        return self.hf_text_config.num_attention_heads

    def get_num_kv_heads(self, parallel_config: "ParallelConfig") -> int:
        """Returns the number of KV heads per GPU."""
        # If tensor parallelism is used, we divide the number of KV heads by
        # the tensor parallel size. We will replicate the KV heads in the
        # case where the number of KV heads is smaller than the tensor
        # parallel size so each GPU has at least one KV head.
        return max(1, self.total_num_kv_heads // parallel_config.tensor_parallel_size)

    @cached_property
    def max_model_len(self) -> int:
        max_model_len = float("inf")
        possible_keys = [
            # OPT
            "max_position_embeddings",
            # GPT-2
            "n_positions",
            # MPT
            "max_seq_len",
            # Others
            "max_sequence_length",
            "max_seq_length",
            "seq_len",
        ]
        for key in possible_keys:
            max_len_key = getattr(
                self.hf_config, key, getattr(self.hf_text_config, key, None)
            )
            if max_len_key is not None:
                max_model_len = min(max_model_len, max_len_key)
        if max_model_len == float("inf"):
            raise ValueError(f"unknown max_model_len for {self.model}")
        return max_model_len

    def get_num_attention_heads(
        self, parallel_config: ParallelConfig = ParallelConfig()
    ) -> int:
        num_heads = getattr(self.hf_text_config, "num_attention_heads", 0)
        return num_heads // parallel_config.tensor_parallel_size

    def get_num_layers(self, parallel_config: ParallelConfig = ParallelConfig()) -> int:
        if hasattr(self.hf_config, "num_hidden_layers"):
            total_num_hidden_layers = self.hf_config.num_hidden_layers
        else:
            total_num_hidden_layers = self.hf_text_config.num_hidden_layers
        assert total_num_hidden_layers % parallel_config.pipeline_parallel_size == 0, (
            f"Number of layers ({total_num_hidden_layers}) must be divisible "
            f"by the size of pipeline parallel group "
            f"({parallel_config.pipeline_parallel_size})."
        )
        return total_num_hidden_layers // parallel_config.pipeline_parallel_size

    def get_model_size_in_bytes(self, parallel_config: ParallelConfig) -> int:
        assert parallel_config.world_size == 1
        nbytes = int(self.model.n_params() * self.dtype_size)
        aligned = int(nbytes) + VRAM_ALIGN - 1
        return aligned - aligned % VRAM_ALIGN

        # total_params = (
        #     self.vocab_size * self.hidden_size  # vocab embed
        #     + self.max_model_len * self.hidden_size  # position embed
        #     + 4
        #     * self.get_num_layers(parallel_config)
        #     * (self.hidden_size ** 2)  # attention
        #     / parallel_config.tensor_parallel_size # attention is divided by tp
        #     + 8
        #     * self.get_num_layers(parallel_config)
        #     * (self.hidden_size ** 2)  # FFN
        #     / parallel_config.tensor_parallel_size # FFN is divided by tp
        #     + 5 * self.get_num_layers(parallel_config) * self.hidden_size  # bias
        # )
        # return total_params * self.dtype_size

    def get_kv_block_shape(
        self,
        parallel_config: ParallelConfig = ParallelConfig(),
    ) -> Tuple[int, ...]:
        """
        Get the KV cache block shape for the model.

        This is per-block shape; i.e., it does not include num_blocks as a dimension.
        """
        return (
            self.get_num_layers(parallel_config),
            2,
            BLOCK_SIZE,
            self.get_num_kv_heads(parallel_config),
            self.head_size,
        )

    def get_max_num_blocks(
        self,
        parallel_config: ParallelConfig = ParallelConfig(),
        device_type: "DeviceType" = DeviceType.H800,
        memory_utilization: float = 72 / 96,
        prefetch_model_config: Optional["ModelConfig"] = None,
    ) -> int:
        """
        Get the maximum number of KV cache blocks that can be allocated on the GPU device.
        """
        capacity_in_bytes = device_type.mem_capacity_in_bytes()
        model_size_in_bytes = self.get_model_size_in_bytes(
            parallel_config=parallel_config
        )
        pf_model_size_in_bytes = (
            0
            if prefetch_model_config is None
            else prefetch_model_config.get_model_size_in_bytes(
                parallel_config=parallel_config
            )
        )
        total_size_in_bytes = (
            capacity_in_bytes * memory_utilization
            - model_size_in_bytes
            - pf_model_size_in_bytes
        )
        block_size_in_bytes = (
            prod(self.get_kv_block_shape(parallel_config=parallel_config))
            * self.dtype_size
        )

        # print("cap: {}, model: {}, pfmodel: {}, kvtot: {}, bsz: {}".format(
        #     capacity_in_bytes/(1024**3),
        #     model_size_in_bytes/(1024**3),
        #     pf_model_size_in_bytes/(1024**3),
        #     total_size_in_bytes/(1024**3),
        #     block_size_in_bytes/(1024**3),
        # ))
        return int(total_size_in_bytes / block_size_in_bytes)


@dataclass(eq=False)
class QuickLoaderConfig:
    """Config for the QuickLoader."""

    # File name for the model cache shared memory.
    model_cache_filename: Optional[str] = None
    # Size of the model cache in bytes.
    model_cache_size: Optional[int] = None
    # Snapshot of cached models.
    model_cache_snapshot: Optional[Dict[str, "ModelHandle"]] = None
    # Size of pinned buffer in the loader in bytes.
    pinned_buffer_size: int = 4 * (1024**3)


_MODEL_CONFIGS = {}


def get_model_config(model: Optional["ModelType"]) -> Optional[ModelConfig]:
    """Get globally cached model config for a model type."""
    if model is None:
        return None
    if model not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model] = ModelConfig(model)
    return _MODEL_CONFIGS[model]

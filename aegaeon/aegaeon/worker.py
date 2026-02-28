"""
Adapted from distserve/worker.py.
"""

import ray
import ray.actor
import torch
import os
import gc
import ctypes
import threading
import time

from typing import List, Optional, Tuple, Dict, TYPE_CHECKING
from collections import defaultdict

from vllm.attention import get_attn_backend
from vllm.distributed import (
    set_custom_all_reduce,
    init_distributed_environment,
    ensure_model_parallel_initialized,
)
from vllm.utils import init_cached_hf_modules

from aegaeon import ops
from aegaeon.config import ModelConfig, ParallelConfig, QuickLoaderConfig, BLOCK_SIZE
from aegaeon.logger import init_logger
from aegaeon.models import get_model
from aegaeon.request import RequestMeta
from aegaeon.allocator import initialize_alloc, get_alloc
from aegaeon.utils import (
    CudaRTLibrary,
    cudaStream_t,
    cudaEvent_t,
    cudaIpcMemHandle_t,
    DeviceType,
    set_random_seed,
    make_tensor_with_pad,
    reduce_cuda_event,
    rebuild_cuda_event,
    get_logits_processor,
    get_lm_head,
    prod,
    GB,
)

if TYPE_CHECKING:
    from aegaeon.stage_engine import Stage

logger = init_logger(__name__)

# Sleep this many seconds when synchronizing for model prefetch.
SLEEP_WHEN_WAITING_PREFETCH = 0.002

"""
XXX: DO NOT rely on ray to assign GPUs to Workers and sort out placement (i.e. by
setting `num_gpus=1` below). We need to share `torch.cuda.Event`s between Workers
which need `torch.device` information to rebuild, requiring a Worker to have
access to all devices.
Instead, set torch.device manually at Worker initialization. 
"""


@ray.remote(num_cpus=16)
class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    initiating KV cache movement, and executing the model on the GPU.
    In case of distributed inference, each worker is assigned a partition of
    the model.

    Workers support lightweight model swapping. Each worker is initialized
    lazily without actually loading a model or allocating kvcache yet, which is
    delayed to the switching time (`switch`).

    XXX: inter-GPU migration is currently not implemented due to the extra
    complexity in managing CUDA IPC resource lifetimes. For the use case of
    Aegaeon workers, the impact should be minimal.
    """

    def __init__(
        self,
        stage: "Stage",
        node_id: int,
        engine_id: int,
        worker_id: int,
        device_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        # CPU cache related
        cpu_cache_filename: str,
        cpu_cache_size: int,
    ) -> None:
        start = time.time()

        # Common lazy initialzation; executed only once
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # get rid of whatever ray sets
        if "AEGAEON_CUDA_VISIBLE_DEVICES" in os.environ:
            # respect custom settings
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ[
                "AEGAEON_CUDA_VISIBLE_DEVICES"
            ]
        assert torch.cuda.is_available(), "Aegaeon is only available on CUDA devices"
        torch.cuda.set_device(
            device_id
        )  # equivalent to CUDA_VISIBLE_DEVICES set by ray; however other devices are still visible

        init_cached_hf_modules()

        self.stage = stage
        self.node_id = node_id
        self.engine_id = engine_id
        self.worker_id = worker_id
        self.device_id = device_id
        self.model = None

        self.model_config: Optional[ModelConfig] = None
        self.parallel_config: ParallelConfig = parallel_config

        self.device = torch.device("cuda", device_id)
        self.device_type = DeviceType.from_str(torch.cuda.get_device_name(self.device))
        assert (
            self.device_type is not None
        ), f"unsupported device type {torch.cuda.get_device_name(self.device)}"

        # K/V cache on GPU
        self.kv_cache: torch.Tensor = None

        # K/V swap on CPU
        assert os.path.isfile(
            cpu_cache_filename
        ), f"no shared memory object found at {cpu_cache_filename}"
        self.kv_swap: torch.Tensor = torch.from_file(
            cpu_cache_filename,
            shared=True,
            size=cpu_cache_size,
            dtype=torch.float16,
            device="cpu",
        )
        self.cudart = CudaRTLibrary()
        # print(self.kv_swap.data_ptr(), self.kv_swap.nbytes)
        self.cudart.cudaHostRegister(
            ctypes.c_void_p(self.kv_swap.data_ptr()), self.kv_swap.nbytes
        )

        # CUDA streams
        # Events from these streams may be shared to other workers for synchronization
        # Refer to block_manager.py for more details.
        self.swap_in_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        self.swap_out_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)
        self.compute_stream: torch.cuda.Stream = torch.cuda.Stream(self.device)

        # Events
        # Mapping: request_id -> all events (created locally for the request)
        self.move_event_table: Dict[int, List[torch.cuda.Event]] = defaultdict(list)

        # Statistics
        self.execution_time = 0.0

        # Model
        self.model: torch.nn.Module = None

        # Fast switching
        self.quick_loader = None
        if os.getenv("FAST_SWITCH") == "1":
            from aegaeon.loader import QuickLoader

            initialize_alloc(
                int(self.device_type.mem_capacity_in_bytes() * 70 / 80), self.device
            )
            self.quick_loader = QuickLoader(self.cudart, loader_config)
        else:
            raise RuntimeError()

        # Prefetch related
        self.prefetch_task: Optional[Tuple[ModelConfig, bool]] = None
        self.prefetch_model: Optional[torch.nn.Module] = None
        self.prefetch_thread = threading.Thread(target=self._prefetch, daemon=True)
        self.prefetch_thread.start()

        # Set random seed
        set_random_seed(0)

        end = time.time()
        logger.info(
            f"launched on {self.node_id} and device #{self.device_id} in {end-start:.2f}s"
        )

    def __repr__(self):
        return f"{self.stage} engine #{self.engine_id}, worker #{self.worker_id}|"

    def get_device_type(self) -> DeviceType:
        return self.device_type

    def initialize(
        self,
        distributed_init_method: str,
        rank: int,
        local_rank: int,
    ) -> None:
        """Finish worker initialization by setting up the distributed environment."""
        # torch.distributed.all_reduce does not free the input tensor until
        # the synchronization point. This causes the memory usage to grow
        # as the number of all_reduce calls increases. This env var disables
        # this behavior.
        # Related issue:
        # https://discuss.pytorch.org/t/cuda-allocation-lifetime-for-inputs-to-distributed-all-reduce/191573
        os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

        self.distributed_init_method = distributed_init_method
        self.rank = rank
        self.local_rank = local_rank
        _init_worker_distributed_environment(
            self.parallel_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
        )

        logger.info(f"initialized on {self.node_id} and device #{self.device_id}")

    @torch.inference_mode()
    def switch(
        self,
        num_gpu_blocks: int,
        model_config: ModelConfig,
        enable_quick_loader: bool = True,
        prefetch_model_config: Optional[ModelConfig] = None,
        prefetch_enable_quick_loader: bool = True,
    ) -> None:
        """Switch the worker to use the given model config.

        Reset the worker state, and (re)loads model weights and initializes kvcache.
        Allocator memory layout:
        +----------+-------+----------------+
        | KV Cache | Model | Prefetch Model |
        +----------+-------+----------------+
        """
        
        self._wait_model_prefetch()
        self._reset()
        self.model_config = model_config
        self._init_cache(num_gpu_blocks)
        self._init_model(model_config, enable_quick_loader)

        # Submit prefetch
        if prefetch_model_config is not None:
            self.prefetch_task = (prefetch_model_config, prefetch_enable_quick_loader)

    def _prefetch(self):
        """Prefetch thread."""
        while True:
            if self.prefetch_task is None or self.prefetch_model is not None:
                # No prefetch, or already prefetched
                time.sleep(SLEEP_WHEN_WAITING_PREFETCH)
                continue

            assert self.prefetch_model is None
            model_config, enable_quick_loader = self.prefetch_task
            if enable_quick_loader:
                if self.quick_loader is None:
                    raise ValueError(f"QuickLoader not initialized")
                quick_loader = self.quick_loader
            else:
                quick_loader = None

            logger.info(f"prefetching {model_config.model} ({self})")
            self.prefetch_model = get_model(
                model_config=model_config,
                parallel_config=self.parallel_config,
                device=self.device,
                quick_loader=quick_loader,
            )

    def _reset(self):
        """Reset the worker state, clearing model weights and kvcache.

        This method will wait for all ongoing block movements to complete before
        release GPU memory. It does not, however, ensure that all requests logically
        on the worker have received move-out operations, which is the control plane's
        responsibility.
        """
        if self.model is None:
            return

        self.wait_for_all_move_out()

        # Clear states
        self.model_config = None
        self.kv_cache = None
        self.kv_cache_run = None
        self.kv_swap_view = None
        self.model = None

        # Ensure GPU memory is released
        if os.getenv("FAST_SWITCH") == "1":
            get_alloc().clear()
        else:
            gc.collect()
            torch.cuda.empty_cache()

    def _wait_model_prefetch(self):
        if self.prefetch_task is not None:
            while self.prefetch_model is None:
                time.sleep(SLEEP_WHEN_WAITING_PREFETCH)

    def _init_model(
        self,
        model_config: ModelConfig,
        enable_quick_loader: bool,
    ) -> None:
        """Initialize the model.

        Loads the model weights from CPU, or wait for the prefetch to complete.
        """
        start = time.time()
        if self.prefetch_task is None:
            # No prefetch submitted
            if enable_quick_loader:
                quick_loader = self.quick_loader
            else:
                quick_loader = None
            self.model = get_model(
                model_config=model_config,
                parallel_config=self.parallel_config,
                device=self.device,
                quick_loader=quick_loader,
            )
            how = f"directly ({self})"
        else:
            if self.prefetch_task[0].model != model_config.model:
                logger.error(
                    f"Mismatch between prefetching model {self.prefetch_task[0].model} "
                    f"and scheduled model {model_config.model}"
                )
                raise RuntimeError()

            # Move parameters
            alloc = get_alloc()
            params = [param for param in self.prefetch_model.parameters()]
            params.sort(key=lambda param: param.data_ptr())
            for param in params:
                new_storage = alloc.allocate(param.shape, param.dtype, raw=True)
                new_storage.copy_(param.untyped_storage(), non_blocking=True)
            torch.cuda.default_stream(self.device).synchronize()

            self.model = self.prefetch_model
            # XXX: this order guarantees that _prefetch won't interleave
            self.prefetch_task = None
            self.prefetch_model = None
            how = f"with prefetch ({self})"

        end = time.time()
        logger.info(
            f"model {self.model_config.model} loaded {how}; elapsed: {end-start:.2f}s"
        )

    def _init_cache(
        self,
        num_gpu_blocks: int,
    ) -> None:
        """Allocate the GPU KV cache (the CPU cache is allocated beforehand and shared
        by all engines and workers on a node).

        The KV cache shape is (num_gpu_blocks, *kv_block_shape), where kv_block_shape
        is decided by the model and parallel setting.
        """
        # GPU cache storage shape
        dtype = self.model_config.torch_dtype
        num_gpu_blocks //= self.parallel_config.pipeline_parallel_size
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.kv_block_shape = self.model_config.get_kv_block_shape(self.parallel_config)
        # [num_block, num_layer, 2, ...]
        self.kv_cache_shape = (num_gpu_blocks, *self.kv_block_shape)
        self.block_size = BLOCK_SIZE

        # CPU cache storage shape
        kv_block_size = prod(self.kv_block_shape)
        num_cpu_blocks = (
            self.kv_swap.numel() // kv_block_size
        )  # assuming both has the same dtype
        self.kv_swap_shape = (num_cpu_blocks, *self.kv_block_shape)
        self.kv_swap_size = num_cpu_blocks * kv_block_size
        self.kv_swap_view = self.kv_swap.narrow(0, 0, self.kv_swap_size).view(
            size=self.kv_swap_shape
        )

        # Running the model with vLLM requires a slightly different kvcache shape
        self.attn_backend = get_attn_backend(
            self.model_config.get_num_attention_heads(self.parallel_config),
            self.model_config.head_size,
            num_kv_heads,
            None,
            dtype,
            None,
            BLOCK_SIZE,
        )
        # [num_layer, 2, num_blocks, ...] (assuming flash-attn backend)
        kv_cache_shape_run = (
            num_layers,
            *self.attn_backend.get_kv_cache_shape(
                num_gpu_blocks, BLOCK_SIZE, num_kv_heads, self.model_config.head_size
            ),
        )

        def assert_kv_perm(
            kv_cache_shape: Tuple[int, ...], kv_cache_shape_run: Tuple[int, ...]
        ) -> Tuple[int, ...]:
            perm = (1, 2, 0, 3, 4, 5)
            assert len(perm) == len(kv_cache_shape) and len(perm) == len(
                kv_cache_shape_run
            )
            assert all(
                kv_cache_shape_run[i] == kv_cache_shape[perm[i]]
                for i in range(len(perm))
            )
            return perm

        self.kv_perm = assert_kv_perm(self.kv_cache_shape, kv_cache_shape_run)

        # Initialize cache with the storage shape
        if os.getenv("FAST_SWITCH") == "1":
            alloc = get_alloc()
            self.kv_cache = alloc.allocate(self.kv_cache_shape, dtype)
        else:
            self.kv_cache = torch.empty(
                self.kv_cache_shape, dtype=dtype, device=self.device
            )
        # Permute the kv-cache tensor for vLLM: (num_blocks, num_layers, 2, ...)
        self.kv_cache_run = self.kv_cache.permute(*self.kv_perm)
        logger.info(
            f"cache ({num_gpu_blocks} blocks, shape={self.kv_cache_shape}) initialized"
        )

    @torch.inference_mode()
    def step(
        self,
        requests: List[RequestMeta],
        block_tables: List[List[int]],
        time_blocked: bool = False,
    ) -> Tuple[List[Optional[int]], Optional[List[float]]]:
        """Run one step of inference on the batch of requests.

        This method will wait for involved requests to finish moving in.
        It does not, however, ensure that all requests logically on the worker have
        received move-in operations, which is the stage engine's responsibility.
        """
        block_times = [] if time_blocked else None

        if len(requests) == 0:
            return ([], block_times)

        # Check whether synchronization is necessary;
        # block only when no request in the batch is ready.
        ready_mask: List[bool] = []
        blocked = False
        for i, request in enumerate(requests):
            request_id = request[0]
            if request_id in self.move_event_table:
                # Query if the event is finished already
                event = self.move_event_table[request_id][-1]
                logger.debug(f"querying for {event}")
                if not event.query():
                    ready_mask.append(False)
                    continue
            # No need to pend for this request
            ready_mask.append(True)

        if not any(ready_mask):
            # Sadly no request is ready; must wait for at least one event.
            # For simplicity, we wait for all events.
            blocked = True
            logger.info(f"blocking for {[request[0] for request in requests]}")
            for request in requests:
                request_id = request[0]
                # Block to wait for the event (requirement #1)
                events = self.move_event_table.get(request_id, None)
                if events is not None:
                    events[-1].wait(self.compute_stream)

            # All ready now
            ready_mask = [True for _ in range(len(requests))]

        # Run forward
        with torch.cuda.stream(self.compute_stream):
            # Prepare model inputs
            input_tokens, input_positions, attn_metadata = self._prepare_model_inputs(
                [requests[i] for i, ready in enumerate(ready_mask) if ready],
                [block_tables[i] for i, ready in enumerate(ready_mask) if ready],
            )

            if time_blocked and blocked:
                start = time.time()
                self.compute_stream.synchronize()
                end = time.time()
            hidden_states = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=self.kv_cache_run,
                attn_metadata=attn_metadata,
                intermediate_tensors=None,
            )
            self.compute_stream.synchronize()

            # Greedy sampling (without SamplingMetadata)
            hidden_states = hidden_states.index_select(
                0, attn_metadata.query_start_loc[1:] - 1
            )
            logits_processor = get_logits_processor(self.model)
            logits = logits_processor._get_logits(
                hidden_states, get_lm_head(self.model), embedding_bias=None
            )
            if logits is not None:
                soft_cap = logits_processor.soft_cap
                scale = logits_processor.scale
                if soft_cap is not None:
                    logits = logits / soft_cap
                    logits = torch.tanh(logits)
                    logits = logits * soft_cap
                if scale != 1.0:
                    logits *= scale

            generated_token_ids_tensor = torch.argmax(logits, dim=-1).cpu()

        # Return tokens
        if time_blocked:
            for ready in ready_mask:
                block_time = end - start if blocked and ready else 0
                block_times.append(block_time)

        return (list(generated_token_ids_tensor), block_times)

    def _prepare_model_inputs(
        self,
        requests: List[RequestMeta],
        _block_tables: List[List[int]],
    ):
        """Helper method to prepare the model inputs based on a given batch.
        Prepares metadata needed for the base model forward pass.
        XXX: we assume only the flash-attn backend is used.
        XXX: sliding window is not supported.

        The API assumes requests is sorted by prefill -> decode.

        The result tensors and data structure also batches input in prefill
        -> decode order. For example,

        - input_tokens[:num_prefill_tokens] contains prefill tokens.
        - input_tokens[num_prefill_tokens:] contains decode tokens.

        NOTE: Definition of context_len, query_len, and seq_len:
        |---------- N-1 iteration --------|
        |---------------- N iteration ---------------------|
        |- tokenA -|......................|-- newTokens ---|
        |---------- context_len ----------|
        |-------------------- seq_len ---------------------|
                                          |-- query_len ---|
        """
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        seq_lens: List[int] = []
        prefill_seq_lens: List[int] = []
        decode_seq_lens: List[int] = []
        context_lens: List[int] = []
        query_lens: List[int] = []
        block_tables: List[List[int]] = []
        num_prefills = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        assert len(requests) > 0

        for request, _block_table in zip(requests, _block_tables):
            _, tokens, seq_len, context_len = request
            is_prompt = context_len == 0

            if not is_prompt:
                # Decode
                block_table = _block_table.copy()
            else:
                # Prefill (without chunked prefill)
                block_table = []
            block_tables.append(block_table)

            seq_lens.append(seq_len)
            context_lens.append(context_len)
            query_len = seq_len - context_len
            query_lens.append(query_len)
            input_tokens.extend(tokens)
            input_positions.extend(list(range(context_len, seq_len)))

            if is_prompt:
                num_prefills += 1
                num_prefill_tokens += len(tokens)
                prefill_seq_lens.append(seq_len)
            else:
                assert (
                    query_len == 1
                ), "seq_len: {}, context_len: {}, query_len: {}".format(
                    seq_len, context_len, query_len
                )
                num_decode_tokens += query_len
                decode_seq_lens.append(seq_len)

            # Compute the slot mapping.
            for i in range(context_len, seq_len):
                block_number = _block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(prefill_seq_lens, default=0)
        max_decode_seq_len = max(decode_seq_lens, default=0)
        max_block_table_len = max(len(block_table) for block_table in block_tables)

        block_tables_tensor = make_tensor_with_pad(
            block_tables,
            max_len=max_block_table_len,
            pad=0,
            dtype=torch.int,
            device=self.device,
        )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        context_lens_tensor = torch.tensor(
            context_lens, dtype=torch.int, device=self.device
        )

        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=self.device)
        query_lens_tensor = torch.tensor(
            query_lens, dtype=torch.long, device=self.device
        )
        query_start_loc = torch.zeros(
            query_lens_tensor.shape[0] + 1, dtype=torch.int32, device=self.device
        )
        seq_start_loc = torch.zeros(
            seq_lens_tensor.shape[0] + 1, dtype=torch.int32, device=self.device
        )

        torch.cumsum(
            seq_lens_tensor, dim=0, dtype=seq_start_loc.dtype, out=seq_start_loc[1:]
        )
        torch.cumsum(
            query_lens_tensor,
            dim=0,
            dtype=query_start_loc.dtype,
            out=query_start_loc[1:],
        )

        input_tokens_tensor = torch.tensor(
            input_tokens, dtype=torch.long, device=self.device
        )
        input_positions_tensor = torch.tensor(
            input_positions, dtype=torch.long, device=self.device
        )
        slot_mapping_tensor = torch.tensor(
            slot_mapping, dtype=torch.long, device=self.device
        )

        logits_soft_cap = getattr(
            self.model_config.hf_config, "attn_logit_softcapping", None
        )
        if logits_soft_cap is not None:
            raise ValueError(
                "Please use Flashinfer backend for models with"
                "logits_soft_cap (i.e., Gemma-2)."
                " Otherwise, the output might be wrong."
            )

        attn_metadata = self.attn_backend.make_metadata(
            # Base AttentionMetadata
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            # FlashAttentionMetadata
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables_tensor,
            use_cuda_graph=False,
        )

        return (input_tokens_tensor, input_positions_tensor, attn_metadata)

    @torch.inference_mode()
    def swap_blocks(
        self,
        request_ids: List[int],
        source_block_ids: List[int],
        target_block_ids: List[int],
        source_event_handles: List[
            Optional[List[ray.ObjectRef]]
        ],  # elem: None or events for all workers
        is_swap_in: bool,
    ):
        """
        Swap some blocks between CPU and GPU.

        If is_swap_in, then move blocks from CPU to GPU, i.e. CPU block
        #source_block_ids[0] will be copied to GPU block #target_block_ids[0]
        and so on. Similar for is_swap_in = False.

        This method may block the corresponding stream if the source blocks
        aren't ready. The availability of the target blocks is ensured at
        the block manager level.

        Return a CUDA event for testing completion of this operation.
        """
        stream = self.swap_in_stream if is_swap_in else self.swap_out_stream
        new_event = torch.cuda.Event(interprocess=True, enable_timing=False)

        # Wait for source blocks if needed
        self._wait_source_events(
            cudaStream_t(stream.cuda_stream),
            new_event,
            request_ids,
            source_event_handles,
        )

        # Swap
        # NOTE: we use a hand-crafted swapping kernel (instead of vLLM's) as both the
        # CPU and the GPU cache is shaped differently from vLLM.
        with torch.cuda.stream(stream):
            assert self.kv_cache.device.type == "cuda"
            assert self.kv_swap_view.device.type == "cpu"
            ops.swap(
                source_block_ids,
                target_block_ids,
                is_swap_in,
                self.kv_cache,
                self.kv_swap_view,
            )

        # Record event
        new_event.record(stream)

        logger.info(
            f"swap requests {request_ids} ({'CPU' if is_swap_in else 'GPU'} to {'GPU' if is_swap_in else 'CPU'})"
        )
        return reduce_cuda_event(self.cudart, new_event)

    def _wait_source_events(
        self,
        stream: cudaStream_t,
        new_event: torch.cuda.Event,
        request_ids: List[int],
        source_event_handles: List[Optional[List[ray.ObjectRef]]],
    ):
        # XXX: this is potentially a choke point
        futures = [
            handles[self.worker_id]
            for handles in source_event_handles
            if handles is not None
        ]
        try:
            event_handles = ray.get(futures)
        except Exception as e:
            logger.error(f"error getting source events; error: {e}")
            ray.actor.exit_actor()
        i = 0
        for request_id, fut in zip(request_ids, source_event_handles):
            if fut is not None:
                # The request had a move operation going and it may have not finished yet.
                # Wait for it before we do anything with it ourselves (the event may be from
                # a stream on another device).
                event = rebuild_cuda_event(
                    self.cudart, event_handles[i], is_worker=True
                )
                if event is None:
                    local_event = True
                    if request_id not in self.move_event_table:
                        logger.error(
                            f"request {request_id} should have a local event here at {self}"
                        )
                    event = cudaEvent_t(
                        self.move_event_table[request_id][-1].cuda_event
                    )
                else:
                    local_event = False
                self.cudart.cudaStreamWaitEvent(stream, event, 0)
                logger.debug(f"waited for {event}")
                if not local_event:
                    self.cudart.cudaEventDestroy(event)
                i += 1
            self.move_event_table[request_id].append(new_event)

    def clear_request_resource(self, request_id: int):
        """Clear the resources associated with the request."""
        """This is called when a request is finished or aborted"""
        self.move_event_table.pop(request_id, None)

    def clear_request_resource_batched(self, request_ids: List[int]):
        """Clear the resources associated with the requests."""
        for request_id in request_ids:
            self.clear_request_resource(request_id)

    def wait_for_all_move_in(self):
        """Wait for all ongoing move-in operations to finish."""
        # start = time.time()
        self.swap_in_stream.synchronize()
        # end = time.time()
        # logger.info(f"synchronized move-in ({end-start:.2f}s)")

    def wait_for_all_move_out(self):
        """Wait for all ongoing move-out operations to finish."""
        # start = time.time()
        self.swap_out_stream.synchronize()
        # end = time.time()
        # logger.info(f"synchronized move-out ({end-start:.2f}s)")

    def migrate_blocks(
        self,
        request_ids: List[int],
        source_engine_id: int,
        source_block_ids: List[int],
        target_block_ids: List[int],
        source_events: List[
            Optional[List[ray.ObjectRef]]
        ],  # elem: None or events for all workers
    ):
        raise NotImplementedError()

    def migrate_blocks_source(
        self,
        request_ids: List[int],
        migration_event: ray.ObjectRef,
    ):
        raise NotImplementedError()

    def register_kvcache_mem_handles(
        self,
        prefill_engine_id: int,
        prefill_parallel_config: ParallelConfig,
        kvcache_ipc_mem_handles: List[List[cudaIpcMemHandle_t]],
    ):
        raise NotImplementedError()

    def unregister_kvcache_mem_handles(
        self,
        prefill_engine_id: int,
    ):
        raise NotImplementedError()


def _init_worker_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
    local_rank: int = -1,
) -> None:
    """Initialize the distributed environment."""
    set_custom_all_reduce(False)

    init_distributed_environment(
        parallel_config.world_size, rank, distributed_init_method, local_rank
    )

    ensure_model_parallel_initialized(
        parallel_config.tensor_parallel_size, parallel_config.pipeline_parallel_size
    )

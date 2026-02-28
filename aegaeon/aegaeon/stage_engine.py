import time
import copy
from typing import Callable, Optional, List
from abc import ABC, abstractmethod
import asyncio

from enum import Enum
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import PlacementGroup
from vllm.transformers_utils.tokenizer import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from aegaeon.logger import init_logger
from aegaeon.config import (
    ModelConfig,
    ParallelConfig,
    QuickLoaderConfig,
    BLOCK_SIZE,
)
from aegaeon.request import (
    Request,
    BatchedRequests,
)
from aegaeon.utils import DeviceType, get_distributed_init_method, get_open_port
from aegaeon.lifetime import LifetimeEvent, LifetimeEventType
from aegaeon.worker import Worker
from aegaeon.prefill_scheduler import PrefillScheduler, get_prefill_stage_scheduler
from aegaeon.decode_scheduler import DecodeScheduler

logger = init_logger(__name__)


class Stage(Enum):
    """The two stages of LLM serving."""

    PREFILL = "prefill"
    DECODE = "decode"

    def __str__(self) -> str:
        return self.value


# Sleep for this many seconds when there is no request in PrefillStageLLMEngine.step()
# We need to sleep for a while because the whole program is a asyncio-based,
# event driven, single thread program. We save some CPU time for other coroutines.
SLEEP_WHEN_PREFILL_NO_REQUEST = 0.003

# Sleep for this many seconds when there is no request in DecodeStageLLMEngine.step()
SLEEP_WHEN_DECODE_NO_REQUEST = 0.003

# Sleep for this many seconds in each event loop, useful for debugging
SLEEP_IN_EACH_EVENT_LOOP = 0

# Print engine status every this many seconds
PRINT_STATUS_INTERVAL = 1


class StepOutput:
    """The output of request in one step of inference.
    It contains the information of corresponding request and the generated tokens until this step.
    """

    def __init__(
        self,
        request: Request,
        new_token: str,
        new_token_id: int,
        dispatch_time: float,
        issue_time: float,
        step_time: float,
        block_overhead: Optional[float] = None,
    ):
        self.request = request
        self.request_id = request.request_id
        self.new_token = new_token
        self.new_token_id = new_token_id
        self.is_finished = request.is_finished

        self.dispatch_time = dispatch_time
        self.issue_time = issue_time
        self.step_time = step_time
        self.block_overhead = block_overhead

    def __repr__(self) -> str:
        return (
            f"StepOutput(request_id={self.request_id}, "
            f"new_token={self.new_token}, "
            f"new_token_id={self.new_token_id}, "
            f"is_finished={self.is_finished})"
        )


class StageEngine(ABC):
    """
    StageEngine: An engine that runs either the prefill stage or the decode stage.

    This class is the base class for PrefillEngine and DecodeEngine.
    """

    @abstractmethod
    def _get_scheduler(self) -> PrefillScheduler | DecodeScheduler:
        raise NotImplementedError()

    def __init__(
        self,
        stage: Stage,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
    ):
        self.engine_id = engine_id
        assert engine_id >= 0, f"valid engine_id should be >= 0"
        self.stage = stage
        self.model_config: Optional[ModelConfig] = None  # initialized later
        self.tokenizer: Optional[PreTrainedTokenizer | PreTrainedTokenizerFast] = None
        self.parallel_config = parallel_config
        self.loader_config = loader_config
        self.device_ids = device_ids
        self.placement_group = placement_group

        self.scheduler: PrefillScheduler | DecodeScheduler = self._get_scheduler()

        self.on_new_step_output_callback = on_new_step_output_callback
        self.on_new_lifetime_event_callback = on_new_lifetime_event_callback
        self.on_requests_finished_callback = on_requests_finished_callback

        # workers[i][j] is the j-th tensor-parallel worker in pipeline stage i
        self.workers = []

    def __repr__(self):
        return f"{self.node_id}|{self.stage} engine #{self.engine_id}"

    def _remote_call_all_workers_async(self, func_name: str, *args, **kwargs):
        """
        Call func_name asynchronously on all workers; return the futures immediately.
        """
        handlers = []
        for stage in self.workers:
            for worker in stage:
                handlers.append(getattr(worker, func_name).remote(*args, **kwargs))
        return handlers

    async def initialize(self):
        """Initialize workers.

        We seperate this function from __init__ because we want to run it in an async way
        to enable parallel initialization between engines.
        """
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.block_manager = Controller.block_manager()

        logger.info(f"({self}) initializing workers")

        # For each pipeline stage, create tensor_parallel_size workers
        # ach worker will be assigned a GPU;
        # Workers are guaranteed to reside in a single node,
        # so the loopback address is sufficient.
        distributed_init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port()
        )

        for i in range(self.parallel_config.pipeline_parallel_size):
            workers = []

            for j in range(self.parallel_config.tensor_parallel_size):
                tmp_parallel_config = copy.deepcopy(self.parallel_config)
                tmp_parallel_config.pipeline_parallel_rank = i
                tmp_parallel_config.tensor_parallel_rank = j

                worker_id = i * self.parallel_config.tensor_parallel_size + j
                worker = Worker.options(
                    resources={self.node_id: 0.01},
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=self.placement_group,
                        placement_group_bundle_index=self.engine_id,
                    ),
                ).remote(
                    stage=self.stage,
                    node_id=self.node_id,
                    engine_id=self.engine_id,
                    worker_id=worker_id,
                    device_id=self.device_ids[worker_id],
                    parallel_config=self.parallel_config,
                    loader_config=self.loader_config,
                    cpu_cache_filename=Controller.cpu_cache_filename(),
                    cpu_cache_size=Controller.cpu_cache_size(),
                )
                workers.append(worker)
            self.workers.append(workers)

        # gather device type
        device_types = await asyncio.gather(
            *self._remote_call_all_workers_async("get_device_type")
        )
        unique_device_type = set(device_types)
        if len(unique_device_type) > 1:
            raise ValueError(
                f"Heterogeneous device types for engine #{self.engine_id}: {unique_device_type}"
            )
        self.device_type: DeviceType = device_types[0]

        # cache estimators
        from aegaeon.estimator import cache_estimators, DecodeEstimator
        from aegaeon.models import ModelType

        cache_estimators(
            DecodeEstimator,
            [ModelType.from_int(i) for i in range(10000, 10080)],
            self.device_type,
        )

        # initialize distributed groups
        handlers = []
        workers_1d = sum(self.workers, [])
        for rank, worker in enumerate(workers_1d):
            handlers.append(
                worker.initialize.remote(
                    distributed_init_method=distributed_init_method,
                    rank=rank,
                    local_rank=rank,
                )
            )

        # This finishes worker initialization
        await asyncio.wait(handlers)

    def switch(
        self,
        model_config: ModelConfig,
        enable_quick_loader: bool = True,
        prefetch_model_config: Optional[ModelConfig] = None,
        prefetch_enable_quick_loader: bool = True,
    ):
        """
        Switch the engine and its workers to the given model config.

        Load the model, and also initialize GPU kv-cache.
        """
        if self.model_config and self.model_config.model == model_config.model:
            return

        logger.info("({}) switching to {}".format(self, model_config.model))

        # Determine number of GPU blocks
        num_gpu_blocks = model_config.get_max_num_blocks(
            self.parallel_config,
            self.device_type,
            prefetch_model_config=prefetch_model_config,
        )

        # Disable prefetch_model_conbencfig if it makes num_gpu_blocks too small
        if num_gpu_blocks < max(1024, model_config.max_model_len / BLOCK_SIZE):
            prefetch_model_config = None
            num_gpu_blocks = model_config.get_max_num_blocks(
                self.parallel_config, self.device_type
            )
        # num_gpu_blocks = 128

        success = self._remote_call_all_workers_async(
            "switch",
            num_gpu_blocks,
            model_config,
            enable_quick_loader=enable_quick_loader,
            prefetch_model_config=prefetch_model_config,
            prefetch_enable_quick_loader=prefetch_enable_quick_loader,
        )
        # ray.get(success)

        self.model_config = model_config

        from aegaeon.llm import Controller

        self.tokenizer = Controller.get_tokenizer(model_config.model.path())

        # Update block manager
        block_shape = self.model_config.get_kv_block_shape(self.parallel_config)
        self.block_manager.register_gpu_space(
            self.engine_id,
            num_gpu_blocks,
            block_shape,
            self._remote_call_all_workers_async,
        )

    @abstractmethod
    async def start_event_loop(self):
        raise NotImplementedError()

    @abstractmethod
    async def print_status(self):
        raise NotImplementedError()


class PrefillEngine(StageEngine):
    """
    ## Event loops
    - Step
        - Call scheduler for the next batch (spin if none)
        - Invoke `step()` on workers
        - Await results
        - For unfinished requests:
            - Invoke swap-out on workers
            - Send the requests to DecodeDispatcher
        - For finished requests:
            - Free requests

    - Report
        - `print_status()`
    """

    def _get_scheduler(self) -> PrefillScheduler:
        return get_prefill_stage_scheduler(self)

    def __init__(
        self,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
        # The prefill scheduler policy
        policy: str = "uni",
    ):
        self.policy = policy
        super().__init__(
            Stage.PREFILL,
            engine_id,
            parallel_config,
            loader_config,
            device_ids,
            placement_group,
            on_new_step_output_callback,
            on_new_lifetime_event_callback,
            on_requests_finished_callback,
        )

        # All the batched requests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []

    async def _step(self):
        """
        Run prefill on the batch of requests chosen by the scheduler.

        NOTE: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # Get the next batch from scheduler
        dispatch_time = time.time()
        batched_requests = self.scheduler.get_next_batch_and_pop()

        if len(batched_requests) == 0:
            # ..no new batch to serve; spin
            self.batches_in_pipeline.append(batched_requests)
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_PREFILL_NO_REQUEST)
        else:
            logger.debug(
                f"({self}) serving {[request.request_id for request in batched_requests.requests]} "
                f"({sum(request.get_input_len() for request in batched_requests.requests)} tokens)"
            )

            # ..allocate blocks as needed
            await self.block_manager.allocate_blocks_batched(
                batched_requests, self.engine_id
            )

            # ..log down the lifetime event
            for request in batched_requests.requests:
                self.on_new_lifetime_event_callback(
                    request.request_id, LifetimeEvent(LifetimeEventType.PrefillBegin)
                )

            # ..push the batch into pipeline
            batched_requests.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batched_requests)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                [req.meta() for req in batched_requests.requests],
                [
                    self.block_manager.get_block_table(request.request_id).blocks
                    for request in batched_requests.requests
                ],
                # time_blocked=True, # TODO: configure this
            )
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                # ..no request in the batch
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                # ..wait for the results
                issue_time = time.time()
                maybe_generated_tokens_ids, block_times = await self.batches_ret_futures[0]
                step_time = time.time()

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(maybe_generated_tokens_ids)

                # ..invoke scheduler callback (move out requests and clear states)
                await self.scheduler.on_finish_requests(finished_batch)

                for i, (request, maybe_new_token_id) in enumerate(zip(
                    finished_batch.requests, maybe_generated_tokens_ids
                )):
                    if maybe_new_token_id is not None:
                        try:
                            token = self.tokenizer.decode(maybe_new_token_id)
                            # token = "<todo>"
                        except Exception as e:
                            logger.warning(
                                f"({self}) failed to decode token_id {maybe_new_token_id}; error: {e}"
                            )
                            token = "<nat>"
                        block_overhead = None if block_times is None else block_times[i]
                        step_output = StepOutput(
                            request,
                            token,
                            maybe_new_token_id,
                            dispatch_time,
                            issue_time,
                            step_time,
                            block_overhead=block_overhead,
                        )
                        self.on_new_lifetime_event_callback(
                            request.request_id,
                            LifetimeEvent(LifetimeEventType.PrefillEnd),
                        )
                        self.on_new_step_output_callback(
                            request.request_id,
                            step_output,
                        )

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

                # ..post processing
                decode_batch = BatchedRequests()
                finished_reqs: List[Request] = []
                for request in finished_batch.requests:
                    if not request.is_finished:
                        decode_batch.add_request(request)
                    else:
                        finished_reqs.append(request)
                # ..free resources for finished requests
                self.on_requests_finished_callback(finished_reqs)

                # ..put decode requests into DecodeDispatcher
                if len(decode_batch) > 0:
                    from aegaeon.llm import Controller

                    dispatcher = Controller.decode_dispatcher()
                    if dispatcher is None:
                        raise NotImplementedError("cross-node dispatching")
                    logger.debug(
                        f"({self}) sending batch {[request.request_id for request in decode_batch.requests]} -> ({dispatcher})"
                    )
                    dispatcher.add(decode_batch)

    async def start_event_loop(self):
        async def prefill_step_loop():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        async def report_loop():
            while True:
                self.print_status()
                await asyncio.sleep(PRINT_STATUS_INTERVAL)

        await asyncio.gather(prefill_step_loop(), report_loop())

    def print_status(self):
        self.scheduler.print_status()


class DecodeEngine(StageEngine):
    """
    ## Event loops
    - Step
        - (top of round) Call Scheduler to group Requests into batches and assign time slices
        - (top of turn) Get the next batch and time slice
            - optionally, invoke model prefetch
            - optionally, re-dispatch waiting batches with DecodeDispatcher to other engines
        - (during turn)
            - Invoke `step()` on workers
            - Await results until time slice expires
    """

    def _get_scheduler(self) -> DecodeScheduler:
        return DecodeScheduler(self)

    def __init__(
        self,
        engine_id: int,
        parallel_config: ParallelConfig,
        loader_config: QuickLoaderConfig,
        device_ids: List[int],
        placement_group: PlacementGroup,
        # The control plane callback when a new StepOutput of a particular request is generated
        on_new_step_output_callback: Callable[[int, StepOutput], None],
        # The control plane callback when a new LifetimeEvent of a particular request is generated
        on_new_lifetime_event_callback: Callable[[int, LifetimeEvent, bool], None],
        # The control plane callback when a batch of requests are finished
        on_requests_finished_callback: Callable[[List[Request]], None],
    ):
        super().__init__(
            Stage.DECODE,
            engine_id,
            parallel_config,
            loader_config,
            device_ids,
            placement_group,
            on_new_step_output_callback,
            on_new_lifetime_event_callback,
            on_requests_finished_callback,
        )

        # All the batchedrequests that are pushed into the pipeline
        # Note: len(batched_in_pipeline) <= pp_size and batches are appended in FIFO
        self.batches_in_pipeline: List[BatchedRequests] = []
        self.batches_ret_futures = []

    async def _step(self) -> None:
        """
        Run one step of decode on the batch of requests chosen by the scheduler.

        NOTE: if pipeline parallelism is used, one step only kicks one stage of execution,
        and each request needs #pp steps in total to generate one token.
        """
        # Get the next batch from scheduler
        #  - this may trigger move-in if some requests are not on GPU
        #  - this may also trigger move-out if the batch grows too much, or we're starting a new turn
        dispatch_time = time.time()
        batch = await self.scheduler.get_next_batch()

        if len(batch) == 0:
            self.batches_in_pipeline.append(BatchedRequests())
            self.batches_ret_futures.append(None)
            await asyncio.sleep(SLEEP_WHEN_DECODE_NO_REQUEST)
        else:
            # ..log down the lifetime event
            for request in batch.requests:
                self.on_new_lifetime_event_callback(
                    request.request_id,
                    LifetimeEvent(LifetimeEventType.DecodeBegin),
                    True,
                )

            # ..allocate blocks as needed
            await self.block_manager.allocate_blocks_batched(batch, self.engine_id)

            # ..push the batch into pipeline
            batch.start_one_iteration(time.time())
            self.batches_in_pipeline.append(batch)
            remote_calls = self._remote_call_all_workers_async(
                "step",
                [req.meta() for req in batch.requests],
                [
                    self.block_manager.get_block_table(request.request_id).blocks
                    for request in batch.requests
                ],
                # time_blocked=True, TODO: configure this
            )
            # only the leader of the last stage return valid output, i.e., generated tokens ids
            pp_size = self.parallel_config.pipeline_parallel_size
            tp_size = self.parallel_config.tensor_parallel_size
            self.batches_ret_futures.append(remote_calls[(pp_size - 1) * tp_size])

        if len(self.batches_in_pipeline) == self.parallel_config.pipeline_parallel_size:
            # if the pipeline is full, block until the earliest batch returns
            # if pipeline parallelism is not used, i.e., pp = 1, this should always be true
            if self.batches_ret_futures[0] is None:
                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)
            else:
                # ..wait for the results
                issue_time = time.time()
                maybe_generated_tokens_ids, block_times = await self.batches_ret_futures[0]
                step_time = time.time()

                finished_batch = self.batches_in_pipeline[0]
                finished_batch.finish_one_iteration(maybe_generated_tokens_ids)

                for i, (request, maybe_new_token_id) in enumerate(zip(
                    finished_batch.requests, maybe_generated_tokens_ids
                )):
                    if maybe_new_token_id is not None:
                        try:
                            token = self.tokenizer.decode(maybe_new_token_id)
                            # token = "<todo>"
                        except Exception as e:
                            logger.warning(
                                f"({self}) failed to decode token_id {maybe_new_token_id}; error: {e}"
                            )
                            token = "<nat>"
                        block_overhead = None if block_times is None else block_times[i]
                        step_output = StepOutput(
                            request,
                            token,
                            maybe_new_token_id,
                            dispatch_time,
                            issue_time,
                            step_time,
                            block_overhead=block_overhead,
                        )
                        self.on_new_step_output_callback(
                            request.request_id,
                            step_output,
                        )
                        if request.is_finished:
                            self.on_new_lifetime_event_callback(
                                request.request_id,
                                LifetimeEvent(LifetimeEventType.DecodeEnd),
                            )

                # ..clear states for finished requests
                finished_reqs = self.scheduler.pop_finished_requests()
                self.on_requests_finished_callback(finished_reqs)

                self.batches_in_pipeline.pop(0)
                self.batches_ret_futures.pop(0)

    async def start_event_loop(self):
        async def decode_step_loop():
            while True:
                await self._step()
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await asyncio.gather(decode_step_loop())

    def print_status(self):
        self.scheduler.print_status()

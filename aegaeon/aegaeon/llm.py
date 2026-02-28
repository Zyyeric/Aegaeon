from typing import Dict, List, Optional, Tuple, NoReturn, AsyncGenerator
from viztracer import VizTracer
import ray
import ray.exceptions
from ray.util.placement_group import PlacementGroup
import torch
import asyncio
import os
import itertools
import traceback
import time
import json
import ctypes

from aegaeon.block_manager import BlockManager
from aegaeon.config import ParallelConfig, NodeConfig, QuickLoaderConfig
from aegaeon.lifetime import LifetimeEvent, LifetimeEventType
from aegaeon.models import ModelType
from aegaeon.stage_engine import PrefillEngine, DecodeEngine, StepOutput
from aegaeon.prefill_dispatcher import PrefillDispatcher
from aegaeon.decode_dispatcher import DecodeDispatcher
from aegaeon.request import Request
from aegaeon.logger import init_logger
from aegaeon.utils import (
    Counter,
    ensure_outfile,
    compute_request_metrics,
    compute_request_latencies,
    TTFT_SLO,
    CudaRTLibrary,
)
from aegaeon.loader import QuickCache

logger = init_logger(__name__)

AEGAEON_CPU_CACHE_FILENAME = "/dev/shm/aegaeon_cpu_cache"

AEGAEON_MODEL_CACHE_FILENAME = "/dev/shm/aegaeon_model_cache"

AEGAEON_WORKER_NUM_CPUS = 16


class LLMService:
    """
    LLMService: Implements LLM inference service in Aegaeon.

    ## Overview

    This class serves as the endpoint API for sending `generate()` requests to
    the managed cluster of nodes with stage engines and workers.

              enqueue                       select
    Endpoint ---------> PrefillDispatcher  -------> PrefillEngine
                                                        |
                                                        | prefilled
                                            select      v
               (Done) <------ DecodeEngine <------- DecodeDispatcher
                                    |                   ^
                                    |-------------------|
                                   (optional) load-balance

    [Data Plane]    The data plane of Aegaeon is managed by StageEngines
                    separately using Ray. Each StageEngine controls a collection
                    of Workers as ray.Actors which occupy GPUs and perform
                    LLM inference asynchronously.
                    The per-node BlockManagers act as central managers of KV cache,
                    coordinating block movement between cache spaces on a node.

                    NOTE: currently, direct block movement between GPUs are not
                    implemented (termed "migration"). Instead, the CPU cache
                    is used explicitly as a staging buffer.

    [Control Plane] The control plane of Aegaeon is based on `asyncio` and is
                    also disaggregated with per-node Controllers. Each Controller
                    corresponds to a GPU node and may consist of the following entities
                    and their event loops:

                    - PrefillDispatcher: A unique entity that queues and dispatches
                        new requests to PrefillEngines.
                    - PrefillEngine: An entity with Worker(s) that executes the
                        Prefill stage of a request.
                    - DecodeDispatcher: A unique entity that queues and (re)dispatches
                        prefilled requests to DecodeEngines.
                    - DecodeEngine: An entity with Worker(s) that executes the
                        Decode stage of a request.

                    All event loops are non-blocking; data synchronization
                    is achieved via CUDA events dispatched to Workers and thus
                    happen in the data plane only when necessary. See the header
                    comment in `block_manager.py` for details.

                    The full control plane is implemented with the Controllers as
                    async ray actors, scaling up to arbirarily many nodes.
    """

    def __init__(self, cluster_config: List[NodeConfig]) -> None:
        self.counter = Counter()

        # TODO: this should be a per-node setting
        env_vars = {"FAST_SWITCH": "1"}
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            env_vars["AEGAEON_CUDA_VISIBLE_DEVICES"] = os.environ[
                "CUDA_VISIBLE_DEVICES"
            ]
        if "AEGAEON_LOG_FILE" in os.environ:
            env_vars["AEGAEON_LOG_FILE"] = os.environ["AEGAEON_LOG_FILE"]

        # Initialize per-node
        self.nodes: List[ray.ObjectRef[Controller]] = []

        init_handles = []
        for i, config in enumerate(cluster_config):
            node = (
                ray.remote(Controller)
                .options(
                    name=f"node_{i}",
                    runtime_env={"env_vars": env_vars},
                    resources={config.node_id: 0.01},
                )
                .remote()
            )
            init_handle = node.init.remote(config)
            self.nodes.append(node)
            init_handles.append(init_handle)
        # ..the node must first finish initialize()
        ray.get(init_handles)

        run_handles = []
        for node in self.nodes:
            run_handle = node.start_event_loops.remote()
            run_handles.append(run_handle)

        # ..and this should return the timeout exception if event loops are healthy
        def sanity_check():
            try:
                ray.get(run_handles, timeout=2.0)
            except ray.exceptions.GetTimeoutError:
                logger.info("(san-check) sound")

        sanity_check()


    def serve(self, requests: List[Request]) -> List[List[StepOutput]]:
        """
        Submit the list of requests to be served.
        """

        async def serve_one(req: Request):
            node_index = req.model.value % len(self.nodes) # simple lb
            step_outputs = await self.nodes[node_index].serve.remote(req)
            return step_outputs

        async def serve_all(requests: List[Request]):
            coros = [serve_one(req) for req in requests]
            return await asyncio.gather(*coros, return_exceptions=True)

        return asyncio.run(serve_all(requests))

    def replay(
        self,
        result_file: str,
        num_models: int,
        arrival_rate: float,
        inlen_scale: float = 1.0,
        outlen_scale: float = 1.0,
        duration: int = 120,
    ):
        from aegaeon.workload import syn_workload

        ensure_outfile(result_file)
        workload = syn_workload(
            model_num=num_models, arrival_rate=arrival_rate, 
            inlen_scale=inlen_scale, outlen_scale=outlen_scale,
            duration=duration,
        )

        reqs: List[Request] = []
        req_cnt = Counter()
        for model, arrival_time, prompt_token_ids, decode_tokens in workload:
            reqs.append(
                Request(
                    model,
                    arrival_time,
                    next(req_cnt),
                    prompt_token_ids=prompt_token_ids,
                    decode_tokens=decode_tokens,
                )
            )

        async def serve_one(req: Request):
            await asyncio.sleep(req.arrival_time)
            req.arrival_time = time.time()
            node_index = req.model.value % len(self.nodes) # simple lb
            step_outputs = await self.nodes[node_index].serve.remote(req)
            return step_outputs

        async def start_trace(latency: int):
            await asyncio.sleep(latency)
            handles = [node.start_trace.remote() for node in self.nodes]
            await asyncio.gather(*handles)

        async def stop_trace(latency: int):
            await asyncio.sleep(latency)
            handles = [node.stop_trace.remote() for node in self.nodes]
            await asyncio.gather(*handles)

        async def serve_all(requests: List[Request]):
            coros = [serve_one(req) for req in requests]

            result = await asyncio.gather(*coros, return_exceptions=True)
            return result[: len(requests)]

        # Execute the workload
        outputs = asyncio.run(serve_all(reqs))

        # Extract metrics
        metrics = {}
        for i, output in enumerate(outputs):
            if isinstance(output, list) and len(output) == reqs[i].decode_tokens:
                ttft, qos, per_token = compute_request_metrics(reqs[i], output)
                latencies = compute_request_latencies(reqs[i], output)
                metrics[reqs[i].request_id] = {
                    "ttft": ttft,
                    "qos": qos,
                    "per_token": per_token,
                    **latencies,
                }
            else:
                metrics[reqs[i].request_id] = {
                    "ttft": TTFT_SLO,
                    "qos": 0,
                    "per_token": [],
                }

        # Save metrics
        with open(result_file, "w") as f:
            json.dump(metrics, f)

    def reset(self):
        """
        Reset the service cluster.
        """
        ray.get([node.reset.remote() for node in self.nodes])

    # TODO: a FastAPI app like sllm


class Controller:
    """
    Manager of the control plane on a GPU node.
    """

    ### Singleton components ###
    _node_id: str = None
    _block_manager: BlockManager = None
    _quick_cache: QuickCache = None
    _cpu_cache_size: int = None
    _cpu_cache: torch.Tensor = None
    _viztracer: VizTracer = None

    _prefill_dispatcher: Optional[PrefillDispatcher] = None
    _prefill_engines: Dict[int, PrefillEngine] = {}
    _decode_dispatcher: Optional[DecodeDispatcher] = None
    _decode_engines: Dict[int, DecodeEngine] = {}

    # Mapping: request_id -> [StepOutput*]
    # - created when request is submitted
    # - cleared when request is finished
    _request_outputs: Dict[int, List[StepOutput]] = {}

    @staticmethod
    def _init_placement_group(
        node_id: str,
        num_engines: int,
        # TODO: parallel config for engines goes here..
    ) -> Tuple[PlacementGroup, List[List[int]]]:
        """
        Create placement groups for all engines and workers on the node.
        """
        pp_size = 1
        tp_size = 1
        device_ids_list = [[i] for i in range(num_engines)]
        pg = ray.util.placement_group(
            [{"CPU": AEGAEON_WORKER_NUM_CPUS, node_id: 0.01}]
            * num_engines,  # engine_id i -> bundle i
            strategy="STRICT_PACK",  # ensures all workers and stage engines are on the same node
        )
        ray.get(pg.ready(), timeout=1000)
        return (pg, device_ids_list)

    @staticmethod
    def _init_cpu_cache(
        cudart: CudaRTLibrary,
        cpu_num_slabs: int,
        cpu_slab_size_bytes: int,
    ) -> torch.Tensor:
        if not os.path.isfile(AEGAEON_CPU_CACHE_FILENAME):
            logger.info(
                f"Creating the shared CPU cache from scratch at {AEGAEON_CPU_CACHE_FILENAME}. "
                "This might take some time."
            )
        else:
            logger.info(f"Found shared CPU cache at {AEGAEON_CPU_CACHE_FILENAME}")

        tensor = torch.from_file(
            AEGAEON_CPU_CACHE_FILENAME,
            shared=True,
            size=cpu_num_slabs * cpu_slab_size_bytes // 2,
            dtype=torch.float16,
        )
        cudart.cudaHostRegister(ctypes.c_void_p(tensor.data_ptr()), tensor.nbytes)
        return tensor

    @staticmethod
    def _on_new_step_output(
        request_id: int,
        step_output: StepOutput,
    ) -> None:
        """Called when a stage engine generates a new step output."""
        Controller._check_initialized()
        _ctrl._request_outputs[request_id].append(step_output)

    @staticmethod
    def _on_new_lifetime_event(
        request_id: int,
        event: LifetimeEvent,
        dont_add_if_dup: bool = False,
    ) -> None:
        """Called when a stage engine generates a new step output."""
        pass

    @staticmethod
    def _on_requests_finished(finished_reqs: List[Request]) -> None:
        """Called when a stage engine finishes a batch of requests."""
        Controller._check_initialized()
        _ctrl._block_manager.free_blocks_batched(finished_reqs)

    @staticmethod
    def _check_initialized():
        global _ctrl
        if _ctrl._node_id is None:
            raise ValueError(f"Controller uninitialized. Call Controller.init() first.")

    @staticmethod
    async def init(
        config: NodeConfig,
    ) -> None:
        """
        Starts the control plane event loops and data plane workers on a node.
        """
        global _ctrl
        if _ctrl._node_id is not None:
            raise ValueError(
                f"Controller.init() has already been called on {_ctrl._node_id}"
            )

        node_id = config.node_id
        num_prefill_engines = config.num_prefill_engines
        num_decode_engines = config.num_decode_engines
        cpu_num_slabs = config.cpu_num_slabs
        cpu_slab_size_bytes = config.cpu_slab_size_bytes
        logger.info(f"({node_id}) Initializing Controller")

        assert num_prefill_engines >= 0 and num_decode_engines >= 0

        # Set visible devices
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # get rid of whatever ray sets
        if "AEGAEON_CUDA_VISIBLE_DEVICES" in os.environ:
            # respect custom settings
            devices = os.environ["AEGAEON_CUDA_VISIBLE_DEVICES"]
            logger.info(f"({node_id}) Found AEGAEON_CUDA_VISIBLE_DEVICES={devices}")
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
        assert torch.cuda.is_available(), "Aegaeon is only available on CUDA devices"

        # Initialize cluster resources
        logger.info(f"({node_id}) Initializing block manager")
        block_manager = BlockManager(cpu_num_slabs, cpu_slab_size_bytes)
        logger.info(
            f"({node_id}) Initializing pinned cpu cache ({cpu_num_slabs} x {cpu_slab_size_bytes/(1024**3):.1f}GB)"
        )
        cpu_cache = Controller._init_cpu_cache(
            block_manager.cudart, cpu_num_slabs, cpu_slab_size_bytes
        )

        logger.info(f"({node_id}) Initializing QuickCache")
        quick_cache = QuickCache(
            block_manager.cudart,
            AEGAEON_MODEL_CACHE_FILENAME,
            config.model_cache_size,
            model_cache_snapshot=None,
        )
        for model in config.cached_models:
            quick_cache.cache_model(model.path())
        model_cache_snapshot = quick_cache.snapshot()

        logger.info(f"({node_id}) Initializing placement group")
        placement_group, device_ids_list = Controller._init_placement_group(
            node_id, num_prefill_engines + num_decode_engines
        )

        # Initialize the control plane components
        logger.info(f"({node_id}) Creating components")
        parallel_config = ParallelConfig()
        loader_config = QuickLoaderConfig(
            model_cache_filename=AEGAEON_MODEL_CACHE_FILENAME,
            model_cache_size=config.model_cache_size,
            model_cache_snapshot=model_cache_snapshot,
        )

        prefill_engines = {}
        for i in range(num_prefill_engines):
            prefill_engines[i] = PrefillEngine(
                i,
                parallel_config,
                loader_config,
                device_ids_list[i],
                placement_group,
                Controller._on_new_step_output,
                Controller._on_new_lifetime_event,
                Controller._on_requests_finished,
                policy=config.prefill_sched_policy,
            )
        prefill_dispatcher = None
        if prefill_engines:
            prefill_dispatcher = PrefillDispatcher(policy=config.prefill_disp_policy)

        decode_engines = {}
        for i in range(num_prefill_engines, num_prefill_engines + num_decode_engines):
            decode_engines[i] = DecodeEngine(
                i,
                parallel_config,
                loader_config,
                device_ids_list[i],
                placement_group,
                Controller._on_new_step_output,
                Controller._on_new_lifetime_event,
                Controller._on_requests_finished,
            )
        decode_dispatcher = None
        if decode_engines:
            decode_dispatcher = DecodeDispatcher() if decode_engines else None

        # Install the controller
        logger.info(f"({node_id}) Installing components")

        _ctrl._node_id = node_id
        _ctrl._block_manager = block_manager
        _ctrl._quick_cache = quick_cache
        _ctrl._cpu_cache_filename = AEGAEON_CPU_CACHE_FILENAME
        _ctrl._cpu_cache_size = cpu_num_slabs * cpu_slab_size_bytes
        _ctrl._cpu_cache = cpu_cache
        _ctrl._viztracer = VizTracer(
            tracer_entries=1000000, min_duration=200, log_async=False
        )
        _ctrl._prefill_dispatcher = prefill_dispatcher
        _ctrl._prefill_engines = prefill_engines
        _ctrl._decode_dispatcher = decode_dispatcher
        _ctrl._decode_engines = decode_engines

        dispatchers = (
            [_ctrl._prefill_dispatcher] if _ctrl._prefill_dispatcher is not None else []
        ) + ([_ctrl._decode_dispatcher] if _ctrl._decode_dispatcher is not None else [])

        # Initialize stage engines and workers
        logger.info(f"({node_id}) Initializing stage engines")
        await asyncio.gather(
            *map(lambda engine: engine.initialize(), _ctrl._prefill_engines.values()),
            *map(lambda engine: engine.initialize(), _ctrl._decode_engines.values()),
            *map(lambda dispatcher: dispatcher.initialize(), dispatchers),
        )

    @staticmethod
    async def start_event_loops():
        Controller._check_initialized()
        dispatchers = (
            [_ctrl._prefill_dispatcher] if _ctrl._prefill_dispatcher is not None else []
        ) + ([_ctrl._decode_dispatcher] if _ctrl._decode_dispatcher is not None else [])

        # Runs forerver
        try:
            await asyncio.gather(
                # Component loops
                _ctrl._block_manager.start_event_loop(),
                *map(
                    lambda engine: engine.start_event_loop(),
                    _ctrl._prefill_engines.values(),
                ),
                *map(
                    lambda engine: engine.start_event_loop(),
                    _ctrl._decode_engines.values(),
                ),
                *map(lambda dispatcher: dispatcher.start_event_loop(), dispatchers),
            )

        except Exception as e:
            # ..something wrong in the control plane
            traceback.print_exception(type(e), e, e.__traceback__)
            for task in asyncio.all_tasks():
                task.cancel()
            await asyncio.gather(*asyncio.all_tasks(), return_exceptions=True)
            raise

    @staticmethod
    async def serve(request: Request) -> List[StepOutput]:
        """Sumbit and serve a new request to the node, returning its tokens."""
        SLEEP_FOR_STEP_OUTPUTS = 0.005

        Controller._check_initialized()
        outputs: List[StepOutput] = []
        _ctrl._request_outputs[request.request_id] = outputs

        Controller._on_new_lifetime_event(
            request.request_id, LifetimeEvent(LifetimeEventType.Issued)
        )
        if _ctrl._prefill_dispatcher is None:
            raise ValueError(
                f"Submitting a new request to {_ctrl._node_id} which has no "
                "prefill engine configured."
            )
        _ctrl._prefill_dispatcher.add(request)

        while True:
            try:
                if outputs and outputs[-1].is_finished:
                    break
                await asyncio.sleep(SLEEP_FOR_STEP_OUTPUTS)
            except asyncio.CancelledError:
                # ..exceptions should be handled by the engine
                return []
        return outputs

    async def reset():
        """Reset the control plane and workers."""
        Controller._check_initialized()

        global _ctrl
        _ctrl._block_manager.reset()
        # TODO
        _ctrl._prefill_dispatcher.reset()
        _ctrl._decode_dispatcher.reset()
        for engine in _ctrl._prefill_engines.values():
            engine.reset()
        for engine in _ctrl._decode_engines.values():
            engine.reset()

        _ctrl._request_outputs.clear()

    async def start_trace(self):
        global _ctrl
        Controller._check_initialized()
        _ctrl._viztracer.start()

    async def stop_trace(self):
        global _ctrl
        Controller._check_initialized()
        _ctrl._viztracer.stop()
        _ctrl._viztracer.save("trace-control.json")

    @staticmethod
    def remote_call_all_workers_async(func_name: str, *args, **kwargs) -> list:
        """
        Call func_name asynchronously on all workers across all engines; return the futures immediately.
        """
        global _ctrl
        Controller._check_initialized()

        handlers = []
        for engine in itertools.chain(
            Controller.prefill_engines().values(), Controller.decode_engines().values()
        ):
            handlers.extend(
                engine._remote_call_all_workers_async(func_name, *args, **kwargs)
            )
        return handlers

    @staticmethod
    def node_id() -> str:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._node_id

    @staticmethod
    def block_manager() -> BlockManager:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._block_manager

    @staticmethod
    def get_tokenizer(name: str):
        global _ctrl
        Controller._check_initialized()
        return _ctrl._quick_cache.get_tokenizer(name)

    @staticmethod
    def cpu_cache_filename() -> str:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._cpu_cache_filename

    @staticmethod
    def cpu_cache_size() -> int:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._cpu_cache_size

    @staticmethod
    def prefill_dispatcher() -> Optional[PrefillDispatcher]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._prefill_dispatcher

    @staticmethod
    def decode_dispatcher() -> Optional[DecodeDispatcher]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._decode_dispatcher

    @staticmethod
    def prefill_engines() -> Dict[int, PrefillEngine]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._prefill_engines

    @staticmethod
    def decode_engines() -> Dict[int, DecodeEngine]:
        global _ctrl
        Controller._check_initialized()
        return _ctrl._decode_engines


_ctrl: Controller = Controller()

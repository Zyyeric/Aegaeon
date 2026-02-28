import asyncio
import heapq
from collections import defaultdict
from typing import Dict, Literal, TYPE_CHECKING

from aegaeon.models import ModelType
from aegaeon.config import ModelConfig, get_model_config
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.block_manager import BlockManager, BLOCK_SIZE
from aegaeon.utils import TTFT_SLO, TPOT_SLO, estimate_switch_time

if TYPE_CHECKING:
    from aegaeon.decode_scheduler import DecodeScheduler

logger = init_logger(__name__)

# Sleep for this many seconds when there is no request to dispatch,
# or when no scheduler can be dispatched to.
SLEEP_WHEN_DISPATCH_PENDS = 0.003

# Sleep for this many seconds in each event loop, useful for debugging.
SLEEP_IN_EACH_EVENT_LOOP = 0


class DecodeDispatcher:
    """
    Dispatcher of decode requests to individual schedulers in decode engines.

    ## Event loops
    - Dispatch
        - Pop a request for dispatching
        - Find a decode scheduler to dispatch to

    Another API is provided for re-dispatching of batches.
    """

    def __init__(
        self,
        policy: Literal["fcfs", "sjf"] = "fcfs",
    ):
        self.policy = policy
        self.node_id: str = None

        # Mapping: model_type -> [request*]
        self.model_queues: Dict[ModelType, list] = defaultdict(list)

        match policy:
            case "fcfs":
                self._add = self._add_fcfs
            case "sjf":
                self._add = self._add_sjf
            case _:
                raise ValueError(f"Unsupported decode dispatcher policy {policy}")

    def __repr__(self):
        return f"{self.node_id}|decode dispatcher"

    async def initialize(self):
        """
        Register decode schedulers as dispatch targets.
        """
        from aegaeon.llm import Controller

        self.node_id = Controller.node_id()
        self.schedulers = [
            engine.scheduler for engine in Controller.decode_engines().values()
        ]

    def add(self, batch: BatchedRequests):
        """
        Add the given batch to the dispatcher queues, waiting to be dispatched.
        """
        for request in batch.requests:
            assert not request.is_finished
            self._add(request)

    def _add_fcfs(self, request: Request):
        self.model_queues[request.model].append(request)

    def _add_sjf(self, request: Request):
        heapq.heappush(
            self.model_queues[request.model], (request.get_total_len(), request)
        )

    def dispatch(self) -> bool:
        """
        Dispatch requests in the dispatcher queue to decode schedulers,
        until no requests are left.
        """

        dispatched = False
        for queue in self.model_queues.values():
            for i in range(len(queue)):
                dispatched = True
                request = queue[i] if isinstance(queue[i], Request) else queue[i][1]
                pick_scheduler: "DecodeScheduler" = None
                for scheduler in self.schedulers:
                    if scheduler.try_accept(request):
                        # ..accepted
                        pick_scheduler = scheduler
                        break
                    else:
                        # ..not accepted by an existing batch; find a scheduler
                        # with the lowest load
                        if pick_scheduler is None or len(
                            scheduler.decode_batches
                        ) < len(pick_scheduler.decode_batches):
                            pick_scheduler = scheduler
                else:
                    pick_scheduler.decode_batches.append(
                        (0, BatchedRequests([request]))
                    )
                logger.info(
                    f"({self}) request {request.request_id} -> ({pick_scheduler.engine})"
                )
            queue.clear()

        return dispatched

    # TODO: API for re-dispatch of a batch

    async def start_event_loop(self):
        async def dispatch_loop():
            while True:
                if not self.dispatch():
                    await asyncio.sleep(SLEEP_WHEN_DISPATCH_PENDS)
                await asyncio.sleep(SLEEP_IN_EACH_EVENT_LOOP)

        await dispatch_loop()

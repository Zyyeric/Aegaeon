import time
import asyncio
from typing import List, Optional, Tuple, AsyncIterable, TYPE_CHECKING
from collections import deque

from aegaeon.models import ModelType
from aegaeon.config import get_model_config
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.estimator import cache_estimators, make_estimator, DecodeEstimator
from aegaeon.block_manager import BLOCK_LOCATION_CPU, BLOCK_SIZE
from aegaeon.utils import TPOT_SLO, estimate_switch_time, prod

if TYPE_CHECKING:
    from aegaeon.stage_engine import DecodeEngine

logger = init_logger(__name__)


class DecodeScheduler:
    """
    Scheduler for the decode stage.

    NOTE: PP is not supported for this stage.
    """

    def __init__(
        self,
        engine: "DecodeEngine",
        max_batch_quota: float = 4.0,
        batch_lookahead: int = 4 * BLOCK_SIZE,
    ):
        self.engine = engine
        self.max_batch_quota = max_batch_quota
        self.batch_lookahead = batch_lookahead

        # Round and turn
        self.round: int = 0
        self.turn: int = 0
        self._turn_start: Optional[float] = None
        self._turn_request_moved: bool = False

        # Batches for the current round
        self.decode_batches: deque[Tuple[float, BatchedRequests]] = deque()

        # Internal quota states for logging
        self.n: List[float] = []
        self.c0: float = 0
        self.alpha: float = 0

        # Execution loop
        self.loop: AsyncIterable[Tuple[BatchedRequests, Optional[ModelType]]] = (
            self._run()
        )

    def _get_block_needed(self, length: int):
        return (length + BLOCK_SIZE - 1) // BLOCK_SIZE

    def _batch_fits(self, batch: BatchedRequests, req: Request) -> bool:
        # TODO: temporary fix
        if self.engine.model_config is None:
            return True

        seq_lens: List[int] = [
            request.get_total_len() + self.batch_lookahead for request in batch.requests
        ]
        seq_lens.append(req.get_total_len() + self.batch_lookahead)
        return (
            sum(self._get_block_needed(length) for length in seq_lens)
            # TODO: this is buggy; must pre-compute max_blocks for all batches
            <= self.engine.block_manager.get_gpu_num_max_blocks(self.engine.engine_id)
        )

    def _batch_fits_strict(self, batch: BatchedRequests) -> bool:
        seq_lens: List[int] = [
            request.get_total_len() + BLOCK_SIZE for request in batch.requests
        ]
        return sum(
            self._get_block_needed(length) for length in seq_lens
        ) <= self.engine.block_manager.get_gpu_num_max_blocks(self.engine.engine_id)

    def try_accept(self, request: Request) -> bool:
        """Try accepting a new decode request to an existing batch."""
        for _, batch in self.decode_batches:
            if batch.model == request.model and self._batch_fits(batch, request):
                batch.add_request(request)
                # self._assign_quota()
                return True
        return False

    def pop_finished_requests(self) -> List[Request]:
        """Pop finished requests from the current batch."""
        return self.decode_batches[0][1].pop_finished_requests()

    async def get_next_batch(self) -> BatchedRequests:
        """Get the decode batch for the next step.

        Optionally return prefetch model for the next turn; turn and round may be
        incremented."""

        decision: Tuple[BatchedRequests, Optional[ModelType]] = await anext(self.loop)
        batch, prefetch_model = decision
        if len(batch) == 0:
            return BatchedRequests()

        # Issue engine switch if needed
        if (
            self.engine.model_config is None
            or self.engine.model_config.model != batch.model
        ):
            # No request is logically on a decode engine's GPU at this
            # time (`_run` will issue move-outs at end of turns).
            self.engine.switch(
                get_model_config(batch.model),
                prefetch_model_config=get_model_config(prefetch_model),
            )

        # Check that the batch still fits; this might not be true when
        # a batch grows too much within its quota.
        if not self._batch_fits_strict(batch):
            # XXX: ..this should be rare
            logger.info(
                f"({self.engine}) batch {[r.request_id for r in batch.requests]} truncated"
            )
            truncated = BatchedRequests()
            for i, request in enumerate(batch.requests):
                if not self._batch_fits(truncated, request):
                    break
                truncated.add_request(request)
            delayed = BatchedRequests(batch.requests[i:])
            await self.engine.block_manager.move_requests(
                delayed.requests, BLOCK_LOCATION_CPU
            )

            # ..split the batch and adjust quota
            self.decode_batches.popleft()
            self.decode_batches.appendleft((0, delayed))
            self.decode_batches.appendleft((0, truncated))
            self._assign_quota()
            batch = truncated

        # We have the batch; issue move-in operations
        request_moved = await self.engine.block_manager.move_requests(
            batch.requests, self.engine.engine_id
        )
        self._turn_request_moved = self._turn_request_moved or request_moved

        return batch

    async def _run(self) -> AsyncIterable[Tuple[BatchedRequests, Optional[ModelType]]]:
        """Main execution loop."""
        while True:
            # Start of the round
            self._enter_round()

            if len(self.decode_batches) == 0:
                # ..no requests; wait
                yield (BatchedRequests(), None)
                continue

            self.print_status()
            n_turn = len(self.decode_batches)
            while self.turn < n_turn and len(self.decode_batches) > 0:
                # ..pick a batch

                n_step = 0
                start = time.time()
                while True:
                    quota, batch = self.decode_batches[0]
                    if len(batch) == 0:
                        # ..all finished
                        break

                    if n_step == 0:
                        logger.info(
                            f"({self.engine}) <round {self.round}|{self.turn}> {[r.request_id for r in batch.requests]} {quota}"
                        )

                    # ..determine prefetch
                    prefetch_model = None
                    for i in range(1, n_turn - self.turn):
                        if i >= len(self.decode_batches):
                            break
                        if self.decode_batches[i][1].model != batch.model:
                            prefetch_model = self.decode_batches[i][1].model
                            break
                    yield (batch, prefetch_model)

                    n_step += 1
                    # ..by now the batch is already stepped once; starts counting quota
                    if self._turn_start is None:
                        self._turn_start = time.time()
                    if self._turn_start + quota < time.time():
                        # ..out of quota
                        break

                # ..turn ends
                # XXX: wait for all move-ins before we end this turn.
                # This enables the block manager to clear all move-in events at a
                # model switch.
                end = time.time()
                logger.info(
                    f"({self.engine}) <round {self.round}|{self.turn}> {n_step} steps; e2e {end-start:.2f}s; expected {quota:.2f}s; first-step {self._turn_start-start:.2f}s"
                )
                if self._turn_request_moved:
                    await asyncio.wait(
                        self.engine._remote_call_all_workers_async(
                            "wait_for_all_move_in"
                        )
                    )
                if len(self.decode_batches) > 1:
                    # XXX: optimization; no need to move out requests if
                    # this is the only batch.
                    await self.engine.block_manager.move_requests(
                        batch.requests, BLOCK_LOCATION_CPU
                    )  # TODO: re-dispatch
                self.turn += 1
                self._turn_start = None
                self._turn_request_moved = False
                if len(batch) > 0:
                    self.decode_batches.rotate(-1)
                else:
                    self.decode_batches.popleft()

    def _enter_round(self):
        """Top of a round; form the decode batches for the round."""
        if len(self.decode_batches) == 0:
            return

        # Form decode batches
        decode_reqs: List[Request] = sum(
            (batch.requests for _, batch in self.decode_batches), []
        )
        decode_batches: List[BatchedRequests] = []
        decode_reqs.sort()  # Sort in FCFS order

        for req in decode_reqs:
            for batch in decode_batches:
                if batch.model == req.model and self._batch_fits(batch, req):
                    batch.add_request(req)
                    break
            else:
                decode_batches.append(BatchedRequests([req]))

        # Ensure a fair order
        decode_batches.sort(key=lambda batch: id(batch.model))

        # Apply updates
        self.decode_batches.clear()
        self.decode_batches.extend((0, batch) for batch in decode_batches)
        self._assign_quota()
        self.round += 1
        self.turn = 0

    def _assign_quota(self):
        """Assign quota for the current decode batches."""
        if len(self.decode_batches) == 0:
            return

        # ..compute alpha: how well can we satisfy the SLOs?
        d = []  # decode latency for one token
        models: set[ModelType] = set()  # unique set of models
        for _, batch in self.decode_batches:
            est = make_estimator(
                DecodeEstimator, model=batch.model, device=self.engine.device_type
            )
            d.append(
                est.predict(-1, -1, [req.get_total_len() for req in batch.requests])
            )
            models.add(batch.model)
        n = [TPOT_SLO / d[i] for i in range(len(d))]  # tpot / d[i]
        self.n = n

        n_prod = prod(n)
        n_sum = sum(n_prod / ni for ni in n)

        c0 = sum(
            estimate_switch_time(
                model, self.engine.device_type, self.engine.parallel_config
            )
            for model in models
        )  # model switching overhead
        self.c0 = c0

        alpha = max(
            c0 / min(n) / self.max_batch_quota + sum(1 / ni for ni in n),
            0.5,
        )
        self.alpha = alpha

        # ..from which we obtain the quota that each batch should run for
        for i in range(len(d)):
            quota = c0 * (n_prod / n[i]) / (n_prod * alpha - n_sum)
            assert quota > 0
            self.decode_batches[i] = (quota, self.decode_batches[i][1])

    def print_status(self):
        batches = " |> ".join(
            [
                f"{[request.request_id for request in batch.requests]} {quota:.2f}"
                for quota, batch in self.decode_batches
            ]
        )
        logger.info(
            f"({self.engine}) batches: {batches} "
            f"(n={self.n}, c0={self.c0}, alpha={self.alpha})"
        )

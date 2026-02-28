import math
import ray
import asyncio
import torch
import os

from typing import List, Callable, Tuple, Dict, TypeAlias, Optional, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
from collections import deque, defaultdict

from aegaeon.config import BLOCK_SIZE
from aegaeon.logger import init_logger
from aegaeon.request import Request, BatchedRequests
from aegaeon.utils import Counter, CudaRTLibrary, rebuild_cuda_event, prod

logger = init_logger(__name__)

# Sleep for this many seconds between the daemon runs
SLEEP_IN_DAEMON_LOOP = 8

# Sleep for this many seconds between attempts to reclaim a move event
SLEEP_WHEN_RECLAIM_PENDS = 0.01

"""
XXX: migration between GPU memory is currently not implemented. The
following description talks about migration at the principle level
but not its actual implementation.

TODO: the whole block_shape mechanism currently does not consider model
parallelism. Add support for it once the non-parallel case runs.

## Synchonization

The block manager need to deal with 4 types of synchronization:

Requirement 1: When calling step(), we need to ensure that all blocks that
    are related to the request are on GPU, and have finished movement if
    we've issued a swap-in/migration operation on them before.

Requirement 2: When issuing movements, we need to ensure that the target blocks
    (i.e. blocks that we are copying to) are free. If there were another
    movements which use them as source, we must wait for those operations
    to finish.

Requirement 3: When issuing movements, the source blocks must be ready.
    For example, if we have issued swapping-in on request #0 before and now 
    are about to migrate it, then the swap-in operation must have finished.

Requirement 4: When calling switch(), we need to ensure that all blocks that
    are related to previous requests are not on GPU, and have finished movement if
    we've issued a swap-out/migration operation on them before.

To facilitate these synchronization efficiently, we rely on CUDA events.
After every movement operation, we push a CUDA event into the corresponding
CUDA stream. We maintain a dict called `move_events`, which maps request_ids,
to a list of MoveEvents which include the cuda events related to that request. 
These events are then dispatched by the block manager to workers, which synchronizes
their streams correctly for the events. To be exact:

1. When calling step() / switch(), we iterate through all requests in the batch, retrieve
the CUDA event from the worker-local `move_event_table` (populated when swap_blocks
and migrate_blocks are invoked on the worker side), and wait for it to finish. 
This ensures Requirement 1 / 4.

2. When issuing a movement, we call event.wait() on the corresponding movement
event and use the corresponding stream (i.e., if we are swapping-in then it is the
swap-in stream) as the argument. Wait() makes all future work submitted to the given 
stream wait for this event, so the movement operation that we're issuing will wait 
for the previous movement operation to finish.
This ensures Requirement 3.

3. For Requirement 2, a possible solution is to mark the CUDA event of every block
and wait for them to finish. However, this is not efficient. Instead, for
each cache space (CPU/GPUs), we maintain two lists: `free_blocks_list` and
`move_blocks_list`. The former contains blocks that are we are sure to
be free (it does not contain any useful data), and the latter contains blocks
involving movement operations that may not have finished. 

Model switches (and thus registering a new GPU cache space) serve as periodical
barriers for the block manager to reclaim blocks from move_blocks_list
synchronously: all movements to a GPU space must have finished before a switch,
and all movements from it no longer relevant. If unfortunately the free_blocks_list
is still insufficient at some allocation, `_get_free_blocks()` will perform
reclaimation itself by gathering events and asynchronously querying them.
This effectively pends the corresponding task without choking the entire
control plane (for example, other engines can still operate unless they
also want to allocate on this blocked location). For details involving race
conditions from workers freeing events and multiple tasks reclaiming the
same move_blocks_list, refer to `_reclaim_XXX()`.
This ensures Requirement 2 with a small overall cost. 
"""

""">=0 for GPU engine ID, -1 for CPU"""
BlockLocation: TypeAlias = int

BLOCK_LOCATION_CPU: BlockLocation = -1


def is_gpu(location: BlockLocation) -> bool:
    return location >= 0


@dataclass
class BlockTable:
    # Location of the blocks
    location: BlockLocation
    # Block shape
    block_shape: Tuple[int, ...]
    # List of block_ids
    blocks: List[int]


@dataclass(eq=False)
class MoveEvent:
    # Serial for ordering move events.
    serial: int
    # Requests involved in this movement
    requests: List[Request]
    # Block shape
    block_shape: Tuple[int, ...]
    # Source block location
    source_location: BlockLocation
    # Target block location
    target_location: BlockLocation
    # List of source blocks
    source_blocks: List[int]

    # Corresponding CUDA events (ray futures of event handles, one for each Worker)
    event_handle_refs: List[ray.ObjectRef]
    _event_handles: Optional[list] = None

    async def _gather_event_handles(self) -> None:
        if self._event_handles is None:
            handles = await asyncio.gather(*self.event_handle_refs)
            # this checks for multiple callers
            if self._event_handles is None:
                self._event_handles = handles

    def event_handles(self) -> Optional[list]:
        return self._event_handles

    def is_swap(self) -> bool:
        return is_gpu(self.source_location) != is_gpu(self.target_location)

    def engine_id(self) -> int:
        """Get the ID of the engine that is responsible for the movement.

        There are 3 types of movement: (1) swap in, (2) swap out, and (3)
        migrate in. For (1) and (2) this is the GPU side; for (3) its the
        target GPU side.
        """
        match (is_gpu(self.source_location), is_gpu(self.target_location)):
            case (_, True):
                return self.target_location
            case (True, False):
                return self.source_location
            case (False, False):
                raise NotImplementedError


class BlockManager:
    """A Block Manager that maintains all key-value caches at the block-level.

    The block manager is first initialized only for the unified CPU swap space.
    GPU cache spaces can then be registered (and replaced) subsequently.

    For subroutines and algorithms related to swapping, please refer to
    the header comment block.

    ## Event loops
    - Daemon
        - log_block_usage
    """

    def __init__(
        self,
        cpu_num_slabs: int,
        cpu_slab_size_bytes: int,
    ):
        self.block_size = BLOCK_SIZE
        self.dtype_size = 2
        self.cudart = CudaRTLibrary()
        self.event_counter = Counter()

        # CPU allocator
        self.cpu_num_slabs = cpu_num_slabs
        self.cpu_slab_size_bytes = cpu_slab_size_bytes

        # Mapping: block shape -> list of blocks
        self.cpu_free_blocks: Dict[Tuple[int, ...], deque[int]] = defaultdict(deque)
        self.cpu_free_slabs: list[int] = [i for i in range(self.cpu_num_slabs)]

        # Mapping: slab index -> number of used blocks within
        self.cpu_slab_usage: list[int] = [0 for _ in range(self.cpu_num_slabs)]
        # Mapping: slab index -> shape of blocks in the slab
        self.cpu_block_shape: list[Tuple[int, ...]] = [
            None for _ in range(self.cpu_num_slabs)
        ]

        # GPU allocator
        # Mapping: engine_id -> block_shape
        self.gpu_block_shape: Dict[int, Tuple[int, ...]] = {}
        # Mapping: engine_id -> num_max_blocks
        self.gpu_num_max_blocks: Dict[int, int] = {}
        # Mapping: engine_id -> free_blocks
        self.gpu_free_blocks: Dict[int, List[int]] = {}
        # Mapping: engine_id -> async func
        self.remote_callables: Dict[int, Callable] = {}

        # Movement
        # Mapping: block_shape -> list of MoveEvents (from cpu)
        self.move_events_cpu: Dict[Tuple[int, ...], deque[MoveEvent]] = defaultdict(
            deque
        )
        # Mapping: block_shape -> number of moving blocks (from cpu)
        self.move_cpu_num_blocks: Dict[Tuple[int, ...], int] = defaultdict(int)
        # Mapping: block_shape -> lastest reclaim event
        self.move_cpu_locks: Dict[Tuple[int, ...], Optional[asyncio.Event]] = (
            defaultdict(lambda: None)
        )
        # Mapping: engine_id -> list of MoveEvents (from gpu)
        self.move_events_gpu: Dict[int, deque[MoveEvent]] = defaultdict(deque)
        # Mapping: engine_id -> number of moving blocks (from gpu)
        self.move_gpu_num_blocks: Dict[int, int] = {}
        # Mapping: engine_id -> lastest reclaim event
        self.move_gpu_locks: Dict[int, Optional[asyncio.Event]] = defaultdict(
            lambda: None
        )
        # Mapping: request_id -> latest CUDA events (as ray future), one for each Worker
        self.latest_events: Dict[int, List[ray.ObjectRef]] = {}
        # List of freed request IDs whose resources are not yet cleared in workers
        self.zombie_request_ids: List[int] = []

        # Global block table
        # Mapping: request_id => BlockTable
        self.block_tables: Dict[int, BlockTable] = {}

    def _get_block_size_bytes(self, block_shape: Tuple[int, ...]) -> int:
        return prod(block_shape) * self.dtype_size

    def _get_slab_range(self, slab: int, block_size_bytes: int) -> Tuple[int, int]:
        start = math.ceil(slab * self.cpu_slab_size_bytes / block_size_bytes)
        end = math.floor((slab + 1) * self.cpu_slab_size_bytes / block_size_bytes)
        return (start, end)

    def _reclaim_gpu_move_blocks_once(
        self,
        move_event: MoveEvent,
    ) -> int:
        self.move_gpu_num_blocks[move_event.source_location] -= len(
            move_event.source_blocks
        )
        self.gpu_free_blocks[move_event.source_location].extend(
            move_event.source_blocks
        )
        return len(move_event.source_blocks)

    def _reclaim_cpu_move_blocks_once(
        self,
        move_event: MoveEvent,
    ) -> int:
        block_size_bytes = self._get_block_size_bytes(move_event.block_shape)
        self.move_cpu_num_blocks[move_event.block_shape] -= len(
            move_event.source_blocks
        )
        self.cpu_free_blocks[move_event.block_shape].extend(move_event.source_blocks)
        # ..update slab usage
        for block in move_event.source_blocks:
            slab = block * block_size_bytes // self.cpu_slab_size_bytes
            self.cpu_slab_usage[slab] -= 1
        return len(move_event.source_blocks)

    def reset(self):
        self.event_counter = Counter()

        self.cpu_free_blocks.clear()
        self.cpu_free_slabs = [i for i in range(self.cpu_num_slabs)]

        self.cpu_slab_usage = [0 for _ in range(self.cpu_num_slabs)]
        self.cpu_block_shape = [None for _ in range(self.cpu_num_slabs)]

        self.gpu_block_shape.clear()
        self.gpu_num_max_blocks.clear()
        self.gpu_free_blocks.clear()

        self.remote_callables.clear()

        self.move_events_cpu.clear()
        self.move_cpu_num_blocks.clear()
        self.move_cpu_locks.clear()
        self.move_events_gpu.clear()
        self.move_gpu_num_blocks.clear()
        self.move_gpu_locks.clear()
        self.latest_events.clear()
        self.zombie_request_ids.clear()

        self.block_tables.clear()

    def register_gpu_space(
        self,
        engine_id: int,
        num_max_blocks: int,
        block_shape: Tuple[int, ...],
        remote_callable: Callable,
    ) -> None:
        """Register a new GPU cache space with the given config.

        Also reclaims all blocks moving from and to the old cache space.
        """

        if self.gpu_free_blocks.get(engine_id, None) is not None:
            # Replacing an old GPU space; make sure that no requests are still in the old space
            for req_id, block_table in self.block_tables.items():
                if block_table.location == engine_id:
                    raise ValueError(
                        f"Error registering new GPU cache space for engine_id {engine_id}; "
                        f"Request #{req_id} still resides in the cache."
                    )

        if engine_id in self.move_events_gpu:
            self.move_events_gpu.clear()
        self.move_gpu_num_blocks[engine_id] = 0

        if (old_block_shape := self.gpu_block_shape.get(engine_id, None)) is not None:
            # ..reclaim CPU blocks moving to the old space
            move_events_cpu = self.move_events_cpu[old_block_shape]
            num_events = 0
            num_blocks = 0
            for _ in range(len(move_events_cpu)):
                move_event = move_events_cpu[0]

                if move_event.target_location == engine_id:
                    move_events_cpu.popleft()
                    num_events += 1
                    num_blocks += self._reclaim_cpu_move_blocks_once(move_event)
                else:
                    move_events_cpu.rotate(-1)

            if num_events > 0:
                logger.debug(
                    f"Reclaimed {num_events} events, {num_blocks} CPU blocks for block shape {block_shape}"
                )
                # self._reclaim_cpu_slabs()

        self.gpu_block_shape[engine_id] = block_shape
        self.gpu_num_max_blocks[engine_id] = num_max_blocks
        self.gpu_free_blocks[engine_id] = list(range(num_max_blocks))
        self.remote_callables[engine_id] = remote_callable

    async def start_event_loop(self):
        async def daemon_loop():
            while True:
                await asyncio.sleep(SLEEP_IN_DAEMON_LOOP)
                self.log_block_usage()
                self.clear_zombie_requests()

        await daemon_loop()

    async def _reclaim_blocks(
        self,
        move_events: deque[MoveEvent],
        target_serial: int,
        reclaim_callback: Callable[[MoveEvent], int],
    ) -> Tuple[int, int]:
        num_events = 0
        num_blocks = 0
        while len(move_events) > 0 and move_events[0].serial <= target_serial:
            # XXX: pop the move_event out; this ensures that multiple concurrent
            # invocations of this function does not access the same event more than once
            move_event = move_events.popleft()
            events: Optional[list] = None
            while True:
                if any(req.is_finished for req in move_event.requests):
                    # XXX: request finished -> this event must have finished;
                    # also, the CUDA event may have been freed, so we must NOT rebuild
                    break

                if (handles := move_event.event_handles()) is None:
                    # ..remote not ready
                    await asyncio.wait(
                        [move_event._gather_event_handles()],
                        timeout=SLEEP_WHEN_RECLAIM_PENDS,
                    )
                    continue

                if events is None:
                    # ..get CUDA events
                    events = [
                        rebuild_cuda_event(self.cudart, handle, is_worker=False)
                        for handle in handles
                    ]

                if all(self.cudart.cudaEventQuery(event) for event in events):
                    # ..all events done
                    break
                else:
                    await asyncio.sleep(SLEEP_WHEN_RECLAIM_PENDS)
            if events is not None:
                # ..clean up
                for event in events:
                    self.cudart.cudaEventDestroy(event)

            # ..free to reclaim this move event!
            num_events += 1
            num_blocks += reclaim_callback(move_event)

        return (num_events, num_blocks)

    async def _reclaim_gpu_move_blocks(
        self,
        location: BlockLocation,
    ):
        """Reclaim all moved GPU blocks at the given location.

        This function may asynchronously spin until all move events from the location
        are reclaimed.
        """
        move_events = self.move_events_gpu[location]
        if len(move_events) > 0:
            # ..reclaiming some new events
            lock = asyncio.Event()
            target_serial = move_events[-1].serial
            self.move_gpu_locks[location] = lock
        else:
            lock = self.move_gpu_locks[location]
            if lock is not None:
                # ..some other task is reclaiming already; wait for them
                await lock.wait()
            return

        num_events, num_blocks = await self._reclaim_blocks(
            move_events, target_serial, self._reclaim_gpu_move_blocks_once
        )
        logger.debug(
            f"Reclaimed {num_events} events, {num_blocks} GPU blocks for engine #{location}"
        )
        lock.set()

    async def _reclaim_cpu_move_blocks(
        self,
        block_shape: Tuple[int, ...],
    ):
        """Reclaim all moved CPU blocks with the given shape.

        This function may asynchronously spin until all move events from the location
        are reclaimed.
        """
        num_events = 0
        num_blocks = 0
        move_events = self.move_events_cpu[block_shape]
        if len(move_events) > 0:
            # ..reclaiming some new events
            lock = asyncio.Event()
            target_serial = move_events[-1].serial
            self.move_cpu_locks[block_shape] = lock
        else:
            lock = self.move_cpu_locks[block_shape]
            if lock is not None:
                # ..some other task is reclaiming already; wait for them
                await lock.wait()
            return

        num_events, num_blocks = await self._reclaim_blocks(
            move_events, target_serial, self._reclaim_cpu_move_blocks_once
        )

        logger.debug(
            f"Reclaimed {num_events} events, {num_blocks} CPU blocks for block shape {block_shape}"
        )
        self._reclaim_cpu_slabs()
        lock.set()

    def _reclaim_cpu_slabs(self) -> bool:
        """Reclaim unused cpu slabs, adding them to the free slab list."""
        reclaimed = 0
        for slab, usage in enumerate(self.cpu_slab_usage):
            block_shape = self.cpu_block_shape[slab]
            if usage == 0 and block_shape is not None:
                # Reclaim this slab
                reclaimed += 1
                block_size_bytes = self._get_block_size_bytes(block_shape)

                cpu_free_blocks = self.cpu_free_blocks[block_shape]
                blocks_start, blocks_end = self._get_slab_range(slab, block_size_bytes)
                for _ in range(len(cpu_free_blocks)):
                    block = cpu_free_blocks[0]
                    # ..filters those within the range
                    if blocks_start <= block and block < blocks_end:
                        cpu_free_blocks.popleft()
                    else:
                        cpu_free_blocks.rotate(-1)
                self.cpu_free_slabs.append(slab)
                self.cpu_block_shape[slab] = None
        logger.debug(f"Reclaimed {reclaimed} CPU slabs")
        return reclaimed > 0

    async def _get_free_blocks(
        self,
        num_blocks: int,
        location: BlockLocation,
        block_shape: Tuple[int, ...],
    ) -> List[int]:
        """Get free blocks at the given location.

        If free blocks are not immediately available, reclaim move events, which
        may spin and block the current caller task.
        """
        block_size_bytes = self._get_block_size_bytes(block_shape)
        if is_gpu(location):
            # GPU allocation
            assert location in self.gpu_free_blocks, f"unknown gpu location {location}"

            gpu_free_blocks = self.gpu_free_blocks[location]
            if len(gpu_free_blocks) < num_blocks:
                # Need to "flush" moving GPU blocks, i.e. make sure all
                # move-out operations have finished for the engine, then move their source blocks
                # to the free list.
                await self._reclaim_gpu_move_blocks(location)
                if len(gpu_free_blocks) < num_blocks:
                    raise ResourceWarning(
                        f"not enough free blocks on GPU, requested {num_blocks}, available {len(gpu_free_blocks)}"
                    )

            blocks = gpu_free_blocks[:num_blocks]
            gpu_free_blocks[:] = gpu_free_blocks[num_blocks:]

        else:
            # CPU allocation
            cpu_free_blocks = self.cpu_free_blocks[block_shape]

            # Always try reserving new slabs first; if that fails, reclaim moving blocks.
            def ensure_slab_blocks(
                block_size_bytes: int, num_blocks_needed: int
            ) -> bool:
                # Test if free slabs contains the given number of blocks
                if num_blocks_needed <= 0:
                    return True
                for slab in self.cpu_free_slabs:
                    assert self.cpu_block_shape[slab] is None
                    blocks_start, blocks_end = self._get_slab_range(
                        slab, block_size_bytes
                    )
                    num_blocks_needed -= blocks_end - blocks_start
                    if num_blocks_needed <= 0:
                        return True
                return False

            if not ensure_slab_blocks(
                block_size_bytes, num_blocks - len(cpu_free_blocks)
            ):
                # ..cannot allocate using existing free blocks and slabs
                # Need to "flush" moving CPU blocks, i.e. make sure all
                # moving-in operations have finished, then move their source blocks
                # to the free list
                await self._reclaim_cpu_move_blocks(block_shape)

                # XXX: cpu_free_blocks is potentailly shrinked by _reclaim_cpu_move_blocks
                if not ensure_slab_blocks(
                    block_size_bytes, num_blocks - len(cpu_free_blocks)
                ):
                    raise ResourceWarning(
                        f"not enough free blocks on CPU, requested {num_blocks}, available {len(cpu_free_blocks)} "
                        f"and {len(self.cpu_free_slabs)} slabs"
                    )

            # ..use free slabs if needed
            while len(cpu_free_blocks) < num_blocks:
                slab = self.cpu_free_slabs.pop()
                blocks_start, blocks_end = self._get_slab_range(slab, block_size_bytes)
                self.cpu_block_shape[slab] = block_shape
                cpu_free_blocks.extend(range(blocks_start, blocks_end))

            blocks = []
            for _ in range(num_blocks):
                block = cpu_free_blocks.popleft()
                # ..also update slab usage
                slab = block * block_size_bytes // self.cpu_slab_size_bytes
                self.cpu_slab_usage[slab] += 1
                blocks.append(block)

        return blocks

    async def _get_free_blocks_batched(
        self,
        num_blocks_list: List[int],
        location: BlockLocation,
        block_shape: Tuple[int, ...],
    ) -> List[List[int]]:
        free_blocks = await self._get_free_blocks(
            sum(num_blocks_list), location, block_shape
        )
        cur = 0
        ret = []
        for num in num_blocks_list:
            ret.append(free_blocks[cur : cur + num])
            cur += num
        return ret

    def get_gpu_num_max_blocks(self, engine_id: int) -> int:
        assert engine_id in self.gpu_num_max_blocks, f"unknown gpu location {engine_id}"
        return self.gpu_num_max_blocks[engine_id]

    def get_num_blocks_needed(self, request: Request):
        """Get the number of blocks needed for a request"""
        return (
            request.get_input_len() + request.get_output_len() + self.block_size - 1
        ) // self.block_size

    def get_num_append_blocks_needed(self, request: Request) -> int:
        """Get the number of blocks needed for a request already in GPU"""
        block_table = self.block_tables[request.request_id]
        assert is_gpu(
            block_table.location
        ), f"request {request.request_id} is not on GPU when calling get_num_append_blocks_needed"
        num_blocks_cur = len(block_table.blocks)
        num_blocks_needed = self.get_num_blocks_needed(request)
        return num_blocks_needed - num_blocks_cur

    # async def allocate_blocks(
    #     self,
    #     request: Request,
    #     engine_id: int,
    # ):
    #     """Allocate blocks for a request on GPU for the engine_id."""

    #     # Make sure the request is not already allocated or its blocks are on GPU for the given engine.
    #     assert (
    #         request.request_id not in self.block_tables
    #         or self.block_tables[request.request_id].location == engine_id
    #     ), (f"request {request.request_id} is allocated but is at {self.block_tables[request.request_id].location}; "
    #         f"please migrate it to engine #{engine_id} before allocating more blocks")

    #     num_blocks_needed = self.get_num_blocks_needed(request)
    #     block_shape = self.gpu_block_shape[engine_id]

    #     if request.request_id not in self.block_tables:
    #         # ..this request has not been allocated before
    #         self.block_tables[request.request_id] = BlockTable(
    #             engine_id,
    #             block_shape,
    #             await self._get_free_blocks(
    #                 num_blocks_needed, engine_id, block_shape))
    #     else:
    #         blocks = self.block_tables[request.request_id].blocks
    #         num_blocks_cur = len(blocks)
    #         if num_blocks_cur < num_blocks_needed:
    #             blocks.extend(await self._get_free_blocks(
    #                 num_blocks_needed - num_blocks_cur, engine_id, block_shape))

    async def allocate_blocks_batched(
        self, batched_requests: BatchedRequests, engine_id: int
    ):
        """Allocate blocks for a batch of requests."""
        if len(batched_requests) == 0:
            return

        num_blocks_list = []
        location = engine_id
        block_shape = self.gpu_block_shape[engine_id]

        for request in batched_requests.requests:
            # Make sure the request is not already allocated or its blocks are on GPU for the given engine.
            assert (
                request.request_id not in self.block_tables
                or self.block_tables[request.request_id].location == engine_id
            ), (
                f"request {request.request_id} is allocated but is at {self.block_tables[request.request_id].location}; "
                f"please migrate it to engine #{engine_id} before allocating more blocks"
            )

            num_blocks_needed = self.get_num_blocks_needed(request)
            if request.request_id not in self.block_tables:
                # ..this request has not been allocated before
                num_blocks_list.append(num_blocks_needed)
            else:
                num_blocks_list.append(
                    num_blocks_needed
                    - len(self.block_tables[request.request_id].blocks)
                )

        # Allocate in batch
        free_blocks_list = await self._get_free_blocks_batched(
            num_blocks_list, location, block_shape
        )

        for request, free_blocks in zip(batched_requests.requests, free_blocks_list):
            if request.request_id not in self.block_tables:
                self.block_tables[request.request_id] = BlockTable(
                    engine_id, block_shape, free_blocks
                )
            else:
                blocks = self.block_tables[request.request_id].blocks
                blocks.extend(free_blocks)

    def free_blocks(self, request_id: int):
        """Free blocks for a request."""
        assert request_id in self.block_tables, f"request {request_id} not allocated"
        block_table = self.block_tables.pop(request_id)
        block_size_bytes = self._get_block_size_bytes(block_table.block_shape)

        if is_gpu(block_table.location):
            self.gpu_free_blocks[block_table.location].extend(block_table.blocks)
        else:
            self.cpu_free_blocks[block_table.block_shape].extend(block_table.blocks)
            # Update slab usage
            for block in block_table.blocks:
                slab = block * block_size_bytes // self.cpu_slab_size_bytes
                self.cpu_slab_usage[slab] -= 1
        self.latest_events.pop(request_id, None)
        self.zombie_request_ids.append(request_id)

    def free_blocks_batched(self, requests: List[Request]):
        """Free blocks for a batch of requests"""
        for request in requests:
            self.free_blocks(request.request_id)

    def get_block_table(self, request_id: int) -> BlockTable:
        """Get the block table for a request"""
        assert request_id in self.block_tables, f"request {request_id} not allocated"
        return self.block_tables[request_id]

    def is_all_requests_on_gpu(self, requests: BatchedRequests):
        """Check if all requests in a batch are on GPU"""
        return all(
            is_gpu(self.block_tables[request.request_id].location)
            for request in requests.requests
        )

    def log_block_usage(self):
        log_lines = []
        log_lines.append("---------- Block Usage ----------")
        for engine_id, gpu_free_list in self.gpu_free_blocks.items():
            num_max_blocks = self.get_gpu_num_max_blocks(engine_id)
            num_free_blocks = len(gpu_free_list)
            num_move_blocks = self.move_gpu_num_blocks[engine_id]
            log_lines.append(
                f"[gpu (engine {engine_id})] {num_free_blocks} / {num_max_blocks} "
                f"({num_free_blocks / num_max_blocks * 100:.2f}%) "
                f"free; "
                f"{num_move_blocks} / {num_max_blocks} "
                f"({num_move_blocks / num_max_blocks * 100:.2f}%) "
                f"moving; "
            )

        for block_shape, cpu_free_list in self.cpu_free_blocks.items():
            slabs: List[int] = []
            for i, shape in enumerate(self.cpu_block_shape):
                if shape == block_shape:
                    slabs.append(i)

            num_free_blocks = len(cpu_free_list)
            num_used_blocks = sum(self.cpu_slab_usage[i] for i in slabs)
            num_move_blocks = self.move_cpu_num_blocks[block_shape]

            if num_max_blocks == 0:
                continue
            num_max_blocks = num_used_blocks + num_free_blocks
            log_lines.append(
                f"[cpu (shape {block_shape}, {len(slabs)} slabs)] {num_free_blocks} / {num_max_blocks} "
                f"({num_free_blocks / num_max_blocks * 100:.2f}%) "
                f"free; "
                f"{num_move_blocks} / {num_max_blocks} "
                f"({num_move_blocks / num_max_blocks * 100:.2f}%) "
                f"moving; "
            )
        log_lines.append("---------------------------------")
        logger.info("\n".join(log_lines))

    def clear_zombie_requests(self):
        from aegaeon.llm import Controller

        Controller.remote_call_all_workers_async(
            "clear_request_resource_batched", self.zombie_request_ids
        )
        self.zombie_request_ids.clear()

    async def move_requests(
        self,
        requests: List[Request],
        target_location: BlockLocation,
    ):
        """
        Move blocks for a batch of requests to `target_location`.

        Depending on the current location of each request, this method may:
        - Do nothing, if the request is already at the target location
        - Issue swap_blocks, if the move is between CPU and GPU
        - Issue migrate_blocks, if the move is between two GPUs
        """
        swapped: List[Request] = []
        swap_cur_location: Optional[BlockLocation] = None
        swap_block_shape: Optional[Tuple[int, ...]] = None
        swap_source_block_ids = []
        swap_target_block_ids = []
        for request in requests:
            assert (
                block_table := self.block_tables.get(request.request_id, None)
            ), f"request {request.request_id} not allocated"
            cur_location = block_table.location
            if cur_location == target_location:
                continue
            elif is_gpu(cur_location) and is_gpu(target_location):
                raise NotImplementedError("Migration is not implemented")
            else:
                if swap_cur_location is not None:
                    assert (
                        block_table.location == swap_cur_location
                    ), f"request {request.request_id} is on {block_table.location}; expected {swap_cur_location}"
                if swap_block_shape is not None:
                    assert (
                        block_table.block_shape == swap_block_shape
                    ), f"request {request.request_id}'s shape is not {swap_block_shape}"
                swap_cur_location = block_table.location
                swap_block_shape = block_table.block_shape
                swapped.append(request)
                # The GPU side of a swap is the user; i.e., its worker will later access the blocks
                swap_engine_id = (
                    target_location if is_gpu(target_location) else swap_cur_location
                )

                old_block_ids = block_table.blocks
                num_blocks = len(old_block_ids)
                new_block_ids = await self._get_free_blocks(
                    num_blocks, target_location, swap_block_shape
                )
                swap_source_block_ids += old_block_ids
                swap_target_block_ids += new_block_ids

                # Update the block table
                block_table.blocks = new_block_ids
                block_table.location = target_location

        # Swapping
        if swapped:
            is_swap_in = True if is_gpu(target_location) else False
            source_event_handles = [
                self.latest_events.get(request.request_id, None) for request in swapped
            ]
            event_refs = self.remote_callables[swap_engine_id](
                "swap_blocks",
                [req.request_id for req in swapped],
                swap_source_block_ids,
                swap_target_block_ids,
                source_event_handles,
                is_swap_in,
            )
            # ray.get(event_refs)

            # Update the move states
            move_event = MoveEvent(
                next(self.event_counter),
                swapped,
                swap_block_shape,
                swap_cur_location,
                target_location,
                swap_source_block_ids,
                event_refs,
            )
            if is_gpu(swap_cur_location):
                self.move_gpu_num_blocks[swap_cur_location] += len(
                    swap_source_block_ids
                )
                self.move_events_gpu[swap_cur_location].append(move_event)
            else:
                self.move_cpu_num_blocks[swap_block_shape] += len(swap_source_block_ids)
                self.move_events_cpu[swap_block_shape].append(move_event)
            for request in swapped:
                self.latest_events[request.request_id] = event_refs

        # Migrated
        pass

        return bool(swapped)

    def _migrate_requests(
        self,
        requests: List[Request],
        target_location: BlockLocation,
    ):
        raise NotImplementedError()
        """Migrate blocks for a batch of requests to `target_location`.
        It is assumed that `requests` reside at the same source GPU location.
        """
        if len(requests) == 0:
            return

        _block_table = self.block_tables[requests[0].request_id]
        block_shape = _block_table.block_shape
        cur_location = _block_table.location
        assert is_gpu(target_location) and is_gpu(
            cur_location
        ), f"both target location {target_location} and current location {cur_location} should be on GPU"
        # The target engine is the user; i.e., its worker will later access the blocks
        engine_id = target_location

        source_block_ids = []  # block ids on cur_location
        target_block_ids = []  # block ids on target_location
        for request in requests:
            assert (
                block_table := self.block_tables.get(request.request_id, None)
            ), f"request {request.request_id} not allocated"
            assert (
                block_table.location == cur_location
            ), f"request {request.request_id} is on {target_location}; expected {cur_location}"
            assert (
                block_table.block_shape == block_shape
            ), f"request {request.request_id}'s shape is not {block_shape}"

            old_block_ids = block_table.blocks
            num_blocks = len(old_block_ids)
            new_block_ids = self._get_free_blocks(
                num_blocks, target_location, block_shape, engine_id
            )
            source_block_ids += old_block_ids
            target_block_ids += new_block_ids

            # Update the block table
            block_table.blocks = new_block_ids
            block_table.location = target_location

        # Gather CUDA events
        source_events = [
            self.latest_events.get(request.request_id, None) for request in requests
        ]
        event_refs = self.remote_callables[engine_id](
            "migrate_blocks",
            requests,
            cur_location,
            source_block_ids,
            target_block_ids,
            source_events,
        )
        # ..this adds the migration event to the source engine as a move-out event
        self.remote_callables[cur_location](
            "migrate_blocks_source", requests, event_ref
        )

        # Update the move states
        move_event = MoveEvent(
            block_shape,
            cur_location,
            target_location,
            old_block_ids,
            event_ref,
        )
        self.move_gpu_num_blocks[cur_location] += num_blocks
        self.move_events_gpu[cur_location].add(move_event)
        for request in requests:
            self.latest_events[request.request_id] = event_ref

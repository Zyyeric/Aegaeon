from . import DeviceType
from .util import pcie
from .worker import Worker
from .scheduler import *
from .request import Request
from .workload import gen_workload
from multiprocessing import Process, Queue

import simpy
import argparse
import logging
import numpy as np
import time
import os
import copy
from matplotlib import pyplot as plt

TTFT_SLO = 10000
TPOT_SLO = 100
DIR = os.path.dirname(__file__)

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # -- general
    parser.add_argument('--name', type=str, default=None, help="Name of experiment")
    parser.add_argument('--max-batch-size', type=int, default=10 ** 7,
                        help="Maximum batch size")
    parser.add_argument('--max-tokens', type=int, default=10 ** 7,
                        help="Maximum number of token in a batch")
    parser.add_argument('--num-models', type=int, nargs="+")
    parser.add_argument('--arrival-rate', type=float, nargs="+")
    parser.add_argument('--inlen-scale', type=float, default=1.0)
    parser.add_argument('--outlen-scale', type=float, default=1.0)
    # -- workload
    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    # -- devices
    parser.add_argument('--devices', nargs='+', type=str, default=['h100'] * 16,
                        help="List of GPU specifications. For example, `--devices a100 a100` "
                        "specifies two NVIDIA A100 gpus for simulation. "
                        "Checkout `DeviceType` for valid GPU types. ")
    parser.add_argument('--h2d', type=pcie, default='gen4x16',
                        help="Host to device PCIe specification. "
                        "Either use the `genGxN` format, or specify the bandwidth "
                        "in GB/s directly.")
    parser.add_argument('--swap-overhead', type=int, default=0,
                        help="Extra model swapping overhead in ms")

    args = parser.parse_args(args=args)

    return args

def main(
    args,
    policy: Policy,
    silent: bool = False,
    ret = None,
) -> list[Request]:
    args.devices = [DeviceType(device) for device in args.devices]

    if not silent:
        print('====================')
        print(args)
        print(policy)

    env = simpy.Environment()

    # Prepare simulation workload
    if args.workload is not None:
        workload = gen_workload(env, file=args.workload, seed=args.seed)
    else:
        workload = gen_workload(env, seed=args.seed)
    if not silent:
        print('size of workload:', len(workload))

    # Prepare the intra-node workers and scheduler
    scheduler = Scheduler(env=env, policy=policy,
                          workers=None, h2d=args.h2d)
    if isinstance(policy, MixServePolicy):
        assert policy.num_prefill_workers < len(args.devices)
        workers = []
        for wid, device in enumerate(args.devices):
            if wid < policy.num_prefill_workers:
                worker = PrefillWorker(scheduler, env=env, worker_id=wid, device=device, swap_overhead_ms=args.swap_overhead)
            else:
                worker = DecodeWorker(
                    scheduler, 
                    env=env, worker_id=wid, device=device, 
                    swap_overhead_ms=args.swap_overhead,
                    tpot_slo_ms=TPOT_SLO)
            workers.append(worker)
    else:
        workers = [Worker(env=env, worker_id=wid, device=device, swap_overhead_ms=args.swap_overhead)
                   for wid, device in enumerate(args.devices)]
    scheduler.workers = workers

    # Register processes
    for worker in workers:
        env.process(worker.run())
    scheduler.put_workload(workload)
    
    # Start the simulation
    try:
        start = time.perf_counter()
        env.run()

        end = time.perf_counter()

        if not silent:
            print(f"Simulation successful for {args}\n{policy}\nelapsed = {end-start}s")
        
            if isinstance(policy, MixServePolicy):
                for worker in workers:
                    if isinstance(worker, PrefillWorker):
                        print(f'Worker #{worker.worker_id} max group size: {worker.max_group_sz}; dist = {np.histogram(worker.group_sz, bins=range(1,worker.max_group_sz+2))}')

        if ret is not None:
            ret.put([req for _, req in workload])
        else:
            return [req for _, req in workload]

    except Exception:
        print("Simulation failed")
        # Any exception triggers an information dump
        print(args.name)
        print(policy)
        for worker in workers:
            worker.dump_state()
        raise


def plot_slo_models(args):
    POLICIES = [
        SeLLMPolicy(),
        SeLLMPlusPolicy(),
    ]
    NMODELS = args.num_models
    RATES = args.arrival_rate
    inlen_scale = args.inlen_scale
    outlen_scale = args.outlen_scale

    ttft_by_x = {}
    tpot_by_x = {}
    reqs_by_x = {}
    
    # Collect metrics
    processes = []
    for policy in POLICIES:
        for m in NMODELS:
            for rate in RATES:
                args.workload = f'synth-{m}-{rate}-{inlen_scale}-{outlen_scale}.csv'

                q = Queue()
                p = Process(target=main, args=(args, policy), kwargs={'silent': True, 'ret': q})
                p.start()

                x = m if len(NMODELS) > 1 else rate
                processes.append(((x, policy), (p, q)))
        
    if args.name is not None: print(args.name)
    print(f"Started {len(processes)} jobs...")
    start = time.perf_counter()
    for i, ((x, policy), (p, q)) in enumerate(processes):
        reqs = q.get(block=True)
        p.join() # Caveat: join must be done after q.get() because the join will hang otherwise
        print(f'({i}/{len(processes)}) {x} {policy} done after {time.perf_counter()-start:.2f} seconds')

        ttft_by_x[(x, policy)] = []
        tpot_by_x[(x, policy)] = []
        reqs_by_x[(x, policy)] = reqs
        
        for req in reqs:
            if not req.is_finished:
                if not req.is_failed:
                    logger.warning(f'{req} is neither finished nor failed; the scheduler must have missed it')
                ttft_by_x[(x, policy)].append(10 ** 7) 
                tpot_by_x[(x, policy)].append(10 ** 7) 
                continue
            ttft_by_x[(x, policy)].append(req.time_first_token_ms - req.time_arrival_ms)
            tpot_by_x[(x, policy)].append(
                int((req.time_last_token_ms - req.time_first_token_ms) / (req.output_len - 1)) \
                if req.output_len > 1 \
                else 0
            )
    
    json_name = args.name if args.name else "slo-models"
    json_path = os.path.join(DIR, '../plots/json', f'{json_name}.json')
    with open(json_path, 'w') as f:
        metrics = {}
        for k, reqs in reqs_by_x.items():
            m = {}
            for req in reqs:
                if req.is_failed:
                    m[req.req_id] = [-1]
                    continue
                per_token_ms = [req.time_first_token_ms - req.time_arrival_ms]
                for end, start in zip(req.time_each_token_ms[1:], req.time_each_token_ms[:-1]):
                    per_token_ms.append(end-start)
                m[req.req_id] = per_token_ms
            metrics[str(k)] = m  
            
        json.dump({str(k): v for k, v in metrics.items()}, f)



if __name__ == '__main__':
    args = parse_args()

    # Set logging level
    logging.basicConfig(level=logging.WARNING)

    model_spec = f'{args.num_models[0]}' if len(args.num_models) == 1 else f'{args.num_models[0]}-{args.num_models[-1]}'
    rate_spec = f'{args.arrival_rate[0]}' if len(args.arrival_rate) == 1 else f'{args.arrival_rate[0]}-{args.arrival_rate[-1]}'
    args.name = f"sim-model{model_spec}-rate{rate_spec}-scale-{args.inlen_scale}-{args.outlen_scale}"
    plot_slo_models(args)

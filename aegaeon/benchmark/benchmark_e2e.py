import os
import argparse
import ray
import psutil
from aegaeon import LLMService, NodeConfig
from aegaeon.models import ModelType

DIR = os.path.dirname(__file__)

def test(args):
    ray.init(address='auto')

    model_list = [
        ModelType.llama2_13b_chat,
        ModelType.yi1_5_9b_chat,
        ModelType.qwen2_5_7b,
    ]
    service = LLMService(
        [
            NodeConfig(
                node_id=f"node_{i}",
                num_prefill_engines=3,
                num_decode_engines=5,
                model_cache_size=64 * (1024**3),
                cpu_num_slabs=args.cpu_num_slabs,
                cached_models=model_list,
            ) for i in range(args.nnodes)
        ]
        
    )

    from aegaeon.utils import TTFT_SLO, TPOT_SLO
    result_file = f'run-{args.num_models}-{args.arrival_rate}-{args.inlen_scale}-{args.outlen_scale}-ttft{TTFT_SLO}-tpot{TPOT_SLO}.json'
    service.replay(
        os.path.join(DIR, '..', 'plots', 'json', result_file),
        args.num_models,
        args.arrival_rate,
        inlen_scale=args.inlen_scale,
        outlen_scale=args.outlen_scale,
        duration=60,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--num-models", type=int, default=40)
    parser.add_argument("--arrival-rate", type=float, default=0.1)
    parser.add_argument("--inlen-scale", type=float, default=1.0)
    parser.add_argument("--outlen-scale", type=float, default=1.0)
    parser.add_argument("--cpu-num-slabs", type=int, default=256)
    args = parser.parse_args()

    test(args)

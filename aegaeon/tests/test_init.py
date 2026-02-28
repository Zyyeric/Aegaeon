import psutil
import ray
import os
from aegaeon import LLMService, NodeConfig


def test():
    ray.init(num_cpus=psutil.cpu_count(), resources={"node_0": 1})
    service = LLMService(
        [
            NodeConfig(
                node_id="node_0",
                num_prefill_engines=1,
                num_decode_engines=1,
                model_cache_size=16 * (1024**3),
                cpu_num_slabs=32,
            ),
        ]
    )
    print("[DONE]")


if __name__ == "__main__":
    test()

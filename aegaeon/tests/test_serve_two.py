import ray
import time
import psutil
from aegaeon import LLMService, NodeConfig, Request
from aegaeon.models import ModelType


def test():
    ray.init(num_cpus=psutil.cpu_count(), resources={"node_0": 1})
    service = LLMService(
        [
            NodeConfig(
                node_id="node_0",
                num_prefill_engines=1,
                num_decode_engines=1,
                model_cache_size=32 * (1024**3),
                cpu_num_slabs=4,
                cached_models=[ModelType.yi1_5_6b_chat, ModelType.llama2_7b],
            ),
        ]
    )

    reqs = [
        Request(
            ModelType.yi1_5_6b_chat,
            time.time(),
            0,
            [1] * 6,
            256,
            prompt="Introduce the capital of France.",
        ),
        Request(
            ModelType.yi1_5_6b_chat,
            time.time(),
            1,
            [1] * 6,
            128,
            prompt="Introduce the capital of France.",
        ),
        Request(
            ModelType.llama2_7b,
            time.time(),
            2,
            [1] * 6,
            512,
            prompt="Introduce the capital of France.",
        ),
        Request(
            ModelType.llama2_7b,
            time.time(),
            3,
            [1] * 6,
            8,
            prompt="Introduce the capital of France.",
        ),
    ]
    outputs = service.serve(reqs)
    for i, output in enumerate(outputs):
        if isinstance(output, list):
            print(
                f"(req {i}) {[step.new_token for step in output][:min(16, len(output))]}.."
            )
        else:
            print(f"(req {i}) {output}")


if __name__ == "__main__":
    test()

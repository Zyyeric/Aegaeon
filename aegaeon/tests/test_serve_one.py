import psutil
import ray
import time
from aegaeon import LLMService, NodeConfig, Request
from aegaeon.models import ModelType


def test():
    ray.init(num_cpus=psutil.cpu_count(), resources={"node_0": 1})

    # model = ModelType.qwen1_5_moe_a2_7b_chat
    # model = ModelType.llava1_5_13b
    # model = ModelType.qwen_14b_chat
    # model = ModelType.llama2_13b_chat
    # model = ModelType.yi1_5_6b_chat
    # model = ModelType.qwen_7b_chat
    model = ModelType.llama2_7b
    # model = ModelType.qwen1_5_moe_a2_7b_chat
    service = LLMService(
        [
            NodeConfig(
                node_id="node_0",
                num_prefill_engines=1,
                num_decode_engines=1,
                model_cache_size=16 * (1024**3),
                cpu_num_slabs=32,
                cached_models=[model],
            ),
        ]
    )

    reqs = [
        Request(
            model,
            time.time(),
            0,
            [1] * 6,
            256,
            prompt="Introduce the capital of France.",
        ),
        Request(
            model,
            time.time(),
            1,
            [1] * 6,
            128,
            prompt="Introduce the capital of France.",
        ),
        Request(
            model,
            time.time(),
            2,
            [1] * 6,
            512,
            prompt="Introduce the capital of France.",
        ),
        Request(
            model, time.time(), 3, [1] * 6, 8, prompt="Introduce the capital of France."
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

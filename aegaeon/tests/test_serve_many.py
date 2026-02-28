import ray
import time
import psutil
from aegaeon import LLMService, NodeConfig, Request
from aegaeon.models import ModelType
from aegaeon.utils import compute_request_metrics, compute_request_latencies


def test():
    ray.init(num_cpus=psutil.cpu_count(), resources={"node_0": 1})
    model_list = [
        ModelType.qwen_7b, ModelType.qwen_7b_chat,
        ModelType.llama2_7b, ModelType.internlm2_5_7b_chat,
        ModelType.yi1_5_9b_chat, ModelType.qwen2_5_7b,
        ModelType.llama2_13b_chat, ModelType.qwen1_5_moe_a2_7b_chat,
        # ModelType.synth_model_06, ModelType.synth_model_14,
        # ModelType.synth_model_22, ModelType.synth_model_30,
        # ModelType.synth_model_07, ModelType.synth_model_15,
        # ModelType.synth_model_23, ModelType.synth_model_31,
        # ModelType.yi1_5_6b_chat,
        # ModelType.qwen_7b_chat,
        # ModelType.llama2_7b,
    ]
    service = LLMService(
        [
            NodeConfig(
                node_id="node_0",
                num_prefill_engines=2,
                num_decode_engines=2,
                model_cache_size=160 * (1024**3),
                cpu_num_slabs=32,
                cached_models=model_list,
            ),
        ]
    )

    reqs = [
        Request(
            model_list[i // 2],
            time.time(),
            i,
            [1] * 6,
            128,
            prompt="Introduce the capital of France.",
        )
        for i in range(2 * len(model_list))
    ]

    outputs = service.serve(reqs)
    for i, output in enumerate(outputs):
        if isinstance(output, list) and len(output) == reqs[i].decode_tokens:
            ttft, qos, per_token = compute_request_metrics(reqs[i], output)
            latencies = compute_request_latencies(reqs[i], output)
            print(
                "(req {}) {}..(ttft={:.2f}, qos={:.2f}, lat={})".format(
                    i,
                    [step.new_token for step in output][: min(16, len(output))],
                    ttft,
                    qos,
                    latencies,
                )
            )
        else:
            print(f"(req {i}) failed")


if __name__ == "__main__":
    test()

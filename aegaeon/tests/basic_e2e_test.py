import os
import time

from aegaeon import LLMService, NodeConfig
from aegaeon.models import ModelType
from aegaeon.request import Request

HF_REPO_ID = "Qwen/Qwen2-1.5B"


def _ensure_model_downloaded(local_model_dir: str) -> None:
    if os.path.isdir(local_model_dir) and os.listdir(local_model_dir):
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        raise RuntimeError(
            "huggingface_hub is required for auto-download. "
            "Install with: pip install -U huggingface_hub"
        ) from e

    os.makedirs(local_model_dir, exist_ok=True)
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=local_model_dir,
        local_dir_use_symlinks=False,
    )


def main() -> None:
    model_dir = os.environ.get(
        "AEGAEON_QWEN2_1_5B_PATH",
        ModelType.qwen2_1_5b.path(),
    )
    _ensure_model_downloaded(model_dir)
    os.environ["AEGAEON_QWEN2_1_5B_PATH"] = model_dir

    svc = LLMService(
        [
            NodeConfig(
                node_id="node_0",
                num_prefill_engines=1,
                num_decode_engines=1,
                model_cache_size=16 * (1024**3),
                cpu_num_slabs=16,
                cached_models=[ModelType.qwen2_1_5b],
            )
        ]
    )

    req = Request(
        model=ModelType.qwen2_1_5b,
        arrival_time=time.time(),
        request_id=0,
        prompt_token_ids=[1] * 16,
        decode_tokens=8,
    )

    out = svc.serve([req])[0]
    print("step outputs:", len(out))
    print(
        "generated token ids:",
        [x.new_token_id for x in out if x.new_token_id is not None],
    )

    svc.reset()


if __name__ == "__main__":
    main()
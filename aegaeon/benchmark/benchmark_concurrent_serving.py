from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

import ray

from aegaeon import LLMService, NodeConfig
from aegaeon.models import ModelType
from aegaeon.request import Request
from aegaeon.utils import TTFT_SLO, compute_request_latencies, compute_request_metrics


def _parse_models_csv(raw: str) -> list[ModelType]:
    models: list[ModelType] = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        model = ModelType.from_str(name)
        if model is None:
            raise SystemExit(
                f"Unsupported model '{name}'. Use names accepted by ModelType.from_str, "
                "for example: qwen2_5_7b, yi1_5_9b_chat, llama2_13b_chat, qwen2_1_5b."
            )
        models.append(model)
    if not models:
        raise SystemExit("At least one valid model is required")
    return models


def _load_trace(trace_json: Path) -> list[dict[str, Any]]:
    try:
        raw = json.loads(trace_json.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"Trace file not found: {trace_json}") from exc

    if not isinstance(raw, list):
        raise SystemExit("Trace JSON must be a list of request objects")
    if not raw:
        raise SystemExit("Trace JSON is empty")
    return raw


def _validate_and_build_requests(
    trace_items: list[dict[str, Any]],
) -> tuple[list[Request], dict[int, dict[str, Any]], list[str]]:
    requests: list[Request] = []
    trace_meta: dict[int, dict[str, Any]] = {}
    models_seen: list[str] = []

    for i, item in enumerate(trace_items):
        if not isinstance(item, dict):
            raise SystemExit(f"Trace item at index {i} must be an object")

        trace_request_id = str(item.get("request_id", f"trace-{i}"))
        model_name = item.get("model")
        prompt_token_ids = item.get("prompt_token_ids")
        max_new_tokens = item.get("max_new_tokens")
        arrival_offset_ms = item.get("arrival_offset_ms")

        if not isinstance(model_name, str) or not model_name:
            raise SystemExit(f"Trace item {trace_request_id} has invalid 'model'")
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids:
            raise SystemExit(
                f"Trace item {trace_request_id} must contain a non-empty 'prompt_token_ids' list"
            )
        if any(not isinstance(tok, int) for tok in prompt_token_ids):
            raise SystemExit(f"Trace item {trace_request_id} has non-integer prompt tokens")
        if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
            raise SystemExit(f"Trace item {trace_request_id} has invalid 'max_new_tokens'")
        if not isinstance(arrival_offset_ms, (int, float)) or arrival_offset_ms < 0:
            raise SystemExit(
                f"Trace item {trace_request_id} has invalid 'arrival_offset_ms'"
            )

        model = ModelType.from_str(model_name)
        if model is None:
            raise SystemExit(
                f"Trace item {trace_request_id} uses unsupported model '{model_name}'. "
                "Generate traces with Aegaeon model names such as qwen2_5_7b or llama2_13b_chat."
            )

        request_id = len(requests)
        requests.append(
            Request(
                model=model,
                arrival_time=float(arrival_offset_ms) / 1000.0,
                request_id=request_id,
                prompt_token_ids=[int(tok) for tok in prompt_token_ids],
                decode_tokens=max_new_tokens,
            )
        )
        trace_meta[request_id] = {
            "trace_request_id": trace_request_id,
            "model": model_name,
            "prompt_length": len(prompt_token_ids),
            "max_new_tokens": max_new_tokens,
            "arrival_offset_ms": float(arrival_offset_ms),
        }
        if model_name not in models_seen:
            models_seen.append(model_name)

    return requests, trace_meta, models_seen


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"avg": None, "p50": None, "p90": None, "p99": None}
    s = sorted(values)
    n = len(s)
    return {
        "avg": float(statistics.fmean(s)),
        "p50": float(s[int(0.50 * (n - 1))]),
        "p90": float(s[int(0.90 * (n - 1))]),
        "p99": float(s[int(0.99 * (n - 1))]),
    }


def _make_service(args: argparse.Namespace, cached_models: list[ModelType]) -> LLMService:
    cluster = [
        NodeConfig(
            node_id=f"node_{i}",
            num_prefill_engines=args.num_prefill_engines,
            num_decode_engines=args.num_decode_engines,
            cpu_num_slabs=args.cpu_num_slabs,
            model_cache_size=int(args.model_cache_size_gb * (1024**3)),
            cached_models=cached_models,
        )
        for i in range(args.nnodes)
    ]
    return LLMService(cluster)


async def _serve_trace(
    service: LLMService,
    requests: list[Request],
) -> list[Any]:
    start_time = time.time()

    async def serve_one(req: Request) -> Any:
        await asyncio.sleep(max(0.0, req.arrival_time - (time.time() - start_time)))
        req.arrival_time = time.time()
        node_index = req.model.value % len(service.nodes)
        return await service.nodes[node_index].serve.remote(req)

    return await asyncio.gather(*(serve_one(req) for req in requests), return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a JSON request trace against Aegaeon for concurrent model serving."
    )
    parser.add_argument("--trace-json", required=True, help="Trace JSON from benchmark.trace_generator")
    parser.add_argument("--out-json", required=True, help="Where to save benchmark results")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--num-prefill-engines", type=int, default=3)
    parser.add_argument("--num-decode-engines", type=int, default=5)
    parser.add_argument("--cpu-num-slabs", type=int, default=256)
    parser.add_argument("--model-cache-size-gb", type=float, default=64.0)
    parser.add_argument(
        "--cached-models",
        default="",
        help="Optional comma-separated model list to pre-cache. Defaults to the unique models found in the trace.",
    )
    args = parser.parse_args()

    if args.nnodes <= 0:
        raise SystemExit("--nnodes must be > 0")
    if args.num_prefill_engines < 0 or args.num_decode_engines < 0:
        raise SystemExit("--num-prefill-engines and --num-decode-engines must be >= 0")
    if args.model_cache_size_gb <= 0:
        raise SystemExit("--model-cache-size-gb must be > 0")

    trace_json = Path(args.trace_json)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    trace_items = _load_trace(trace_json)
    requests, trace_meta, trace_models = _validate_and_build_requests(trace_items)
    cached_models = (
        _parse_models_csv(args.cached_models)
        if args.cached_models
        else [ModelType.from_str(name) for name in trace_models]
    )
    cached_models = [model for model in cached_models if model is not None]

    ray.init(address="auto")
    service = _make_service(args, cached_models)

    wall_start = time.time()
    outputs = asyncio.run(_serve_trace(service, requests))
    wall_end = time.time()

    per_request: list[dict[str, Any]] = []
    metrics_by_trace_request_id: dict[str, dict[str, Any]] = {}
    ttfts: list[float] = []
    tpots: list[float] = []
    qos_values: list[float] = []
    num_slo_violations = 0

    for req, output in zip(requests, outputs):
        meta = trace_meta[req.request_id]
        if isinstance(output, list) and len(output) == req.decode_tokens:
            ttft, qos, per_token = compute_request_metrics(req, output)
            latencies = compute_request_latencies(req, output)
            tpot_values = per_token[1:] if len(per_token) > 1 else []
            avg_tpot = float(statistics.fmean(tpot_values)) if tpot_values else None
            record = {
                "trace_request_id": meta["trace_request_id"],
                "model": meta["model"],
                "prompt_length": meta["prompt_length"],
                "max_new_tokens": meta["max_new_tokens"],
                "arrival_offset_ms": meta["arrival_offset_ms"],
                "ttft": ttft,
                "qos": qos,
                "per_token": per_token,
                "tpot_avg": avg_tpot,
                **latencies,
            }
            ttfts.append(float(ttft))
            qos_values.append(float(qos))
            if avg_tpot is not None:
                tpots.append(avg_tpot)
            if ttft > TTFT_SLO:
                num_slo_violations += 1
        else:
            record = {
                "trace_request_id": meta["trace_request_id"],
                "model": meta["model"],
                "prompt_length": meta["prompt_length"],
                "max_new_tokens": meta["max_new_tokens"],
                "arrival_offset_ms": meta["arrival_offset_ms"],
                "ttft": TTFT_SLO,
                "qos": 0.0,
                "per_token": [],
                "tpot_avg": None,
                "error": str(output),
            }
            num_slo_violations += 1

        per_request.append(record)
        metrics_by_trace_request_id[meta["trace_request_id"]] = record

    result = {
        "service": "aegaeon-concurrent-serving",
        "trace_json": str(trace_json),
        "out_json": str(out_json),
        "models": trace_models,
        "num_requests": len(requests),
        "cluster": {
            "nnodes": args.nnodes,
            "num_prefill_engines": args.num_prefill_engines,
            "num_decode_engines": args.num_decode_engines,
            "cpu_num_slabs": args.cpu_num_slabs,
            "model_cache_size_gb": args.model_cache_size_gb,
            "cached_models": [model.name for model in cached_models],
        },
        "summary": {
            "ttft_s": _summary(ttfts),
            "tpot_s": _summary(tpots),
            "qos": _summary(qos_values),
            "slo_violations": num_slo_violations,
            "successes": sum(1 for item in per_request if "error" not in item),
            "failures": sum(1 for item in per_request if "error" in item),
            "wall_time_s": wall_end - wall_start,
        },
        "metrics": metrics_by_trace_request_id,
        "per_request": per_request,
    }

    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"out_json": str(out_json), "num_requests": len(requests)}, indent=2))


if __name__ == "__main__":
    main()

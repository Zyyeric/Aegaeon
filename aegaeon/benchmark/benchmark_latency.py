"""Benchmark the latency of processing a single batch of requests."""
import argparse
import json
from typing import List

import numpy as np
from tqdm import tqdm

from vllm import LLM, SamplingParams

def main(args: argparse.Namespace):
    print(args)

    llm_args = {
        'model': args.model,
        'trust_remote_code': True,
        'tensor_parallel_size': args.tensor_parallel_size,
        'dtype': args.dtype,
        'kv_cache_dtype': args.kv_cache_dtype,
        'enforce_eager': args.enforce_eager,
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }

    try:
        llm = LLM(max_model_len=args.max_model_len, **llm_args)
    except ValueError:
        llm = LLM(**llm_args)

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    dummy_prompt_token_ids = np.random.randint(10000,
                                               size=(args.batch_size,
                                                     args.input_len))
    dummy_inputs: List = [{
        "prompt_token_ids": batch
    } for batch in dummy_prompt_token_ids.tolist()]

    def run_to_completion():
        prefilled = False
        prefill_latency = None
        decode_latency = None
        decode_iter = 0

        for dummy_input in dummy_inputs:
            llm._add_request(dummy_input, sampling_params)
        while True:
            request_outputs = llm.llm_engine.step()
            request_output = request_outputs[0]
            if request_output.metrics.first_token_time is not None:
                if prefilled:
                    decode_iter += 1
                    if request_output.finished:
                        decode_latency = (request_output.metrics.finished_time - request_output.metrics.first_token_time) / decode_iter
                        break
                    else:
                        pass
                else:
                    assert all(ro.metrics.first_token_time is not None for ro in request_outputs), \
                        "Prefill requests are queueing"
                    
                    prefilled = True
                    prefill_latency = request_output.metrics.first_token_time - request_output.metrics.first_scheduled_time
            else:
                pass

        return (prefill_latency, decode_latency)

    print("Warming up...")
    for _ in tqdm(range(args.num_iters_warmup), desc="Warmup iterations"):
        run_to_completion()

    # Benchmark.
    prefill_latencies = []
    decode_latencies = []
    for _ in tqdm(range(args.num_iters), desc="Profiling iterations"):
        prefill_lat, decode_lat = run_to_completion()
        prefill_latencies.append(prefill_lat)
        decode_latencies.append(decode_lat)
    prefill_latencies = np.array(prefill_latencies)
    decode_latencies = np.array(decode_latencies)

    percentages = [10, 25, 50, 75, 90]
    prefill_percentiles = np.percentile(prefill_latencies, percentages)
    decode_percentiles = np.percentile(decode_latencies, percentages)

    print(f'Avg prefill latency: {np.mean(prefill_latencies)} seconds')
    for percentage, percentile in zip(percentages, prefill_percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')
    
    print(f'Avg decode latency: {np.mean(decode_latencies)} seconds')
    for percentage, percentile in zip(percentages, decode_percentiles):
        print(f'{percentage}% percentile latency: {percentile} seconds')

    # Output JSON results if specified
    if args.output_json:
        results = {
            "avg_prefill_latency": np.mean(prefill_latencies),
            "prefill_latencies": prefill_latencies.tolist(),
            "prefill_percentiles": dict(zip(percentages, prefill_percentiles.tolist())),
            "avg_decode_latency": np.mean(decode_latencies),
            "decode_latencies": decode_latencies.tolist(),
            "decode_percentiles": dict(zip(percentages, decode_percentiles.tolist())),
        }
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Benchmark the latency of processing a single batch of '
        'requests till completion.')
    parser.add_argument('--model', type=str, default='facebook/opt-125m')
    parser.add_argument('--max-model-len', type=int, default=4096)
    parser.add_argument('--enforce-eager',
                        action='store_true',
                        help='enforce eager mode and disable CUDA graph')
    parser.add_argument('--tensor-parallel-size', '-tp', type=int, default=1)
    parser.add_argument('--input-len', type=int, default=32)
    parser.add_argument('--output-len', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-iters-warmup',
                        type=int,
                        default=4,
                        help='Number of iterations to run for warmup.')
    parser.add_argument('--num-iters',
                        type=int,
                        default=12,
                        help='Number of iterations to run.')
    parser.add_argument(
        '--dtype',
        type=str,
        default='auto',
        choices=['auto', 'half', 'float16', 'bfloat16', 'float', 'float32'],
        help='data type for model weights and activations. '
        'The "auto" option will use FP16 precision '
        'for FP32 and FP16 models, and BF16 precision '
        'for BF16 models.')
    parser.add_argument(
        '--kv-cache-dtype',
        type=str,
        choices=['auto', 'fp8', 'fp8_e5m2', 'fp8_e4m3'],
        default="auto",
        help='Data type for kv cache storage. If "auto", will use model '
        'data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. '
        'ROCm (AMD GPU) supports fp8 (=fp8_e4m3)')
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save the latency results in JSON format.')
    parser.add_argument('--gpu-memory-utilization',
                        type=float,
                        default=0.98,
                        help='the fraction of GPU memory to be used for '
                        'the model executor, which can range from 0 to 1.'
                        'If unspecified, will use the default value of 0.9.')
    args = parser.parse_args()
    main(args)

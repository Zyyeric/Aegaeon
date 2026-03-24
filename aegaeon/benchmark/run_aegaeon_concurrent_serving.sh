#!/usr/bin/env bash
set -euo pipefail

# Run one part at a time:
#   trace : generate a mixed-model JSON trace ingestable by Aegaeon
#   run   : replay the trace against Aegaeon concurrent model serving
#   all   : generate the trace, then run the benchmark

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PART=""
MODELS="llama2_7b_chat,llama2_13b_chat"
NUM_REQUESTS=64
PROMPT_TOKENS=512
MAX_NEW_TOKENS=64
ARRIVAL_RATE_RPS=2.0
SEED=0

NNODES=1
NUM_PREFILL_ENGINES=3
NUM_DECODE_ENGINES=5
CPU_NUM_SLABS=256
MODEL_CACHE_SIZE_GB=64
CACHED_MODELS=""

RESULTS_DIR="benchmark/results/aegaeon_concurrent_serving"
TRACE_JSON=""
OUT_JSON=""

usage() {
  cat <<EOF
Usage:
  bash benchmark/run_aegaeon_concurrent_serving.sh --part <trace|run|all> [options]

Core options:
  --part <name>                    Which part to run
  --models "<csv>"                 Aegaeon model names (default: llama2_7b_chat,llama2_13b_chat)
  --results-dir <path>             Output directory (default: benchmark/results/aegaeon_concurrent_serving)
  --trace-json <path>              Trace path (default: <results-dir>/request_trace.json)
  --out-json <path>                Benchmark output path (default: <results-dir>/aegaeon_concurrent_serving.json)

Trace generation:
  --num-requests <int>             Number of requests in the mixed-model trace (default: 64)
  --prompt-tokens <int>            Prompt length in tokens (default: 512)
  --max-new-tokens <int>           Generated tokens per request (default: 64)
  --arrival-rate-rps <float>       Request arrival rate (default: 2.0)
  --seed <int>                     Random seed (default: 0)

Aegaeon replay:
  --nnodes <int>                   Number of Aegaeon nodes (default: 1)
  --num-prefill-engines <int>      Prefill engines per node (default: 3)
  --num-decode-engines <int>       Decode engines per node (default: 5)
  --cpu-num-slabs <int>            CPU cache slabs per node (default: 256)
  --model-cache-size-gb <float>    Model cache size per node in GB (default: 64)
  --cached-models "<csv>"          Optional explicit pre-cached model list (default: infer from trace)

Examples:
  bash benchmark/run_aegaeon_concurrent_serving.sh --part trace
  bash benchmark/run_aegaeon_concurrent_serving.sh --part run
  bash benchmark/run_aegaeon_concurrent_serving.sh --part all --arrival-rate-rps 4.0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --part) PART="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --num-requests) NUM_REQUESTS="$2"; shift 2 ;;
    --prompt-tokens) PROMPT_TOKENS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --arrival-rate-rps) ARRIVAL_RATE_RPS="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --nnodes) NNODES="$2"; shift 2 ;;
    --num-prefill-engines) NUM_PREFILL_ENGINES="$2"; shift 2 ;;
    --num-decode-engines) NUM_DECODE_ENGINES="$2"; shift 2 ;;
    --cpu-num-slabs) CPU_NUM_SLABS="$2"; shift 2 ;;
    --model-cache-size-gb) MODEL_CACHE_SIZE_GB="$2"; shift 2 ;;
    --cached-models) CACHED_MODELS="$2"; shift 2 ;;
    --results-dir) RESULTS_DIR="$2"; shift 2 ;;
    --trace-json) TRACE_JSON="$2"; shift 2 ;;
    --out-json) OUT_JSON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${PART}" ]]; then
  echo "--part is required"
  usage
  exit 1
fi

mkdir -p "${RESULTS_DIR}"

if [[ -z "${TRACE_JSON}" ]]; then
  TRACE_JSON="${RESULTS_DIR}/request_trace.json"
fi
if [[ -z "${OUT_JSON}" ]]; then
  OUT_JSON="${RESULTS_DIR}/aegaeon_concurrent_serving.json"
fi

run_trace() {
  echo "[TRACE] Generating mixed-model trace at ${TRACE_JSON}"
  python -m benchmark.trace_generator \
    --models "${MODELS}" \
    --model-mix-policy round_robin \
    --num-requests "${NUM_REQUESTS}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --arrival-rate-rps "${ARRIVAL_RATE_RPS}" \
    --seed "${SEED}" \
    --out-json "${TRACE_JSON}"
}

run_aegaeon() {
  if [[ ! -f "${TRACE_JSON}" ]]; then
    echo "Missing trace: ${TRACE_JSON}"
    echo "Run --part trace first or pass --trace-json"
    exit 1
  fi

  local cached_args=()
  if [[ -n "${CACHED_MODELS}" ]]; then
    cached_args=(--cached-models "${CACHED_MODELS}")
  fi

  echo "[AEGAEON] Replaying ${TRACE_JSON}"
  python -m benchmark.benchmark_concurrent_serving \
    --trace-json "${TRACE_JSON}" \
    --out-json "${OUT_JSON}" \
    --nnodes "${NNODES}" \
    --num-prefill-engines "${NUM_PREFILL_ENGINES}" \
    --num-decode-engines "${NUM_DECODE_ENGINES}" \
    --cpu-num-slabs "${CPU_NUM_SLABS}" \
    --model-cache-size-gb "${MODEL_CACHE_SIZE_GB}" \
    "${cached_args[@]}"
}

case "${PART}" in
  trace) run_trace ;;
  run) run_aegaeon ;;
  all)
    run_trace
    run_aegaeon
    ;;
  *)
    echo "Unsupported --part: ${PART}"
    usage
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Latency/throughput sweep on a (confidential) GPU VM.
#
# Run setup_gpu.sh first: it pins vllm==0.8.5 and fetches benchmark_latency.py,
# which is not shipped in the wheel.
#
# A failing configuration is recorded and the sweep continues, so one bad run
# does not throw away the GPU time already paid for. Configurations whose JSON
# already exists are skipped, so a crashed sweep can be resumed by pointing
# RESULTS_DIR at the directory it was writing to.
#
# Environment overrides:
#   SWEEP=figure11|full   which configurations to run (default figure11)
#   RESULTS_DIR=<dir>     write here instead of a fresh timestamped directory
#   BENCHMARK_SCRIPT=...  path to vLLM's benchmark_latency.py
#   NUM_ITERS/NUM_WARMUP  measured / warmup iterations (defaults match the paper)
#   FORCE=1               re-run configurations that already have a JSON
set -uo pipefail
# NOTE: -e is deliberately not set. With pipefail the `python ... | tee` pipeline
# returns python's non-zero status, and -e would abort the whole sweep on the
# first failing configuration.
# export HF_HOME="/mnt"

# Model and benchmark settings
MODEL="${MODEL:-meta-llama/Llama-2-7b-hf}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
DTYPE="${DTYPE:-bfloat16}"
NUM_ITERS="${NUM_ITERS:-10}"
# benchmark_latency.py defaults to 10 warmup iterations; set explicitly so the
# methodology is visible in the script rather than inherited from the tool
NUM_WARMUP="${NUM_WARMUP:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_SCRIPT="${BENCHMARK_SCRIPT:-${SCRIPT_DIR}/../benchmarks/benchmark_latency.py}"

# Configurations as "<input_len>:<batch_size>" pairs.
#
# figure11 (default) reproduces the two one-dimensional sweeps the GPU figure is
# built from: throughput over batch size at a fixed input length, and over input
# length at a fixed batch size. This is what plot_GPUs.py consumes.
#
# full is the complete cross-product. vLLM processes a batch that exceeds the KV
# cache in several waves rather than failing, so the large configurations do run;
# they are just slow (input 2048 x batch 512 is roughly 80 s per iteration).
FIXED_INPUT="${FIXED_INPUT:-128}"
FIXED_BATCH="${FIXED_BATCH:-4}"
SWEEP_BATCHES=(1 2 4 8 16 32 64 128 256 512)
SWEEP_INPUTS=(128 256 512 1024 2048)
FULL_INPUTS=(128 1024 2048)

build_configs() {
    local sweep="$1"
    local -a configs=()
    if [[ "${sweep}" == "full" ]]; then
        for input_len in "${FULL_INPUTS[@]}"; do
            for batch in "${SWEEP_BATCHES[@]}"; do
                configs+=("${input_len}:${batch}")
            done
        done
    else
        # batch sweep at the fixed input length
        for batch in "${SWEEP_BATCHES[@]}"; do
            configs+=("${FIXED_INPUT}:${batch}")
        done
        # input sweep at the fixed batch size, skipping the shared point
        for input_len in "${SWEEP_INPUTS[@]}"; do
            if [[ "${input_len}" != "${FIXED_INPUT}" ]]; then
                configs+=("${input_len}:${FIXED_BATCH}")
            fi
        done
    fi
    printf '%s\n' "${configs[@]}"
}

SWEEP="${SWEEP:-figure11}"
if [[ "${SWEEP}" != "figure11" && "${SWEEP}" != "full" ]]; then
    echo "SWEEP must be 'figure11' or 'full', got '${SWEEP}'" >&2
    exit 1
fi
mapfile -t CONFIGS < <(build_configs "${SWEEP}")

# ---- preflight: fail before spending GPU time -------------------------------
if [[ ! -f "${BENCHMARK_SCRIPT}" ]]; then
    cat >&2 <<EOF
benchmark_latency.py not found at ${BENCHMARK_SCRIPT}

It is part of vLLM's source tree and is not installed by pip. Run ./setup_gpu.sh,
or fetch it manually:

  mkdir -p "$(dirname "${BENCHMARK_SCRIPT}")"
  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.8.5/benchmarks/benchmark_latency.py \\
    -o "${BENCHMARK_SCRIPT}"
EOF
    exit 1
fi

# importing vllm prints a platform-detection line to stdout; keep only the version
vllm_version=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null | tail -1)
if [[ -z "${vllm_version}" ]]; then
    echo "vllm is not importable. Run ./setup_gpu.sh" >&2
    exit 1
fi
if [[ "${vllm_version}" != "0.8.5" ]]; then
    echo "WARNING: vllm ${vllm_version} found, but this sweep is written for 0.8.5." >&2
    echo "         VLLM_USE_V1=0 requires the V0 engine, removed in 0.9+." >&2
fi

# Timestamped directory for results, unless resuming into an existing one
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="${RESULTS_DIR:-results_${TIMESTAMP}}"
mkdir -p "${RESULTS_DIR}"
FAILED_FILE="${RESULTS_DIR}/failed.txt"

echo "sweep=${SWEEP} (${#CONFIGS[@]} configurations), model=${MODEL}, vllm=${vllm_version}"
echo "results in ${RESULTS_DIR}"

failures=0
for config in "${CONFIGS[@]}"; do
  INPUT_LEN="${config%%:*}"
  BATCH="${config##*:}"

  LOG_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.log"
  JSON_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.json"

  if [[ -s "${JSON_FILE}" && "${FORCE:-0}" != "1" ]]; then
    echo "skipping input=${INPUT_LEN} batch=${BATCH}: ${JSON_FILE} already exists"
    continue
  fi

  echo "===== Input length = ${INPUT_LEN}, Batch size = ${BATCH} =====" | tee "${LOG_FILE}"

  if VLLM_USE_V1=0 python3 "${BENCHMARK_SCRIPT}" \
    --model "${MODEL}" \
    --input-len "${INPUT_LEN}" \
    --output-len "${OUTPUT_LEN}" \
    --batch-size "${BATCH}" \
    --dtype "${DTYPE}" \
    --output-json "${JSON_FILE}" \
    --num-iters "${NUM_ITERS}" \
    --num-iters-warmup "${NUM_WARMUP}" \
    2>&1 | tee -a "${LOG_FILE}"; then
    echo "Results ➜ log: ${LOG_FILE}, json: ${JSON_FILE}"
  else
    # Keep going: the remaining configurations are still worth the GPU time
    echo "FAILED: input=${INPUT_LEN} batch=${BATCH} (see ${LOG_FILE})" | tee -a "${FAILED_FILE}" >&2
    failures=$((failures + 1))
  fi
  echo
done

echo "sweep finished: $(( ${#CONFIGS[@]} - failures ))/${#CONFIGS[@]} configurations succeeded"
if (( failures > 0 )); then
  echo "failed configurations are listed in ${FAILED_FILE}" >&2
  echo "re-run with RESULTS_DIR=${RESULTS_DIR} to retry only those" >&2
  exit 1
fi

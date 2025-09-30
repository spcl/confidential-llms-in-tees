#!/usr/bin/env bash
set -euo pipefail
# export HF_HOME="/mnt"

# Model and benchmark settings
MODEL="meta-llama/Llama-2-7b-hf"
OUTPUT_LEN=128
DTYPE="bfloat16"

# Timestamped directory for results
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")
RESULTS_DIR="results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Input lengths (128 or 1024)
for INPUT_LEN in 128 1024 2048; do
  # Batch sizes up to 1024
  for BATCH in 1 2 4 8 16 32 64 128 256 512; do
    LOG_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.log"
    JSON_FILE="${RESULTS_DIR}/latency_in${INPUT_LEN}_bs${BATCH}.json"

    echo "===== Input length = ${INPUT_LEN}, Batch size = ${BATCH} =====" | tee "${LOG_FILE}"

    VLLM_USE_V1=0 python ../benchmarks/benchmark_latency.py \
      --model "${MODEL}" \
      --input-len "${INPUT_LEN}" \
      --output-len "${OUTPUT_LEN}" \
      --batch-size "${BATCH}" \
      --dtype "${DTYPE}" \
      --output-json "${JSON_FILE}" \
      --num-iters 10 \
      2>&1 | tee -a "${LOG_FILE}"

    echo "Results ➜ log: ${LOG_FILE}, json: ${JSON_FILE}"
    echo
  done
done

#!/usr/bin/env bash
# Prepare a GPU VM for benchmark_vllm.sh.
#
# vLLM is pinned: the sweep sets VLLM_USE_V1=0 to force the V0 engine, and
# benchmark_latency.py lives in the source tree (not the wheel), so its path and
# its JSON output format are version specific. 0.9 and later remove the V0
# engine and move the script.
set -euo pipefail

VLLM_VERSION="${VLLM_VERSION:-0.8.5}"
# Where benchmark_vllm.sh looks for the benchmark by default
BENCHMARK_DIR="${BENCHMARK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/benchmarks}"
BENCHMARK_URL="https://raw.githubusercontent.com/vllm-project/vllm/v${VLLM_VERSION}/benchmarks/benchmark_latency.py"

echo "Installing vllm==${VLLM_VERSION}"
pip install "vllm==${VLLM_VERSION}"
# vllm 0.8.5 does not cap transformers; 5.x drops tokenizer attributes the V0
# engine still reads (LlamaTokenizer has no attribute all_special_tokens_extended).
# Pin to the last line the 0.8.x engine was tested against.
pip install "transformers<4.52" "tokenizers<0.22"

echo "Fetching benchmark_latency.py and its helper from the v${VLLM_VERSION} tag into ${BENCHMARK_DIR}"
mkdir -p "${BENCHMARK_DIR}"
curl -fsSL "${BENCHMARK_URL}" -o "${BENCHMARK_DIR}/benchmark_latency.py"
# benchmark_latency.py imports benchmark_utils from its own directory; the tag
# ships it alongside, the wheel does not.
curl -fsSL "${BENCHMARK_URL%benchmark_latency.py}benchmark_utils.py" -o "${BENCHMARK_DIR}/benchmark_utils.py"

# meta-llama/Llama-2-7b-hf is gated; failing here costs seconds, failing after
# the sweep starts costs GPU time
echo "Checking access to the model"
python3 - <<'EOF'
import sys
from huggingface_hub import model_info
model = "meta-llama/Llama-2-7b-hf"
try:
    model_info(model)
except Exception as exc:
    sys.exit(
        f"Cannot access {model}: {exc}\n"
        "Request access on huggingface.co and run: huggingface-cli login"
    )
print(f"{model} is accessible")
EOF

echo "Setup complete. Run ./benchmark_vllm.sh from the GPU directory."

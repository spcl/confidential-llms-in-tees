#!/bin/bash

# fail on error
set -euxo pipefail

# Assumes system runs on Ubuntu 23.10 with kernel 5.11 or later (requirement for upstream SGX kernel drivers)
#  submodules initialized

# Get ipex 2.2
cd intel-extension-for-pytorch/
git checkout examples/cpu/inference/python/llm/distributed/run_generation_with_deepspeed.py examples/cpu/inference/python/llm/single_instance/run_generation.py examples/cpu/inference/python/llm/single_instance/run_quantization.py
git checkout release/2.2
git submodule update --recursive --init
git apply ../ipex.patch
cd -

# build 2.2 ipex docker
DOCKER_BUILDKIT=1 docker build -f intel-extension-for-pytorch//examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.2.0 intel-extension-for-pytorch/

# build 2.2 ipex sgx docker
DOCKER_BUILDKIT=1 docker build -f sgx/Dockerfile.sgx -t sgx-ipex-llm:2.2.0 .

# Turn ipex back to 2.3
cd intel-extension-for-pytorch/
git checkout examples/cpu/inference/python/llm/distributed/run_generation_with_deepspeed.py examples/cpu/inference/python/llm/single_instance/run_generation.py examples/cpu/inference/python/llm/single_instance/run_quantization.py
git checkout release/2.3
git submodule update --recursive --init
git apply ../ipex.patch
cd -

# quantize models to int8
mkdir -p models/
# Gen 7B model
docker run --rm --privileged -v $HOME/.cache:/home/ubuntu/.cache --device /dev/sgx_enclave -v $HOME/llm-benchmarking/models:/home/ubuntu/models ipex-llm:2.2.0 bash -c "source miniconda3/bin/activate && conda activate py310 && source llm/tools/env_activate.sh && sudo chown -R ubuntu:ubuntu models .cache && python llm/run.py -m meta-llama/Llama-2-7b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir models && mv models/best_model.pt models/7b-int8.pt"
docker run --rm --privileged -v $HOME/.cache:/home/ubuntu/.cache --device /dev/sgx_enclave -v $HOME/llm-benchmarking/models:/home/ubuntu/models ipex-llm:2.2.0 bash -c "source miniconda3/bin/activate && conda activate py310 && source llm/tools/env_activate.sh && sudo chown -R ubuntu:ubuntu models .cache && python llm/run.py -m meta-llama/Llama-2-13b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir models && mv models/best_model.pt models/13b-int8.pt"
docker run --rm --privileged -v $HOME/.cache:/home/ubuntu/.cache --device /dev/sgx_enclave -v $HOME/llm-benchmarking/models:/home/ubuntu/models ipex-llm:2.2.0 bash -c "source miniconda3/bin/activate && conda activate py310 && source llm/tools/env_activate.sh && sudo chown -R ubuntu:ubuntu models .cache && python llm/run.py -m meta-llama/Llama-2-70b-hf --ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp --output-dir models && mv models/best_model.pt models/70b-int8.pt"
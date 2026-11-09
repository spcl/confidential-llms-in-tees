# Confidential LLM inference benchmarking in CC

Repository to include scripts to run inference benchmarks in CC environments.

## Prerequisites

In our work we run on SPR or EMR Intel Xeon (generation 4 or older) CPUs and H100 GPUs. We used Ubuntu 24.04 as the host OS. Later Ubuntu versions should also work.

For benchmarks with SGX or TDX, please follow the respective sections on SGX or TDX setup.
For GPU benchmarks, follow the GPU section.
Finally, for RAG benchmarks, see the corresponding section. Note RAG currently only operates on CPUs.

## CPUs
### Common Setup
To setup the host for running experiments, please first initalize the repository, by cloning it and applying appropriate patches:
```sh
git clone https://github.com/spcl/confidential-llms-in-tees.git
cd confidential-llms-in-tees
git submodule update --init --recursive
cd CPU/tdx
git apply ../tdx.patch
cd ../intel-extension-for-pytorch
git apply ../ipex.patch
cd ..
```

The submodules are pinned to exact commits, so `--init` checks out the revisions
the patches were written against and no `git submodule sync` is needed:

| Submodule | Upstream | Pinned revision |
| --- | --- | --- |
| `CPU/intel-extension-for-pytorch` | intel/intel-extension-for-pytorch | `f92dcd4f` (tag `v2.3.100+cpu`) |
| `CPU/tdx` | canonical/tdx | `4f4ff286` (tag `2.0`) |
| `CPU/gramine` | gramineproject/gramine | `10e93534` (tag `v1.7`) |
| `RAG/beir` | beir-cellar/beir | `49d4338c` |

`--recursive` is required, not optional: IPEX is compiled from source (see
below) and that needs its own `third_party` submodules. Expect the recursive
clone to pull on the order of 10 GB.

The `canonical/tdx` branch this repository used to track (`noble-24.04`) no
longer exists upstream — the repository was reorganised and that history is now
only reachable through tag `2.0`, which is what the pin above points at. This is
also the revision `tdx.patch` applies to.

Then run the host setup script which will setup hugging face, create Docker, and build the necessary image:
```sh
HUGGINGFACE_TOKEN=<token> ./host_setup.sh
```
Relogin to apply changes in groups. Finally, compile the docker container. The
build context has to be the IPEX checkout itself, since the Dockerfile copies
the context into the image and builds IPEX out of it:
```sh
DOCKER_BUILDKIT=1 docker build \
    -f intel-extension-for-pytorch/examples/cpu/inference/python/llm/Dockerfile \
    -t ipex-llm:2.3.100 \
    intel-extension-for-pytorch/
```

This compiles IPEX, PyTorch, LLVM, oneCCL and DeepSpeed from source and takes
roughly 1–2 hours on a Xeon server. It is not the fast path by choice — see the
next section.

### IPEX is end-of-life: the prebuilt wheels are gone

Intel has discontinued Intel® Extension for PyTorch and revoked the published
CPU wheels. The pinned `intel-extension-for-pytorch==2.3.100+cpu` wheel is still
listed on the package index, but the file itself now answers `403 AccessDenied`:

```sh
curl -sI https://download.pytorch-extension.intel.com/ipex_stable/cpu/intel_extension_for_pytorch-2.3.100%2Bcpu-cp310-cp310-linux_x86_64.whl
```

That makes `tools/env_setup.sh 6`, the prebuilt install path, unusable — pip
resolves the wheel from the index and then fails to download it. (`oneccl-bind-pt`
is unaffected and still serves, so the failure looks like a partial outage
rather than an EOL.)

`ipex.patch` therefore changes the Dockerfile default to `ARG COMPILE=ON`,
routing the build to `tools/env_setup.sh 2` (compile from source). No
`--build-arg` is needed. If Intel ever restores the wheels, the old behaviour is
still reachable with `--build-arg COMPILE=` (an empty value).

The 2.3.100 source tree freezes its build dependencies at versions that current
toolchains reject, so `ipex.patch` also carries the following build fixes. They
are listed here because the same failures appear if you build IPEX outside this
repository's Dockerfile:

- **CMake 4 rejects the pinned oneCCL.** oneCCL `2021.11` still declares
  `cmake_minimum_required(VERSION 2.8)`, and CMake 4 removed compatibility with
  anything below 3.5 — it errors out with *"Compatibility with CMake < 3.5 has
  been removed"*. Both places that provide CMake are capped: `tools/env_setup.sh`
  installs `cmake>=3.5,<4` from conda-forge, and `scripts/compile_bundle.sh`
  installs `"cmake<4"` instead of a bare `cmake`. Capping only the conda one is
  not enough — the unpinned `pip install cmake` in `compile_bundle.sh` shadows it
  on `PATH`. `CMAKE_POLICY_VERSION_MINIMUM=3.5` is set in the Dockerfile as a
  second line of defence.
- **`ModuleNotFoundError: No module named 'pkg_resources'`.** setuptools 70
  dropped the bundled `pkg_resources` that the frozen build scripts still import.
  Both conda environments (`compile_py310` for the build, `py310` for the
  runtime) pin `setuptools<70`, and the runtime pin is reapplied at the end of
  the deploy stage so that it survives the dependency installs in between.

### SGX Setup

Please follow the script in ```sgx_setup.sh```. It installs the dependencies for Gramine and builds and installs Gramine.

Following the SGX setup should allow you to run the following hello world Gramine example.

```
cd CPU/gramine/CI-Examples/helloworld
make SGX=1
gramine-sgx helloworld
```
In case you encounter errors related to Gramine, please refer to [its documentation](`https://gramine.readthedocs.io/en/stable/`) for debugging instructions.  

### TDX Setup
#### Prepare a TDX VM image
Use TDX guest tools to generate a TDX VM image. By default, we create a 300GB image but it should be at least 200GB (required for 70B Llama2 model). For more in depth treatment such as BIOS configuration for TDX, follow the instructions within the [Ubuntu's TDX](https://github.com/canonical/tdx) repository. In short, run:
```sh
cd CPU/tdx/guest-tools/image/
sudo ./create-td-image.sh
```
Update the `td_guest.xml` to point to the newly created image. Then, define and start the TD:
```sh
sudo virsh define td_guest.xml
sudo virsh start tdx
```
The default PW of user `ubuntu` is `123456`. The default port on which the VM will be available is 10022.
If you run into permission issues, it might be useful to copy the qcow2 file to libvirt's images:
```sh
sudo cp ~/confidential-llms-in-tees/CPU/tdx/guest-tools/image/tdx-guest-ubuntu-24.04-generic.qcow2 /var/lib/libvirt/images/
```
Consider creating an ssh key and copying it to the running TD for faster login.

#### Copy the repository to the VM
Initialize the repository in the VM exactly as outlined above in host setup or use `rsync` to copy the files to the VM:
```sh
rsync -avzog --exclude 'CPU/tdx/' -e 'ssh -p 10022' confidential-llms-in-tees/ tdx@localhost:~/confidential-llms-in-tees
```
SSH to the VM and run the host setup script:
```sh
ssh -p 10022 tdx@localhost
cd confidential-llms-in-tees/CPU
HUGGINGFACE_TOKEN=<token> ./host_setup.sh
```
Relogin to apply changes in groups. Finally, compile the docker container:
```sh
cd ~/confidential-llms-in-tees/CPU/intel-extension-for-pytorch/
DOCKER_BUILDKIT=1 docker build -f examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.100 .
```

#### Enable hugepages
In case you would like to measure the VMs with enabled 1GB hugepages, first modify Grub configuration in `/etc/default/grub` (e.g., for `<num_hugepages>=300`)
```sh
GRUB_CMDLINE_LINUX="nomodeset kvm_intel.tdx=1 default_hugepagesz=1G hugepagesz=1G hugepages=<num_hugepages> transparent_hugepages=always"
```
Then system.ctl `/etc/sysctl.conf`
```sh
vm.nr_hugepages=<num_hugepages>
```
Update grub
```sh
sudo update-grub
sudo reboot
```

To verify that the hugepages are indeed enabled, after reboot run:
```sh
cat /proc/meminfo | grep HugePages
```
which should report `<num_hugepages>`. 

Once rebooted, remember to use the hugepages version of the `.xml` VM definition file and modify it with `<num_hugepages>`. Then define and start this new VM:
```sh
sudo virsh define td_guest-hugepages.xml
sudo virsh tdx-hugepages
```
As of writing this, TDX does not support hugepages, so if you allocate 300GB of 1GB pages, it will still try to use 2MB pages and you might run out of memory. We used these pages only for VM measurements, and for TDX we used the default pages.

### Running baseline experiments

```sh
nohup ./run.sh baseline &
```

This will generate a folder under `results/` with the current date and time and add an entry into the experiment log. All generated files will have the form `baseline-system-in_size-out_size-vCPUs-numa-batch_size-model-data_type.txt`.

### Running TDX experiments
SSH to the running TDX VM as created above.
```sh
ssh -P 10022 root@localhost
```

Run the experiments via:

```sh
nohup ./run.sh tdx &
```

### Running SGX experiments

#### Preparing the docker image for SGX

Requires image ipex-llm:2.3.100 to already exist. Create SGX/graminized version of the docker image
by running:

```sh
DOCKER_BUILDKIT=1 docker build -f sgx/Dockerfile.sgx -t sgx-ipex-llm:2.3.100 .
```

#### Running SGX docker image for Benchmark
Run the docker image and then before running a workload activate environment:

```sh
source ./llm/tools/env_activate.sh
```

Run a workload - preferably on a single socket:
```sh
numactl -N 0,1 -m 0,1 -C 0-31 gramine-sgx LLM ~/llm/single_instance/run_generation.py --dtype bfloat16 -m meta-llama/Llama-2-7b-hf --input-tokens 1024 --max-new-tokens 128 --num-iter 30 --num-warmup 5 --batch-size 1 --greedy --benchmark
```

### Quantizing models
To quantize the models, follow `genQuantLLamaModels.sh`.

### Processing Results

`run_parser.py` gathers all token latencies from each experiments and places
them into a csv file. It accepts one or more results folders to look in for
results files, searched recursively. Any file ending in `.txt` is considered a
result file.

```sh
python processing/run_parser.py results/<date>-<time> [results/<date>-<time> ...]
```

The resulting `results.csv` is written to the current directory. These can be parsed by some of the plotting helper functions we provide in `/processing`.

#### System labels

The plotting scripts select rows by exact match on the `system` column, which
by default is the first argument you passed to `run.sh`. Each script expects a
specific set of labels:

| Script | Expected `system` labels | `numa` |
| --- | --- | --- |
| `AMX_latency.py`, `AMX_throughput.py` | `VM (AMX)`, `TDX (AMX)`, `VM (no AMX)`, `TDX (no AMX)` | `2s` / `1s` |
| `model_scaling_single_socket.py` | `baremetal`, `VM`, `TDX`, `SGX` | `1s` |
| `model_scaling_double_socket.py` | `baremetal`, `VM FH`, `VM TH`, `TDX` | `2s` |
| `model_scaling_70B.py` | `VM B`, `TDX`, `VM NB` | `2s` |
| `vCPUs_batch_size.py`, `vCPUs_input.py` | `baremetal`, `VM`, `TDX` | — |

Labels containing spaces or parentheses cannot be passed to `run.sh`, as they
become part of the result file names. To attach such a label, put it in a
`system.txt` file next to the results; it then overrides the label for every
result file in that directory:

```sh
echo "TDX (no AMX)" > results/<date>-<time>/system.txt
python processing/run_parser.py results/<date>-<time>
```

Labels that do contain dashes (e.g. `TDX-no-AMX`) are parsed correctly from the
file name. If a filter matches no rows, the scripts report which labels were
requested and which ones the CSV contains instead of drawing an empty figure.

### Tracing
To obtain traces, start the Docker container:
```
docker run --rm --privileged --shm-size=2gb -it -v /home/mchrapek/.cache:/home/ubuntu/.cache ipex-llm:2.3.100 bash 
```
Inside run the inference command with `--profile`, e.g.:
```
cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank --num_accelerators 1 --bind_core_list 0-59 distributed/run_generation_with_deepspeed.py --deployment-mode --benchmark -m meta-llama/Llama-2-7b-hf --ipex --batch-size 4 --num-iter 15 --num-warmup 5 --max-new-tokens 128 --input-tokens 128 --token-latency --greedy --profile
```
This will generate log files which can be processed and plotted by `traces_parser.py`. It accepts two files with traces that correspond to two compared systems.

## GPU

### Machines
The GPU numbers compare a confidential VM against a non-confidential one, so the
same sweep has to run twice. Azure offers no bare metal H100, so we use a matched
pair of single-H100 VMs and keep everything else identical between the two runs:

| Series | SKU | $/hour | Confidential |
| --- | --- | --- | --- |
| `cGPU` | `Standard_NCC40ads_H100_v5` | 8.90 | yes, CC mode on |
| `GPU` | `Standard_NC40ads_H100_v5` | 9.08 | no |

Prices are West Europe, Linux, pay-as-you-go, from the Azure Retail Prices API
with the Spot and Low Priority meters excluded (verified 2026-11-09). The
confidential SKU is about 2% cheaper than the non-confidential one, so in cost
per token the confidential VM's performance overhead is slightly offset rather
than compounded.

### Setup
vLLM must be **0.8.5**. The sweep sets `VLLM_USE_V1=0` to force the V0 engine,
which 0.9 and later removed, and it calls `benchmark_latency.py`, which lives in
vLLM's source tree rather than in the wheel and moved in later releases. The
setup script pins the version, fetches that script from the matching tag, and
checks that the (gated) model is accessible before any GPU time is spent:

```sh
cd GPU
./setup_gpu.sh
```

### Running the benchmark
```sh
./benchmark_vllm.sh
```

By default this runs the 14 configurations behind the GPU figure: a batch size
sweep (1 → 512) at input length 128, and an input length sweep (128 → 2048) at
batch size 4. Use `SWEEP=full` for the complete 3 × 10 cross-product of input
lengths {128, 1024, 2048} and batch sizes 1 → 512 (roughly 1.6 GPU-hours).
Batches whose KV cache exceeds GPU memory are not an error: vLLM processes them
in several waves, so the large configurations run, they are just slow.

A failing configuration is recorded in `<results dir>/failed.txt` and the sweep
continues. To retry only the failures, re-run with `RESULTS_DIR` set to the
existing directory; configurations that already produced a JSON are skipped.

### Processing results
Run these on your laptop, not on the GPU VM. `parse.py` prints cost per million
generated tokens, in the format used by the `gpu_cc_cost` / `gpu_raw_cost`
constants in `CPU/processing/vCPUs_*.py`. The price of the VM the results came
from has to be given explicitly, since the two VMs are priced differently:

```sh
python parse.py <cGPU results dir> --sku NCC40ads_H100_v5   # $8.90/h
python parse.py <GPU results dir>  --sku NC40ads_H100_v5    # $9.08/h
```

Use `--cost <usd_per_hour>` for a region or SKU outside that table. The chosen
price is echoed on stderr, so stdout stays a dictionary that can be pasted
directly into the plotting scripts.

Plot the comparison by passing both directories:

```sh
python plot_GPUs.py <cGPU results dir> <GPU results dir>
```

Throughput is counted in **generated** tokens, `output_len * batch / latency`,
matching the CPU side. Note that results produced before this was corrected used
the input length instead, which understated cost per token by
`input_len / output_len` for every run with an input length other than 128, and
priced both VMs at $6.98/hour, which matches neither SKU. Cost constants
generated before those two corrections need to be regenerated.

## RAG
Make sure you have your submodules initialized. Then, enter the RAG directory and apply the patch:
```
cd RAG/beir
git apply ../beir.patch
```
Start the elasticsearch database:
```
cd RAG
docker compose up elasticsearch
```
To build and run the benchmarks container:
```
docker compose run --rm --build rag
```
Within just run:
```
./run.sh
```

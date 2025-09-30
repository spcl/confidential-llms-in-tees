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
git submodule sync
git submodule update --init --recursive
cd CPU/tdx
git apply ../tdx.patch
cd ../intel-extension-for-pytorch
git apply ../ipex.patch
cd ..
```
Then run the host setup script which will setup hugging face, create Docker, and build the necessary image:
```sh
HUGGINGFACE_TOKEN=<token> ./host_setup.sh
```
Relogin to apply changes in groups. Finally, compile the docker container:
```sh
DOCKER_BUILDKIT=1 docker build -f intel-extension-for-pytorch/examples/cpu/inference/python/llm/Dockerfile -t ipex-llm:2.3.100 .
```

### SGX Setup

Please follow the script in ```sgx_setup.sh```. It installs the dependencies for Gramine and builds and installs Gramine.

Following the SGX setup should allow you to run the following hello world Gramine example.

```
cd gramine/CI-Example/helloworld
make SGX=1
gramine-sgx helloworld
```
In case you encounter errors related to Gramine, please refer to [its documentation](`https://gramine.readthedocs.io/en/stable/`) for debugging instructions.  

### TDX Setup
#### Prepare a TDX VM image
Use TDX guest tools to generate a TDX VM image. By default, we create a 300GB image but it should be at least 200GB (required for 70B Llama2 model). For more in depth treatment such as BIOS configuration for TDX, follow the instructions within the [Ubuntu's TDX](https://github.com/canonical/tdx) repository. In short, run:
```sh
cd tdx/guest-tools/image/
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
sudo cp ~/confidential-llms-in-tees/tdx/guest-tools/image/tdx-guest-ubuntu-24.04-generic.qcow2 /var/lib/libvirt/images/
```
Consider creating an ssh key and copying it to the running TD for faster login.

#### Copy the repository to the VM
Initialize the repository in the VM exactly as outlined above in host setup or use `rsync` to copy the files to the VM:
```sh
rsync -avzog --exclude tdx/ -e 'ssh -p 10022' confidential-llms-in-tees/ tdx@localhost:~/confidential-llms-in-tees
```
SSH to the VM and run the host setup script:
```sh
ssh -p 10022 tdx@localhost
cd confidential-llms-in-tees
HUGGINGFACE_TOKEN=<token> ./host_setup.sh
```
Relogin to apply changes in groups. Finally, compile the docker container:
```sh
cd confidential-llms-in-tees/intel-extension-for-pytorch/
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
them into a csv file. It requires a single argument for the results folder to
look in for results files. Any file ending in `.txt` is considered a result
file.

```sh
python parse_results.py results/<date>-<time>
```

The resulting CSV file will be stored in `results/<date>-<time>`. These can be parsed by some of the plotting helper functions we provide in `/processing`.

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
GPUs require vLLM. Follow their installation instructions to enable them on your system.
You can then run the benchmark using:
```
./benchmark_vllm.sh
```
You can parse these using `parse.py` and plot using `plot_GPUs.py`.

## RAG
Enter the RAG directory and apply the patch:
```
cd RAG/beir
git apply ../beir.patch
```
Then start the elasticsearch database:
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

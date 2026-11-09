#!/bin/bash
# Parameterized CPU (TDX/VM) sweep runner for the targeted experiments.
#
# Presets (SWEEP=):
#   batch  -> fixed input 128, batch 1..512, bf16+int8  (paper Fig 8/9/12)
#   input  -> fixed batch 64, input 32..2048, bf16+int8 (paper Fig 10)
#   both   -> batch then input (default)
#   single -> one configuration: INPUT (default 128) x BATCH (default 64)
#
# Usage:
#   SWEEP=input  ./run_cpu_sweeps.sh "TDX (AMX)"     # on the confidential VM
#   SWEEP=both   ./run_cpu_sweeps.sh "VM (AMX)"      # on the non-confidential VM
#   RANKS=2 VCPU_LIST="$(seq 0 2 126 | paste -sd,)" SWEEP=batch ./run_cpu_sweeps.sh "TDX dual"
#
# Env overrides:
#   MODELS, QUANTS, IMAGE
#   VCPU_LIST   CPUs to run on. Default: the first hardware thread of every
#               physical core, so hyperthread siblings are never both used.
#               Accepts ranges and comma lists (0-31, 0,2,4,...,62, 0-15,32-47).
#   RANKS       DeepSpeed ranks (default 1). Use one per socket on multi-socket
#               VMs so each rank's memory stays on its own NUMA node.
#   BINDFLAGS   DeepSpeed binding flags. Default pins ranks to VCPU_LIST and
#               their memory to the local node; set BINDFLAGS="" to run unbound.
#   BEAM_SEARCH Set to 1 to restore the IPEX default of beam 4 above batch 1
#               (paper Fig 4). Default is greedy (beam 1) at every batch size,
#               which is what the paper's scaling and cost figures use.
# Deliberately NOT set -e: one failing config is logged and skipped, not fatal.
set -uo pipefail

config="${1:?Pass a label, e.g. \"TDX (AMX)\" or \"VM (AMX)\"}"
SWEEP="${SWEEP:-both}"

# --- one hardware thread per physical core by default ---------------------------
# Azure VMs enumerate hyperthread siblings interleaved (CPU 0 and 1 share core 0),
# bare metal usually enumerates them in blocks (CPU 0 and 32 share core 0). A
# range such as 0-31 therefore means 32 cores on one machine and 16 cores run
# twice on the other. Ask the kernel instead of guessing.
if [[ -z "${VCPU_LIST:-}" ]]; then
    VCPU_LIST="$(lscpu -p=CPU,CORE | awk -F, '!/^#/ && !seen[$2]++ {print $1}' | paste -sd,)"
fi
# Count CPUs across comma-separated entries, each a single CPU or an a-b range.
vCPUs_num=$(awk -v RS=, -F- 'NF {n += (NF == 2) ? $2 - $1 + 1 : 1} END {print n}' <<<"$VCPU_LIST")
RANKS="${RANKS:-1}"
BINDFLAGS="${BINDFLAGS---bind_cores_to_rank --bind_core_list $VCPU_LIST}"

MODELS="${MODELS:-meta-llama/Llama-2-7b-hf}"
QUANTS="${QUANTS:-bf16 int8}"
IMAGE="${IMAGE:-ipex-llm:2.3.100}"
OUT_TOKEN=128

# --- sweep definitions -------------------------------------------------------
case "$SWEEP" in
batch) PAIRS=$(for b in 1 2 4 8 16 32 64 128 256 512; do echo "128:$b"; done) ;;
input) PAIRS=$(for i in 32 64 128 256 512 1024 2048; do echo "$i:64"; done) ;;
both) PAIRS=$({
    for b in 1 2 4 8 16 32 64 128 256 512; do echo "128:$b"; done
    for i in 32 64 128 256 512 1024 2048; do echo "$i:64"; done
}) ;;
single) PAIRS="${INPUT:-128}:${BATCH:-64}" ;;
*)
    echo "SWEEP must be batch|input|both|single" >&2
    exit 1
    ;;
esac

date=$(date +"%F-%H-%M")
directory="results/$date"
mkdir -p "$directory"
echo "label='$config' sweep=$SWEEP cores=$VCPU_LIST ($vCPUs_num vCPU) ranks=$RANKS bind='${BINDFLAGS}' -> $directory"
lscpu &>"$directory/lscpu.out"
numactl --hardware &>"$directory/numactl-hw.out" || true

run_one() {
    local in_token="$1" batch_size="$2" model="$3" dt="$4"
    case "$model" in *7b*) mt=7b ;; *13b*) mt=13b ;; *70b*) mt=70b ;; *) mt=other ;; esac
    local name="$directory/${config}-${in_token}in-${OUT_TOKEN}out-${vCPUs_num}vCPU-1s-${batch_size}bs-${mt}-${dt}"
    [[ -s "$name.txt" ]] && grep -q "Finished" "$name.txt" 2>/dev/null && {
        echo "skip: $(basename "$name")"
        return
    }

    local quant=''
    [[ "$dt" == int8 ]] && quant='--ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp'
    local num_iter=50 num_warmup=10 greedy='--greedy'
    if [[ "$batch_size" -ne 1 ]]; then
        num_iter=25 num_warmup=5
        [[ "${BEAM_SEARCH:-0}" == 1 ]] && greedy=''
    fi

    local cmd=(docker run --rm --privileged --shm-size=2gb -v "$HOME/.cache:/home/ubuntu/.cache" "$IMAGE" bash -c
        "cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --num_accelerators $RANKS $BINDFLAGS distributed/run_generation_with_deepspeed.py --deployment-mode --profile --benchmark -m $model $quant --ipex --batch-size $batch_size --num-iter $num_iter --num-warmup $num_warmup --max-new-tokens $OUT_TOKEN --input-tokens $in_token --token-latency $greedy")
    echo "${cmd[@]}" >"$name.txt"
    if "${cmd[@]}" &>>"$name.txt"; then
        echo "ok: $(basename "$name")"
    else echo "FAILED: $(basename "$name")" | tee -a "$directory/failed.txt" >&2; fi
}

for model in $MODELS; do
    for pair in $PAIRS; do
        in_token="${pair%%:*}"
        batch_size="${pair##*:}"
        for dt in $QUANTS; do run_one "$in_token" "$batch_size" "$model" "$dt"; done
    done
done
echo "Finished sweep=$SWEEP -> $directory"

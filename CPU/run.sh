#!/bin/bash

# exit on failure
set -euxo pipefail

config=$1

config_num_iter=50
config_num_warmup=10
config_out_token=128
config_in_token=1024
config_procs=120
config_socket=$(( config_procs / 2 )) 

# per date folder
date=$(date +"%F-%H-%M")
directory=results/$date
mkdir -p $directory

echo "storing results in $directory"
echo "$1 stored in $directory" >> experiment.log

{
    lscpu &> $directory/lscpu.out
    lshw &> $directory/lshw.out
    numactl --hardware &> $directory/numactl-hw.out

    # initialize variables with different values
    for vCPUs in '1' '1-2' '1-4' '1-8' '1-16' '1-32' '1-48' '0-59'; do # if you want to use all cores available to the system just leave empty ''; if you want to use cores accross sockets, you can use `--num_accelerators 2` in the main command
        for batch_size in 1 2 4 8 16 32 64 128; do
            for in_token in 32 64 128 256 512 1024 2048; do
                for out_token in 128; do
                    for quant in '' '--ipex-weight-only-quantization --weight-dtype INT8 --quant-with-amp'; do # needs first quantizing
                        for model in 'meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Llama-2-70b-hf'; do # 
                            # other working options 'EleutherAI/gpt-j-6b' 'tiiuae/falcon-7b' 'baichuan-inc/Baichuan2-7B-Chat' 'Qwen/Qwen-7B-Chat' 'meta-llama/Meta-Llama-3-8B'
                            # for these you need to modify the name outputting
                            # not working 'mosaicml/mpt-7b' (error) 'liuhaotian/llava-v1.5-7b' (no class) 'mistralai/Mistral-7B-v0.1' (error)                    
                            num_iter=$config_num_iter
                            num_warmup=$config_num_warmup
                            # cmp output name
                            name=$directory/$1
                            name=$name-${in_token}in
                            name=$name-${out_token}out
                            if [[ -n ${vCPUs//[[:space:]]/} ]]; then
                                vCPUs_num=$(awk -F- '{print (NF==1)?$1:($2-$1+1)}' <<< "$vCPUs")
                                vCPUs="--bind_core_list $vCPUs"
                            else
                                vCPUs_num=$config_procs
                            fi
                            name=$name-${vCPUs_num}vCPU
                            if (( vCPUs_num > config_socket )); then
                                numa='2s'
                            else
                                numa='1s'
                            fi
                            name=$name-$numa
                            name=$name-${batch_size}bs
                            if [[ $model == *"7b"* ]]; then
                                name=$name-7b
                            elif [[ $model == *"13b"* ]]; then
                                name=$name-13b
                            elif [[ $model == *"70b"* ]]; then
                                name=$name-70b
                                num_iter=$(( $num_iter/2 ))
                                num_warmup=$(( $num_warmup/2 ))
                            fi
                            if [ ! -z "${quant}" ]; then
                                name=$name-int8
                            else
                                name=$name-bf16
                            fi
                            # set greedy if single batch
                            greedy=''
                            if [[ "$batch_size" -eq 1 ]]; then
                                greedy='--greedy'
                            else
                                num_iter=$(( $num_iter/2 ))
                                num_warmup=$(( $num_warmup/2 ))
                            fi

                            # safety
                            if [[ "$num_iter" -le 1 ]]; then
                                num_iter=5
                            fi
                            if [[ "$num_warmup" -le 1 ]]; then
                                num_warmup=2
                            fi

                            cmd=(docker run --rm --privileged --shm-size="2gb" -v $HOME/.cache:/home/ubuntu/.cache ipex-llm:2.3.100 bash -c \
                                "cd llm && source ../miniforge3/bin/activate && conda activate py310 && source tools/env_activate.sh && sudo chown -R 1000:1000 ~/.cache && deepspeed --bind_cores_to_rank $vCPUs distributed/run_generation_with_deepspeed.py --deployment-mode --profile --benchmark -m $model $quant --ipex --batch-size $batch_size --num-iter $num_iter --num-warmup $num_warmup --max-new-tokens $out_token --input-tokens $in_token --token-latency $greedy" )

                            # log cmd
                            echo "${cmd[@]}" > $name.txt

                            # run cmd
                            "${cmd[@]}" &>> $name.txt

                            # Finished run
                            echo "Finished"
                            exit 0
                        done
                    done
                done
            done
        done
    done
# store run log
} &> $directory/run.out



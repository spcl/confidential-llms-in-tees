#!/bin/sh

set -e

workdir=quant-models-$(date +"%F-%H-%M")
echo "Starting quantizing models and storing output in $workdir"

mkdir -p $workdir

for m in Llama-2-7b-chat-hf Llama-2-13b-chat-hf Llama-2-70b-chat-hf
do
    model=meta-llama/$m
    for type in int8 #int8-bf16-mixed
    do
        python3 single_instance/run_llama_quantization.py --ipex-smooth-quant  --output-dir $workdir/$type --$type -m $model
        mv $workdir/$type/best_model.pt $workdir/$m-$type.pt
        rmdir $workdir/$type
    done;
done;
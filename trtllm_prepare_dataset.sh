#!/bin/bash
model_name=nvidia/Llama-3.3-70B-Instruct-FP8
num_requests=3000
isl=8192
osl=1024
dataset_file=$HOME/code/trtllm/datasets/random.txt

python benchmarks/cpp/prepare_dataset.py \
    --tokenizer=${model_name} \
    --stdout token-norm-dist \
    --num-requests=${num_requests} \
    --input-mean=${isl} \
    --output-mean=${osl} \
    --input-stdev=0 \
    --output-stdev=0 > $dataset_file

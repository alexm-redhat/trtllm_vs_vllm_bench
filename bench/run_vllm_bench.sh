#!/bin/bash

source run_config.sh

for model_path in ${model_path_list}; do
    model_name=${model_path##*/}
    
    results_dir="../results/${model_name}/vllm_isl_${isl}_osl_${osl}_tp_${num_gpus}"
    mkdir -p ${results_dir}
    rm ${results_dir}/*

    for concurrency in ${concurrency_list}; do
        num_prompts=$((concurrency * multi_round))
        printf -v padded_concurrency "%05d" ${concurrency}

        CUDA_VISIBLE_DEVICES=${cuda_gpus} $VLLM_PROFILE_CMD \
            vllm bench throughput \
                --model ${model_path} \
                --dataset-name random \
                --disable-log-stats \
                --tensor-parallel-size ${num_gpus} \
                --input-len ${isl} \
                --output-len ${osl} \
                --num-prompts ${num_prompts} \
                --max-num-seqs ${concurrency} \
                --output-json "${results_dir}/vllm_results__model_${model_name}__isl_${isl}__osl_${osl}__TP_${num_gpus}__num-prompts_${num_prompts}__batch_${padded_concurrency}.json" \
                --no-enable-prefix-caching
    done
done
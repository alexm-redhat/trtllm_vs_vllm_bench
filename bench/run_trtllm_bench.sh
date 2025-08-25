#!/bin/bash

source run_config.sh


dataset_file=../datasets/random.txt
ignore_eos="--eos_id -1"

for model_path in ${model_path_list}; do
    model_name=${model_path##*/}
    
    results_dir="../results/${model_name}/trtllm_isl_${isl}_osl_${osl}_tp_${num_gpus}"
    mkdir -p ${results_dir}
    rm ${results_dir}/*

    for concurrency in ${concurrency_list}; do
        num_requests=$((concurrency * multi_round))
        printf -v padded_concurrency "%05d" ${concurrency}

        CUDA_VISIBLE_DEVICES=${cuda_gpus} $TRT_PROFILE_CMD \
        trtllm-bench \
                --model ${model_path} \
                ${mode} \
                --backend pytorch \
                --dataset $dataset_file \
                --num_requests ${num_requests} \
                --max_batch_size ${concurrency} \
                --tp ${num_gpus} \
                --ep ${num_gpus} \
                --report_json "${results_dir}/trtllm_results__model_${model_name}__isl_${isl}__osl_${osl}__TP_${num_gpus}__num-prompts_${num_requests}__batch_${padded_concurrency}.json" \
                --streaming \
                $ignore_eos 
                
                #--extra_llm_api_options ./extra-llm-api-config-2.yml \
    done
done
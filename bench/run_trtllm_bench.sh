#!/bin/bash

source run_config.sh


dataset_file=../datasets/random.txt
ignore_eos="--eos_id -1"

# DEBUG: NSYS profile params
#
#-e TLLM_PROFILE_RECORD_GC=1,TLLM_LLMAPI_ENABLE_NVTX=1,TLLM_TORCH_PROFILE_TRACE=trace.json \
#-c cudaProfilerApi
#TLLM_PROFILE_START_STOP=100-200 
#-t 'cuda,nvtx,python-gil'
# TLLM_PROFILE_START_STOP=50-100
# nsys profile \
#     -t cuda \
#     -o run1 \
#     -f true \
#     --trace-fork-before-exec=true \
#     --cuda-graph-trace=node \
#     --delay 30 --duration 60 \

# DEBUG: Other possible params for trtllm
#
#--concurrency ${concurrency} \
#--enable_chunked_context \
#--kv_cache_free_gpu_mem_fraction 0.9 \
#--max_batch_size 64 \
#--max_num_tokens 16384 \
#--max_seq_len 16384 \
#--extra_llm_api_options $llm_options \
#--ep ${num_gpus} \
#--pp ${num_gpus} \
#--max_batch_size 1024 \
#--max_seq_len 1024 \
#--kv_cache_free_gpu_mem_fraction 0.9 \

for model_path in ${model_path_list}; do
    model_name=${model_path##*/}
    
    results_dir="../results/${model_name}/trtllm_isl_${isl}_osl_${osl}_tp_${num_gpus}"
    mkdir -p ${results_dir}
    rm ${results_dir}/*

    for concurrency in ${concurrency_list}; do
        num_requests=$((concurrency * multi_round))
        printf -v padded_concurrency "%05d" ${concurrency}

        CUDA_VISIBLE_DEVICES=${cuda_gpus} \
        trtllm-bench \
                --model ${model_path} \
                ${mode} \
                --backend pytorch \
                --dataset $dataset_file \
                --num_requests ${num_requests} \
                --max_batch_size ${concurrency} \
                --tp ${num_gpus} \
                --report_json "${results_dir}/trtllm_results__model_${model_name}__isl_${isl}__osl_${osl}__TP_${num_gpus}__num-prompts_${num_prompts}__batch_${padded_concurrency}.json" \
                --streaming \
                $ignore_eos
    done
done
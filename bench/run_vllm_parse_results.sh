#!/bin/bash

RESULTS_DIR=../results/Llama-3.3-70B-Instruct-FP8/vllm_isl_8192_osl_1024_tp_4
python vllm_parse_results.py \
    --results-dir=$RESULTS_DIR \
    --output-file=$RESULTS_DIR/parsed_results.txt

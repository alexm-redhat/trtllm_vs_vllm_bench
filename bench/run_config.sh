# GPUs to use
cuda_gpus="0,1,2,3,4,5,6,7"
num_gpus=8

# Run configs
#concurrency_list="1 16 32 64 128 256 512 1024"
concurrency_list="1" #1 16 32 64 128 256 512 1024"
multi_round=2
isl=8192
osl=1024
model_path_list="nvidia/Llama-3.3-70B-Instruct-FP8"

# Same for vllm and trtllm
mode=throughput # latency


## TRTLLM specific env vars
# TODO: Add here if necessary

## VLLM specific env vars

# Enable flashinfer MOE
#export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_USE_FLASHINFER_MOE_FP8=1

# Choose flashinfer MOE backend
export VLLM_FLASHINFER_MOE_BACKEND="throughput"
#export VLLM_FLASHINFER_MOE_BACKEND="latency"

# Enable CUTLASS MLA
export VLLM_ATTENTION_BACKEND=CUTLASS_MLA



# GPUs to use
cuda_gpus="0,1,2,3,4,5,6,7"
num_gpus=8

# Run configs
# concurrency_list="1 16 32 64 128 256 512 1024"
concurrency_list="1" #1 16 32 64 128 256 512 1024"
# concurrency_list="2048"
multi_round=1024
isl=8192
osl=1024
model_path_list="nvidia/Llama-3.3-70B-Instruct-FP8"
# model_path_list="deepseek-ai/DeepSeek-R1-0528"

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


# NSYS start-end time range (function of batch size)
## batch_size = 2048
# export PROFILE_DELAY=660
# export PROFILE_DURATION=30
## batch_size = 1
export PROFILE_DELAY=300
export PROFILE_DURATION=5

# NSYS PROFILE CMD
## Disable
# export PROFILE_CMD=""
# export TRT_PROFILE_CMD=""
# export VLLM_PROFILE_CMD=""
## Enable
export PROFILE_CMD="nsys profile \
        -s cpu \
        -b fp \
        -t cudnn,cuda,nvtx,osrt \
        -f true \
        -c cudaProfilerApi \
        --trace-fork-before-exec=true \
        --delay $PROFILE_DELAY --duration $PROFILE_DURATION \
        --cuda-graph-trace=node"
export TRT_PROFILE_CMD="${PROFILE_CMD} -o run1_trtllm"
export VLLM_PROFILE_CMD="${PROFILE_CMD} -o run1_vllm"

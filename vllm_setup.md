### VLLM Setup

Use commands below to setup vllm python package inside a virtual environment.

Create venv:

```bash
uv venv vllm --python 3.12 --seed
source vllm/bin/activate
```

Here is a summary of vllm incremental compilation steps that include the flashinfer library (based on https://docs.vllm.ai/en/latest/contributing/incremental_build.html?h=incremental). 

```bash
# 
VLLM_USE_PRECOMPILED=1 uv pip install -U -e .[flashinfer] --torch-backend=auto
uv pip install -r requirements/build.txt --torch-backend=auto

# Creates compile configs
python tools/generate_cmake_presets.py

# Creates compile dir
cmake --preset release

# This is the compilation command that can be repeated
cmake --build --preset release --target install
```
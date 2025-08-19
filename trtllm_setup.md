### TRTLLM Setup

Use commands below to setup trtllm python package inside a virtual environment.

Create venv:

```bash
uv venv trtllm --python 3.12 --seed
source trtllm/bin/activate
```

For B200 GPU:
```bash
uv pip install -U torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Setup openmpi library:

```bash
sudo apt-get -y install libopenmpi-dev
```

If you do not have sudo or experience issues with apt-get installation of openmpi, then you can point 
to a manual installation of openmpi:

```bash
# Change this to your dir
export OPENMPI_DIR=/usr/mpi/gcc/openmpi-4.1.7rc1  

# Set bin/lib
export PATH=$PATH:$OPENMPI_DIR/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPENMPI_DIR/lib
```

Install trtllm:

```bash
uv pip install -U pip setuptools
uv pip install -U tensorrt_llm
```

You may experience cuda version mismatch on ubuntu. To fix it, run this command:

```bash
uv pip install -U cuda-python==12.8
```

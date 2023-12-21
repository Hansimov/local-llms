## Local-LLMs
Personal notes for running LLM in local machine with CPU and GPUs.

All these commands are run on `Ubuntu 22.04.2 LTS`.

## Installation

### Install `huggingface_hub`

This is used to use `huggingface-cli` to download models.

```sh
pip install 'huggingface_hub[cli,torch]'
```

### Download models from HuggingFace

Example of download:

- https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF
- `dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf`

```sh
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/dolphin-2.5-mixtral-8x7b-GGUF dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf --local-dir ./models/ --local-dir-use-symlinks False
```

### Install NVIDIA CUDA Toolkit

This is to enable GPU acceleration.

```sh
# nvcc --version
sudo apt install nvidia-cuda-toolkit
```

### Install `llama-cpp-python`

```sh
LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python
```

If you have installed `llama-cpp-python` before setup nvcc correctly, you need setup nvcc first, then reinstall `llama-cpp-python`:

```sh
LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## Usage

Here is a quick and dirty exmample script. (Improvements are in progress.)

```py
from llama_cpp import Llama

model_path = "./models/dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf"
llm = Llama(
    model_path=model_path,
    n_gpu_layers=30,
    n_threads=6,
    n_ctx=3584,
    n_batch=521,
    verbose=True,
)

prompt = (
    "USER:\nWho are you?\nASSISTANT:\n"
)
stream = llm.create_completion(
    prompt,
    stream=True,
    repeat_penalty=1.2,
    max_tokens=64,
    temperature=0.1,
    stop=["USER:", "ASSISTANT:", "User:", "Assistant:"],
    echo=False,
)

for chunk in stream:
    print(chunk["choices"][0]["text"], end="")

```
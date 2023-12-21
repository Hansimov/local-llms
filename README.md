## Local-LLMs
Personal notes for running LLM in local machine with CPU and GPUs.

All these commands are run on `Ubuntu 22.04.2 LTS`.

## Installation

### Install `huggingface_hub`

This is used to use `huggingface-cli` to download models.

> See: **Installation of huggingface_hub**
> - https://huggingface.co/docs/huggingface_hub/installation#install-optional-dependencies

```sh
pip install 'huggingface_hub[cli,torch]'
```

### Download models from HuggingFace

> See: **Model card of dolphin-2.5-mixtral-8x7b-GGUF**
> - https://huggingface.co/TheBloke/dolphin-2.5-mixtral-8x7b-GGUF
> - `dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf`

Example of download:

```sh
HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download TheBloke/dolphin-2.5-mixtral-8x7b-GGUF dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf --local-dir ./models/ --local-dir-use-symlinks False
```

### Install NVIDIA CUDA Toolkit

This is to enable GPU acceleration.

> See: **How to install CUDA & cuDNN on Ubuntu 22.04**
> - https://gist.github.com/denguir/b21aa66ae7fb1089655dd9de8351a202

```sh
# nvcc --version
sudo apt install nvidia-cuda-toolkit
```

### Install `llama-cpp-python`

This package is Python Bindings for llama.cpp, which enables running LLM locally with both CPU and GPUs.

> See: **README of llama-cpp-python**
> - https://github.com/abetlen/llama-cpp-python/tree/main?tab=readme-ov-file#installation
> 
> See: **OpenAI Compatible Server of llama-cpp-python**
> - https://llama-cpp-python.readthedocs.io/en/latest/server/#installation

```sh
LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python[server]
```

If you have installed `llama-cpp-python` before setup nvcc correctly, you need setup nvcc first, then reinstall `llama-cpp-python`:

```sh
LLAMA_CUBLAS=1 CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python[server] --upgrade --force-reinstall --no-cache-dir
```

## Usage

### Chat via server API - [Recommended]
#### Run server

This will launch a LLM server which supports requests in OpenAI API format.

> See: **OpenAI Compatible Server in llama-cpp-python**
> - https://llama-cpp-python.readthedocs.io/en/latest/server/#running-the-server

```sh
python -m llama_cpp.server --model "./models/dolphin-2.5-mixtral-8x7b.Q5_K_M.gguf" --model_alias "dolphin-2.5-mixtral-8x7b" --n_ctx 16192 --n_gpu_layers 28 --host 0.0.0.0 --port 23333 --interrupt_requests True
```
Go to API docs: `http://<host>:<port>/docs`.

#### Chat via `openai`
> See: [exmaples/chat_with_openai.py](./examples/chat_with_openai.py)

```py
from openai import OpenAI

# If runnning this service with proxy, you might need to unset `http(s)_proxy`.
base_url = "http://127.0.0.1:23333/v1"
api_key = "sk-xxxxx"

client = OpenAI(base_url=base_url, api_key=api_key)
response = client.chat.completions.create(
    model="dolphin-2.5-mixtral-8x7b",
    messages=[
        {
            "role": "user",
            "content": "what is your model",
        }
    ],
    stream=True,
    stop=["[INST]", "[/INST]"],
)

for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
    elif chunk.choices[0].finish_reason == "stop":
        print()
    else:
        pass
```

### Chat completion function in codes - [Not Recommended]
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

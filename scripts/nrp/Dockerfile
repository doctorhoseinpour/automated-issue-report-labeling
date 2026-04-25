FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv git curl ca-certificates tar gzip \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 ships pip/wheel/setuptools as Debian packages that pip refuses
# to uninstall. Side-step the issue with a venv.
RUN python3 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/llm-labler

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

ENV HF_HOME=/workspace/hf_cache \
    TRANSFORMERS_CACHE=/workspace/hf_cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf_cache

CMD ["/bin/bash"]

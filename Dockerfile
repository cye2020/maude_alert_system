FROM apache/airflow:latest-python3.11

USER root

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY lib/ /opt/airflow/lib/

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# torch/torchvision: CUDA 버전 명시 설치
RUN uv pip install --system --no-cache \
    torch torchvision --index-url https://download.pytorch.org/whl/cu128

# cuml: nvidia 인덱스에서 설치
RUN uv pip install --system --no-cache \
    cuml-cu12 --extra-index-url https://pypi.nvidia.com

# 공통 패키지 설치 (vllm 제외)
# --index-strategy unsafe-best-match: 여러 인덱스에서 가장 최신 호환 버전 선택
# (기본 first-match는 PyTorch 인덱스의 구버전 requests를 선택해 충돌 발생)
RUN uv pip install --system --no-cache \
    --index-strategy unsafe-best-match \
    -e /opt/airflow/lib/[all] \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.nvidia.com

# vllm worker 전용 가상환경 (tokenizers 충돌 없이 vllm 설치)
RUN uv venv /opt/vllm-env --python 3.11
RUN uv pip install --no-cache \
    --python /opt/vllm-env/bin/python \
    -e /opt/airflow/lib/[worker] \
    --extra-index-url https://download.pytorch.org/whl/cu128

USER airflow
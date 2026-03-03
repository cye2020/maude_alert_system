FROM apache/airflow:latest-python3.11

USER root

# 1. CUDA 저장소 등록 및 nvcc 설치 (주소를 변수 없이 직접 한 줄로 작성)
RUN apt-get update && apt-get install -y curl gnupg2 gcc g++ build-essential \
    && curl -fL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
        -o cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get install -y cuda-nvcc-12-8 cuda-cudart-dev-12-8 \
    && rm -rf /var/lib/apt/lists/*

# 2. Inductor 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda-12.8
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"


# 3. uv 및 파일 복사
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# lib 폴더 복사 및 pyproject.toml 위치 세팅
WORKDIR /opt/airflow
COPY lib/ /opt/airflow/lib/
# lib 안에 있는 toml을 루트로 복사 (uv가 인덱스 설정을 읽기 위함)
RUN cp /opt/airflow/lib/pyproject.toml /opt/airflow/pyproject.toml

# 4. 패키지 설치 (toml의 cu128 설정을 자동 사용)
RUN uv pip install --system --no-cache --index-strategy unsafe-best-match -e ./lib/[all]

# 5. worker 공용 가상환경(LLM + clustering) 설정
RUN uv venv /opt/vllm-env --python 3.11 \
    && uv pip install --no-cache --python /opt/vllm-env/bin/python --index-strategy unsafe-best-match -e ./lib/[worker]

# 6. 컴파일 캐시 권한 설정
RUN mkdir -p /tmp/torchinductor_airflow && chmod 777 /tmp/torchinductor_airflow
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_airflow

USER airflow

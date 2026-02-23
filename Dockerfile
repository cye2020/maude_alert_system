# Dockerfile
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

# Airflow 설치
RUN python3 -m pip install apache-airflow

# 프로젝트 패키지 설치
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install "transformers[torch]"
RUN python3 -m pip install vllm --extra-index-url https://download.pytorch.org/whl/cu128
RUN python3 -m pip install aws-secretsmanager-caching 

COPY lib/ /opt/airflow/lib/
RUN python3 -m pip install -e /opt/airflow/lib/
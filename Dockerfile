# 공식 Airflow 이미지를 베이스로 사용 (entrypoint 보존)
FROM apache/airflow:latest

USER root

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 프로젝트 라이브러리 복사
COPY lib/ /opt/airflow/lib/

# 패키지 설치 (root에서 실행해야 system site-packages에 쓸 수 있음)
RUN uv pip install --system --no-cache -e /opt/airflow/lib/
RUN uv pip install --system --no-cache -e /opt/airflow/lib/[all]

USER airflow
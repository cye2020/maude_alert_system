#!/usr/bin/env bash
set -euo pipefail

############################
# 기본 설정
############################

ENV_NAME="maude"
PYTHON="python"
LOG_FILE="nohup_help.out"

############################
# 스크립트 위치를 작업 디렉터리로 설정
############################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

############################
# conda 초기화
############################

if ! command -v conda &> /dev/null; then
    echo "Error: conda not found in PATH"
    exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

############################
# conda 환경 생성 (없을 때만)
############################

if ! conda env list | grep -q "^$ENV_NAME "; then
    conda create -n "$ENV_NAME" python=3.10.12 -y
fi

############################
# conda 환경 활성화
############################

conda activate "$ENV_NAME"

############################
# 패키지 설치
############################

if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt not found"
    exit 1
fi

pip install -r requirements.txt

############################
# 전처리 실행 (백그라운드)
############################

nohup "$PYTHON" -m src.preprocess.text_preprocess \
    > "$LOG_FILE" 2>&1 &

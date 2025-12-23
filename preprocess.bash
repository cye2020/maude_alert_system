#!/usr/bin/env bash
set -euo pipefail

############################
# 환경 설정
############################

ENV_NAME=maude
PYTHON=python3

# conda 환경 생성 (1회만)
# conda create -n $ENV_NAME python=3.10.12 -y

# conda 활성화
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate $ENV_NAME

# 패키지 설치 (1회만)
# pip install -r requirements.txt

############################
# 데이터 분할
############################

$PYTHON -m src.preprocess.split_data --weights 3 2 1

############################
# 실행할 파트 선택
############################

PART_IDX=0   # 0=3비중, 1=2비중, 2=1비중

############################
# 선택 파트 전처리 실행
############################

INPUT_PATH="data/temp/part_${PART_IDX}.parquet"
LOG_FILE="nohup_part_${PART_IDX}.out"

nohup $PYTHON -m src.preprocess.text_preprocess \
    -i "$INPUT_PATH" \
    > "$LOG_FILE" 2>&1 &

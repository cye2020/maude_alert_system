#!/bin/bash

# MAUDE & UDI 데이터 다운로드 스크립트
# 사용법: ./download_data.sh

set -e  # 에러 발생 시 스크립트 중단

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 로그 함수
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 프로젝트 루트 디렉토리
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
DATA_DIR="${PROJECT_ROOT}/data"

# 데이터 디렉토리 확인 및 생성
log_info "데이터 디렉토리 확인 중..."
if [ ! -d "${DATA_DIR}" ]; then
    log_warning "데이터 디렉토리가 없습니다. 생성 중..."
    mkdir -p "${DATA_DIR}"
    log_success "데이터 디렉토리 생성 완료: ${DATA_DIR}"
else
    log_info "데이터 디렉토리 존재: ${DATA_DIR}"
fi

# Python 가상환경 활성화 확인 (venv 또는 conda)
if [ -z "${VIRTUAL_ENV}" ] && [ -z "${CONDA_DEFAULT_ENV}" ]; then
    log_warning "가상환경이 활성화되지 않았습니다."
    log_info "가상환경을 활성화하거나 계속 진행하려면 Enter를 누르세요..."
    read -r
else
    if [ -n "${CONDA_DEFAULT_ENV}" ]; then
        log_info "Conda 환경 활성화됨: ${CONDA_DEFAULT_ENV}"
    elif [ -n "${VIRTUAL_ENV}" ]; then
        log_info "가상환경 활성화됨: ${VIRTUAL_ENV}"
    fi
fi

echo ""
echo "=========================================="
echo "  MAUDE & UDI 데이터 다운로드"
echo "=========================================="
echo ""

# 설정
MAUDE_START_YEAR=2023
MAUDE_END_YEAR=2025
MAX_WORKERS=10

# 1. MAUDE 데이터 다운로드
log_info "MAUDE 데이터 다운로드 시작 (${MAUDE_START_YEAR}-${MAUDE_END_YEAR})..."
echo ""

python -m src.download_maude \
    --name event \
    --start ${MAUDE_START_YEAR} \
    --end ${MAUDE_END_YEAR} \
    --output-file "${DATA_DIR}/maude.parquet" \
    --max-workers ${MAX_WORKERS}

if [ $? -eq 0 ]; then
    log_success "MAUDE 데이터 다운로드 완료"
else
    log_error "MAUDE 데이터 다운로드 실패"
    exit 1
fi

echo ""
echo "------------------------------------------"
echo ""

# 2. UDI 데이터 다운로드
log_info "UDI 데이터 다운로드 시작..."
echo ""

python -m src.download_maude \
    --name udi \
    --output-file "${DATA_DIR}/udi.parquet" \
    --max-workers ${MAX_WORKERS}

if [ $? -eq 0 ]; then
    log_success "UDI 데이터 다운로드 완료"
else
    log_error "UDI 데이터 다운로드 실패"
    exit 1
fi

echo ""
echo "=========================================="
echo ""

# 다운로드 완료 후 파일 확인
log_info "다운로드된 파일 확인 중..."
echo ""

if [ -f "${DATA_DIR}/maude.parquet" ]; then
    MAUDE_SIZE=$(du -h "${DATA_DIR}/maude.parquet" | cut -f1)
    log_success "MAUDE 데이터: ${DATA_DIR}/maude.parquet (${MAUDE_SIZE})"
else
    log_error "MAUDE 데이터 파일을 찾을 수 없습니다"
fi

if [ -f "${DATA_DIR}/udi.parquet" ]; then
    UDI_SIZE=$(du -h "${DATA_DIR}/udi.parquet" | cut -f1)
    log_success "UDI 데이터: ${DATA_DIR}/udi.parquet (${UDI_SIZE})"
else
    log_error "UDI 데이터 파일을 찾을 수 없습니다"
fi

echo ""
log_success "모든 데이터 다운로드 완료!"
echo ""

# MAUDE & UDI 데이터 분석 파이프라인

FDA MAUDE(의료기기 부작용 보고) 데이터와 UDI(의료기기 고유 식별) 데이터를 통합 분석하는 end-to-end 파이프라인입니다.

## 📊 프로젝트 개요

### 핵심 기능

- **데이터 수집**: FDA API에서 MAUDE/UDI 데이터 자동 다운로드 및 Parquet 변환
- **데이터 전처리**: Bronze → Silver → Gold 레이어 기반 데이터 품질 관리
- **UDI 매칭**: 불완전한 UDI 데이터를 퍼지 매칭 및 메타데이터 기반으로 보완
- **클러스터링**: 자유 텍스트 형식의 부작용 보고를 자동 유형화
- **통계 분석**: 부작용 급증 탐지(Spike Detection) 및 컬럼 간 통계 검정
- **대시보드**: Streamlit 기반 인터랙티브 분석 대시보드

### 데이터 레이어 구조

```text
Bronze (Raw)          → MAUDE 원본 데이터
  ↓
Silver Stage 1        → 기본 데이터 정제 (NA 패턴 제거, 타입 변환)
  ↓
Silver Stage 2        → 텍스트 전처리 (필드 정규화, UDI 매칭)
  ↓
Silver Stage 3        → 클러스터링 (유사 사례 그룹핑)
  ↓
Gold (Aggregates)     → 비즈니스 집계 (TBD - 동적 집계 방식 검토 중)
```

---

## 🏗️ 프로젝트 구조

```text
Project4/
├── config/                      # 설정 파일 (YAML)
│   ├── base.yaml               # 프로젝트 기본 설정
│   ├── preprocess/             # 전처리 설정
│   │   ├── cleaning.yaml       # NA 패턴, 텍스트 정규화
│   │   ├── udi_matching.yaml   # UDI 매칭 전략
│   │   ├── filtering.yaml      # 데이터 필터링
│   │   └── ...
│   └── dashboard/              # 대시보드 UI 설정
│
├── src/
│   ├── loading/                # 데이터 로딩
│   │   ├── data_loader.py      # FDA API → Parquet 변환
│   │   ├── zip_streamer.py     # ZIP 스트리밍
│   │   ├── flattener.py        # JSON 평탄화
│   │   └── parquet_writer.py   # Parquet 병렬 작성
│   │
│   ├── preprocess/             # 데이터 전처리
│   │   ├── udi_preprocessor.py # UDI 매칭 메인 로직
│   │   ├── config.py           # 설정 로더
│   │   ├── clean.py            # 데이터 클린징
│   │   ├── transforms.py       # 데이터 변환
│   │   ├── udi.py              # UDI 유틸리티
│   │   └── mdr.py              # MDR 텍스트 처리
│   │
│   └── utils/                  # 공통 유틸리티
│       ├── polars/             # Polars 헬퍼
│       ├── visualization/      # 시각화 헬퍼
│       └── chunk.py            # 청크 단위 처리
│
├── dashboard/                  # Streamlit 대시보드
│   ├── Home.py                 # 메인 앱
│   ├── overview_tab.py         # 개요 탭
│   ├── eda_tab.py              # EDA 탭
│   ├── spike_tab.py            # 급증 탐지 탭
│   ├── cluster_tab.py          # 클러스터 분석 탭
│   └── utils/                  # 대시보드 유틸리티
│
├── notebooks/                  # Jupyter 노트북
│   ├── 01_data_overview.ipynb
│   ├── 02_preprocess.ipynb
│   ├── 03_clustering_local_.ipynb
│   ├── 04_statistical_analysis.ipynb
│   └── 05_spike_detection.ipynb
│
├── data/                       # 데이터 디렉토리 (gitignore)
│   ├── bronze/                 # Raw 데이터
│   ├── silver/                 # 전처리된 데이터
│   └── gold/                   # 집계 데이터
│
└── requirements.txt            # Python 패키지
```

---

## 🚀 시작하기

### 1. 환경 설정

```bash
# Python 3.10.12 권장
pip install -r requirements.txt
```

### 2. 데이터 다운로드

```bash
# 스크립트 값 조정 후 실행
bash download_data.sh
```

### 3. 데이터 전처리

```bash
# 전처리 파이프라인 실행
# (notebooks/02_preprocess.ipynb 참고)
```

### 4. 텍스트 전처리 (MDR Text)

```bash
# MDR 텍스트 전처리 실행
bash mdr_text_preprocess.bash
```

### 5. 클러스터링

```bash
# 클러스터링 파이프라인 실행
# (notebooks/03_clustering_local.ipynb 참고)
```

### 6. 대시보드 실행

```bash
streamlit run dashboard/Home.py
```

---

## 🔧 주요 컴포넌트 설명

### 1. 데이터 로딩 ([src/loading](src/loading/))

- **스트리밍 처리**: ZIP 파일을 메모리에 전체 로드하지 않고 스트리밍으로 처리
- **병렬 다운로드**: ProcessPoolExecutor로 여러 파일 동시 다운로드
- **스키마 자동 수집**: 전체 파일을 순회하며 동적 스키마 생성
- **Parquet 변환**: 효율적인 컬럼 기반 저장 포맷

### 2. UDI 매칭 ([src/preprocess/udi_preprocessor.py](src/preprocess/udi_preprocessor.py))

#### 배경

MAUDE 데이터는 **수기 입력**으로 작성되어 데이터 품질이 낮음:

- UDI가 누락되거나 불완전한 경우가 많음
- 제조사명, 제품명 등이 일관성 없이 입력됨

#### 매칭 전략 (4단계)

```text
1. Primary 직접 매칭 (Direct Match)
   - MAUDE의 UDI-DI가 UDI DB의 Primary UDI와 정확히 일치
   - 가장 신뢰도 높음 (Score: 3)
   - Match Type: "direct"

2. Secondary 매칭 (Score 기반)
   - UDI DB의 Secondary Identifier와 매칭
   - Brand, Model Number, Catalog Number 일치도로 점수 계산
   - Score >= 3/2/1 순으로 단계적 매칭
   - Match Type: "secondary"

3. No UDI 매칭 (메타데이터 기반)
   - UDI가 아예 없는 경우
   - 제조사 + 메타데이터(Brand, Model, Catalog)로 매칭
   - Match Type: "meta"

4. 매칭 실패 케이스
   - Secondary 매칭 실패: UDI는 있지만 DB에서 찾지 못함
     → Match Type: "udi_no_match"
   - No UDI 매칭 실패: UDI도 없고 메타데이터로도 찾지 못함
     → Match Type: "no_match"
   - 날짜 필터링 실패: publish_date > report_date로 시간적으로 불가능
     → Match Type: "not_in_mapping"
```

#### Score 계산 (config/preprocess/udi_matching.yaml 참고)

```yaml
score_weights:
  brand: 1
  model_number: 1
  catalog_number: 1

score_levels: [3, 2, 1]  # 단계적 매칭 시도
```

#### 메모리 관리 전략

- **Path 기반 설계**: LazyFrame 대신 Parquet Path를 반환하여 메모리 부담 최소화
- **청크 단위 처리**: 대용량 데이터를 chunk 단위로 처리
- **Temp 파일 관리**: 중간 결과를 temp 파일로 저장 후 최종 병합

### 3. 클러스터링

- **목적**: 자유 텍스트 형식의 부작용 보고를 자동 유형화
- **방법**: (구현 세부사항은 코드 참고)

### 4. Spike Detection ([notebooks/05_spike_detection.ipynb](notebooks/05_spike_detection.ipynb))

- **목적**: 특정 제품/제조사의 부작용 급증 탐지
- **방법**: 시계열 분석 및 통계적 이상치 탐지

### 5. 통계 분석 ([notebooks/04_statistical_analysis.ipynb](notebooks/04_statistical_analysis.ipynb))

- **목적**: 컬럼 간 관계 분석
- **방법**: 통계 검정 (Chi-square, Fisher's exact 등)

---

## 📋 설정 파일 가이드

### [config/base.yaml](config/base.yaml)

- 프로젝트 전체 설정
- 데이터 경로, 로깅, 성능 튜닝

### [config/preprocess/](config/preprocess/)

- **cleaning.yaml**: NA 패턴, 텍스트 정규화
- **udi_matching.yaml**: UDI 매칭 전략 (score, threshold)
- **filtering.yaml**: 데이터 필터링 규칙
- **quality.yaml**: 데이터 품질 검증

### [config/dashboard/](config/dashboard/)

- **defaults.yaml**: 대시보드 기본값
- **sidebar.yaml**: 사이드바 필터 설정
- **ui_standards.yaml**: UI 스타일 가이드

---

## 📈 대시보드 탭 설명

### 1. Overview

- 전체 데이터 개요
- 주요 지표 (총 보고 건수, 제조사 수, 기간 등)

### 2. EDA (Exploratory Data Analysis)

- 제조사/제품/기기별 분포
- 시계열 트렌드
- 결함 유형 분석

### 3. Spike Detection

- 부작용 급증 탐지
- 제조사/제품별 급증 이벤트

### 4. Cluster Analysis

- 유사 사례 그룹 분석
- 클러스터별 특성 시각화

---

## 🛠️ 기술 스택

- **데이터 처리**: Polars, PySpark, Pandas
- **시각화**: Plotly, Seaborn, Matplotlib
- **대시보드**: Streamlit
- **ML/NLP**: Transformers, vLLM, Torch
- **통계 분석**: SciPy, Statsmodels, Pingouin
- **유틸리티**: PyArrow, tqdm, rapidfuzz

---

## 🐛 Known Issues & TODO

### Dashboard

- [ ] TODO: 캐싱 전략 개선 필요 ([dashboard/Home.py:180-186](dashboard/Home.py#L180-L186))
  - 현재 필터 변경 시 모든 캐시 클리어 (너무 aggressive)
  - 개선 방안: 필터별 독립적인 캐시 키 사용

### Data Pipeline

- [ ] Silver Stage 중간 에러 시 롤백 전략 검토
- [ ] Gold Layer 집계 방식 결정 (동적 집계 vs 사전 정의 집계)

### UDI Matching

- [ ] Score weights 튜닝 자동화
- [ ] 매칭 실패 케이스 분석 및 개선

### Code Quality

- [ ] 설정 파일 활용도 개선
  - [config/](config/) 디렉토리에 구조화된 YAML 파일들이 있지만 실제 코드에서 활용이 덜 됨
  - 하드코딩된 값들을 설정 파일로 이관 필요
  - 예: 매직 넘버, 필터링 조건, UI 상수 등
- [ ] 코드 중복 제거 및 리팩토링
  - 여러 파일에 겹치는 함수들 정리 필요
  - 유틸리티 함수 재사용성 향상
  - 공통 로직 추상화

---

## 📝 참고 자료

- [FDA MAUDE Database](https://www.fda.gov/medical-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities/manufacturer-and-user-facility-device-experience-database-maude)
- [FDA openFDA API](https://open.fda.gov/apis/device/event/)
- [UDI Database](https://accessgudid.nlm.nih.gov/)

---

## 📄 라이선스

(라이선스 정보 추가 필요)

---

## 👥 기여자

(기여자 정보 추가 필요)

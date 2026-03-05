# MAUDE Early Alert System

> FDA 의료기기 부작용 데이터(MAUDE)를 수집·정제·분석하여 **이상 신호를 조기에 탐지**하는 end-to-end 데이터 파이프라인
> 

[이미지 표시](https://www.python.org/)

[이미지 표시](https://airflow.apache.org/)

[이미지 표시](https://www.snowflake.com/)

[이미지 표시](https://aws.amazon.com/s3/)

[이미지 표시](https://www.docker.com/)

[이미지 표시](https://streamlit.io/)

[이미지 표시](https://github.com/vllm-project/vllm)

---

## 📌 프로젝트 소개

### 배경 및 목적

FDA MAUDE(Manufacturer and User Facility Device Experience) 데이터베이스에는 매월 수만 건의 의료기기 부작용 보고서가 누적됩니다. 그러나 데이터 대부분이 수기 입력으로 작성되어 UDI 누락, 제조사명 불일치, 비정형 텍스트 등 품질 문제가 심각합니다. 이 프로젝트는 그 데이터를 신뢰할 수 있는 형태로 정제하고, 이상 신호를 자동으로 탐지하는 end-to-end 파이프라인을 구축하는 것을 목표로 합니다.

### 무엇을 만들었는가

FDA API에서 데이터를 수집하는 것부터 분석 대시보드에 결과를 표시하기까지의 전 과정을 자동화한 시스템입니다. 핵심 기능은 다음과 같습니다.

**데이터 수집 및 적재**: FDA openFDA API에서 MAUDE / UDI 데이터를 매월 자동 수집하여 AWS S3에 적재하고, Snowflake External Stage를 통해 BRONZE 레이어로 로드합니다.

**데이터 품질 관리 (Silver 14단계 파이프라인)**: 중복 제거, 결측값 처리, 텍스트 정규화, 고위험 기기 스코핑(Class 3), UDI 매칭, 품질 필터링, SCD2 이력 관리까지 14단계 전처리를 Snowflake SQL로 처리합니다. event 카테고리의 전처리는 UDI 데이터와의 동기화 시점을 기준으로 `pre_sync` / `post_sync`로 분리되어, UDI 테이블 준비 전·후 단계가 병렬로 안전하게 실행됩니다.

**UDI 매칭**: MAUDE 데이터에 UDI가 누락되거나 불완전한 경우, Direct → Secondary(Score 기반) → 메타데이터(제조사+모델명+카탈로그) 순의 3단계 폴백 전략으로 매칭률을 최대화합니다.

**LLM 기반 MDR 텍스트 구조화**: 자유 텍스트 형식의 부작용 보고서(MDR Text)를 vLLM + Pydantic 스키마로 구조화합니다. 환자 위해도(사망/중증/경증/무해), 결함 유형, 결함 확인 여부, 문제 컴포넌트를 자동 추출합니다.

**클러스터링**: LLM 추출 결과를 Sentence-Transformers로 임베딩하고, HDBSCAN(GPU 가속)으로 유사 사례를 자동 그룹핑합니다. 사전 학습된 모델로 추론 전용 실행하여 일관된 클러스터 레이블을 유지하며, 결과는 Snowflake SILVER에 적재됩니다. LLM·클러스터링은 공용 GPU venv(`/opt/vllm-env`)에서 `task.external_python`으로 격리 실행됩니다.

**이상 신호 탐지**: Gold 레이어에서 시계열 Spike Detection과 통계 검정(Fisher's Exact Test, Chi-square)을 실행하여 특정 제조사·제품의 부작용 급증을 자동 탐지합니다.

**분석 대시보드**: Streamlit 기반 인터랙티브 대시보드에서 위 분석 결과를 Overview / EDA / Spike / Cluster 탭으로 시각화합니다.

### 기존 버전 대비 변경사항

| 구분 | 기존 (로컬 파이프라인) | 현재 (클라우드 파이프라인) |
| --- | --- | --- |
| 실행 방식 | 노트북/스크립트 수동 실행 | Airflow 3 Asset 기반 자동 트리거 |
| 데이터 저장소 | 로컬 Parquet 파일 | AWS S3 + Snowflake (BRONZE/SILVER/GOLD) |
| 전처리 엔진 | Polars (로컬 메모리) | Snowflake SQL (서버사이드 처리) |
| MDR 텍스트 처리 | 규칙 기반 정규화 | vLLM 배치 추출 + Pydantic 스키마 |
| 클러스터링 환경 | 로컬 CPU | LLM·클러스터링 공용 GPU venv 격리 (`task.external_python`) |
| 시크릿 관리 | 환경변수 직접 관리 | AWS Secrets Manager 자동 주입 |

---

## 🏗️ 아키텍처 / 데이터 플로우

### 시스템 아키텍처

![시스템아키텍처.png](assets/images/시스템아키텍처.png)

### DAG별 역할 및 트리거 방식

| DAG | 스케줄 | 역할 | 트리거 Asset |
| --- | --- | --- | --- |
| `maude_ingest` | 매월 1회 | FDA API → AWS S3 적재 | - |
| `maude_bronze` | Asset 트리거 | S3 → Snowflake BRONZE 로드 | `s3://amazon-s3-fda` |
| `maude_silver` | Asset 트리거 | BRONZE → SILVER 전처리 (SCD2) | `snowflake://MAUDE/BRONZE/*` |
| `maude_llm` | Asset 트리거 | MDR 텍스트 LLM 구조화 추출 | `snowflake://MAUDE/SILVER/*` |
| `maude_cluster` | Asset 트리거 | LLM 결과 → 클러스터링 | `snowflake://MAUDE/SILVER/*_LLM_EXTRACTED` |
| `maude_gold` | Asset 트리거 | Gold 집계 + 이상 신호 탐지 | `snowflake://MAUDE/SILVER/*_CLUSTERED` |

> **Asset 기반 트리거**: Airflow 3의 Data-Aware Scheduling을 활용하여 DAG 간 직접 의존성 없이 데이터 자산 업데이트 시 자동 트리거됩니다.
> 

### Snowflake 레이어 구조

```
MAUDE.BRONZE    ← S3 External Stage에서 COPY INTO로 원본 적재
     ↓
MAUDE.SILVER    ← 전처리(SCD2) + UDI 매칭 + LLM 추출 결과 + 클러스터링 결과
     ↓
MAUDE.GOLD      ← 대시보드용 집계 + Spike/Statistical 이상 탐지
```

### LLM 파이프라인 (maude_llm DAG)

MDR 텍스트를 5,000건 단위 청크로 분할하여 GPU venv에서 병렬 처리합니다.

```
extract_records          # Snowflake MDR 텍스트 추출 → /tmp 청크 파일
      ↓
llm_chunk_group          # task.external_python: vLLM 배치 추출
  ├─ llm_extract_chunk   # vllm-env에서 실행 (GPU)
  └─ load_chunk_results  # 결과 Snowflake 적재
      ↓
fetch_failures           # 추출 실패 레코드 조회 → 재처리
      ↓
failure_chunk_group      # 실패 케이스 재시도
      ↓
join_extraction          # 최종 JOIN → {category}_LLM_EXTRACTED
      ↓
cleanup_checkpoint       # 임시 파일 정리 (always_done)
```

**추출 항목**: `patient_harm` / `defect_type` / `defect_confirmed` / `problem_components` (Pydantic 스키마 기반)

---

## 🛠️ 기술 스택

| 구분 | 기술 |
| --- | --- |
| **오케스트레이션** | Apache Airflow 3.1.7 (CeleryExecutor), Docker Compose |
| **데이터 웨어하우스** | Snowflake (BRONZE / SILVER / GOLD 레이어) |
| **클라우드 스토리지** | AWS S3 (`ap-northeast-2`) |
| **LLM 추론** | vLLM 0.15.1, Sentence-Transformers, OpenAI API |
| **클러스터링** | HDBSCAN, cuML (GPU 가속), RAPIDS |
| **패키지 관리** | uv (pyproject.toml), 멀티 venv 전략 |
| **데이터 처리** | Pandas, Polars, PyArrow |
| **통계 분석** | SciPy, Statsmodels |
| **시각화/대시보드** | Streamlit, Plotly |
| **시크릿 관리** | AWS Secrets Manager (Airflow SecretsManagerBackend) |
| **로깅** | structlog (contextvars 기반 DAG/run_id 자동 바인딩) |

---

## 📁 폴더 구조

```
maude_alert_system/
├── airflow/
│   ├── dags/
│   │   ├── ingest_dag.py       # FDA API → S3
│   │   ├── bronze_dag.py       # S3 → Snowflake BRONZE
│   │   ├── silver_dag.py       # BRONZE → SILVER (SCD2)
│   │   ├── llm_dag.py          # MDR 텍스트 LLM 구조화
│   │   ├── cluster_dag.py      # HDBSCAN 클러스터링
│   │   └── gold_dag.py         # Gold 집계 + 이상 탐지
│   ├── logs/
│   └── plugins/
│
├── lib/                        # Airflow 환경 패키지 (pyproject.toml)
│   ├── pyproject.toml          # uv 기반 의존성 (airflow/snowflake/worker 등)
│   └── src/maude_early_alert/
│       ├── pipelines/          # DAG에서 호출하는 파이프라인 클래스
│       │   ├── ingest.py
│       │   ├── bronze.py
│       │   ├── silver.py
│       │   ├── llm_pipeline.py
│       │   ├── cluster_pipeline.py
│       │   └── gold.py
│       ├── preprocessors/      # LLM 프롬프트 / Pydantic 스키마
│       │   ├── prompt.py
│       │   └── mdr_extractor.py
│       ├── assets.py           # Airflow Asset 정의 (DAG 간 의존성)
│       └── utils/
│           ├── config_loader.py  # YAML 설정 로더 (환경변수 치환)
│           └── secrets.py        # AWS Secrets Manager 연동
│
├── config/
│   ├── airflow.cfg             # Airflow 설정 (SecretsManagerBackend 포함)
│   ├── base.yaml               # 프로젝트 기본 설정
│   ├── storage.yaml            # S3 / Snowflake 경로 설정
│   ├── pipeline.yaml           # 파이프라인 단계 정의
│   └── preprocess/             # 전처리 세부 설정
│       ├── cleaning.yaml
│       ├── udi_matching.yaml
│       └── llm_extraction.yaml
│
├── dashboard/                  # 대시보드 환경 (requirements.txt)
│   ├── Home.py
│   ├── overview_tab.py
│   ├── eda_tab.py
│   ├── spike_tab.py
│   └── cluster_tab.py
│
├── scripts/
│   └── create_vllm_env.sh      # GPU venv 생성 스크립트
│
├── Dockerfile                  # Airflow 커스텀 이미지 (CUDA 12.8 포함)
├── docker-compose.yaml         # Airflow 전체 스택
├── .env.template               # Airflow 환경변수 템플릿
├── .envrc.template             # direnv 환경변수 템플릿
└── requirements.txt            # Dashboard 환경 의존성
```

---

## 🚀 설치 및 실행

이 프로젝트는 **두 개의 독립적인 실행 환경**으로 구성됩니다.

### 환경 1: Airflow 오케스트레이션 환경

Airflow + Snowflake + AWS 파이프라인 실행 환경입니다. Docker Compose와 uv 기반으로 구성됩니다.

### 사전 요구사항

- Docker & Docker Compose
- NVIDIA GPU 드라이버 (CUDA 12.8 호환)
- AWS 계정 및 S3 버킷 설정 (별도 설정 필요)
- Snowflake 계정 설정 (별도 설정 필요)

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/cye2020/maude_alert_system.git
cd maude_alert_system

# 2. direnv 설정 (권장)
cp .envrc.template .envrc
# .envrc 파일에서 AWS 키, AIRFLOW_UID 등 설정
direnv allow

# 또는 .env 파일 직접 사용
cp .env.template .env
# .env 파일 편집

# 3. Airflow 이미지 빌드 및 실행
docker compose up --build -d

# 4. 초기화 (최초 1회)
docker compose run --rm airflow-init
```

### Airflow 접속

`http://localhost:8080`

### GPU venv 수동 생성 (필요 시)

```bash
# Docker 컨테이너 내부에서 실행 시 자동 생성됨
# 로컬 개발 환경에서 별도 생성 필요 시:
bash scripts/create_vllm_env.sh
```

---

### 환경 2: Dashboard 시각화 환경

Streamlit 대시보드 전용 환경입니다. Snowflake 또는 로컬 Parquet 파일에서 데이터를 읽습니다.

### 설치

```bash
# Python 3.11 권장
pip install -r requirements.txt
```

### 실행

```bash
streamlit run dashboard/Home.py
```

---

## 🔑 환경변수 목록

### `.env` — Airflow Docker Compose 환경변수

| 변수명 | 설명 | 예시 |
| --- | --- | --- |
| `AIRFLOW_UID` | Airflow 컨테이너 실행 유저 ID | `50000` |
| `AIRFLOW_PROJ_DIR` | 볼륨 마운트 기준 경로 | `/home/user/maude_alert_system` |
| `AWS_DEFAULT_REGION` | AWS 리전 | `ap-northeast-2` |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 | - |
| `AIRFLOW_CONN_AWS_DEFAULT` | Airflow AWS 커넥션 URI | `aws://...` |

### `.envrc` — direnv 로컬 개발 환경변수

| 변수명 | 설명 |
| --- | --- |
| `AWS_DEFAULT_REGION` | AWS 리전 |
| `AWS_ACCESS_KEY_ID` | AWS 액세스 키 |
| `AWS_SECRET_ACCESS_KEY` | AWS 시크릿 키 |
| `AIRFLOW_CONN_AWS_DEFAULT` | Airflow AWS 커넥션 URI |

> **Snowflake 연결 정보**는 AWS Secrets Manager (`snowflake/de`)에서 관리되며 Airflow SecretsManagerBackend를 통해 자동으로 주입됩니다. `.env` / `.envrc`에 직접 기재하지 않습니다.
> 

---

## 📊 대시보드

Streamlit 기반 인터랙티브 대시보드로, Snowflake Gold 레이어 또는 로컬 Parquet 파일을 데이터 소스로 사용합니다.

| 탭 | 내용 |
| --- | --- |
| **Overview** | 총 보고 건수, 제조사 수, 기간 등 주요 지표 |
| **EDA** | 제조사/제품/기기 분포, 시계열 트렌드, 결함 유형 분석 |
| **Spike Detection** | 부작용 급증 탐지 (제조사/제품별 이상 이벤트) |
| **Cluster Analysis** | LLM 추출 + HDBSCAN 클러스터 특성 시각화 |

---

## 🐛 Known Issues & TODO

- [ ]  Dashboard 캐싱 전략 개선: 필터 변경 시 전체 캐시 클리어 → 필터별 독립 캐시 키로 개선 필요
- [ ]  Silver DAG 중간 실패 시 롤백 전략 구체화
- [ ]  UDI 매칭 Score weights 자동 튜닝
- [ ]  config/ 설정 파일 활용도 개선 (하드코딩된 값 이관)

---

## 📝 참고 자료

- [FDA MAUDE Database](https://www.fda.gov/medical-devices/mandatory-reporting-requirements-manufacturers-importers-and-device-user-facilities/manufacturer-and-user-facility-device-experience-database-maude)
- [FDA openFDA API](https://open.fda.gov/apis/device/event/)
- [UDI Database (GUDID)](https://accessgudid.nlm.nih.gov/)
- [Apache Airflow 3 - Data-Aware Scheduling](https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/datasets.html)

---

## 📄 라이선스

This project uses public data provided by openFDA (https://open.fda.gov). The data is available under the Creative Commons CC0 1.0 Universal license.
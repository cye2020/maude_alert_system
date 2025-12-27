# 용어 통일 시스템 - 완료 요약

## ✅ 완료된 작업

### 1. 중앙 집중식 용어 사전 구축

#### 📄 생성된 파일

```
config/dashboard/
├── terminology.yaml              # 🆕 용어 사전 (모든 용어 정의)
├── TERMINOLOGY_GUIDE.md         # 🆕 상세 사용 가이드
└── README.md                    # 🆕 전체 Config 설명

dashboard/utils/
├── terminology.py               # 🆕 TerminologyManager 클래스
└── constants.py                 # ✏️ Terms 클래스 추가

프로젝트 루트/
├── MIGRATION_GUIDE.md          # 🆕 마이그레이션 가이드
├── apply_terminology.py        # 🆕 하드코딩 검사 도구
└── TERMINOLOGY_SUMMARY.md      # 이 문서
```

---

## 🎯 핵심 개선사항

### Before (문제점)
```python
# 😱 코드 곳곳에 하드코딩된 한글
st.metric("치명률", ...)              # 파일 A
st.metric("사망률", ...)              # 파일 B
st.metric("치명률(CFR)", ...)         # 파일 C
st.subheader("결함 유형별 상위 문제 & 사건 유형 분포")  # 너무 더티!

# 문제:
# - 용어 혼용 (치명률 vs 사망률)
# - 표현 불일치 (치명률 vs 치명률(CFR))
# - 수정 시 모든 파일 찾아야 함
# - 오타 위험
```

### After (해결)
```python
from dashboard.utils.constants import Terms

# ✅ 모든 코드에서 일관된 용어 사용
st.metric(Terms.KOREAN.CFR, ...)
st.metric(Terms.KOREAN.DEATH_RATE, ...)
st.metric(Terms.KOREAN.CFR_FULL, ...)
st.subheader(Terms.KOREAN.DEFECT_TYPE_ANALYSIS)

# 장점:
# ✅ 용어 통일 (terminology.yaml 한 곳에서 관리)
# ✅ 표현 일관성 (같은 지표 = 같은 이름)
# ✅ 수정 용이 (yaml만 수정하면 전체 반영)
# ✅ 오타 방지 (IDE 자동완성)
# ✅ 가독성 향상
```

---

## 📚 용어 사전 구조

### terminology.yaml 주요 섹션

```yaml
korean_terms:
  metrics:           # 지표 (CFR, 사망률, 보고 건수 등)
  entities:          # 엔티티 (제조사, 제품군, 결함 유형 등)
  time:             # 시간 (시계열, 추이, 월별 등)
  analysis:         # 분석 (급증, 증가, 분포 등)
  sections:         # 섹션 제목 (결함 유형 분석, 환자 피해 분포 등)
  quality:          # 품질 (정보 없음, 알 수 없음 등)
  severity:         # 심각도 (심각, 경고, 주의 등)

column_names:       # DB 컬럼명 매핑
column_headers:     # DataFrame 표시용 헤더
message_templates:  # 메시지 템플릿
term_descriptions:  # 용어 설명 (툴팁용)
display_formats:    # 숫자/날짜 포맷
```

---

## 🚀 사용법

### 1. 기본 사용

```python
from dashboard.utils.constants import Terms

# 메트릭
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")
st.metric(Terms.KOREAN.SEVERE_HARM, f"{severe:,}건")

# 제목
st.subheader(f"📊 {Terms.KOREAN.DEFECT_TYPE_ANALYSIS}")
st.subheader(f"🔍 {Terms.KOREAN.RISK_MATRIX}")

# 차트 라벨
fig.add_trace(go.Bar(name=Terms.KOREAN.REPORT_COUNT, ...))
```

### 2. 템플릿 사용

```python
# 섹션 제목 템플릿
title = Terms.section_title('entity_analysis',
                            entity=Terms.KOREAN.DEFECT_TYPE)
# 결과: "결함 유형 분석"

title = Terms.section_title('metric_by_entity',
                            entity=Terms.KOREAN.MANUFACTURER,
                            metric=Terms.KOREAN.CFR)
# 결과: "제조사별 치명률"

# 메시지 템플릿
msg = Terms.format_message('high_cfr_alert',
                           device='ABC Corp',
                           cfr=12.5,
                           count=100)
# 결과: "⚠️ **ABC Corp**의 치명률이 **12.50%**로 매우 높습니다 (중대 피해 100건)"
```

### 3. DataFrame 컬럼 변환

```python
from dashboard.utils.terminology import get_term_manager

term = get_term_manager()

# 전체 컬럼 한글 변환
display_df = df.rename(columns=term.column_headers)

# 또는 개별 변환
display_df = df.rename(columns={
    'death_count': Terms.get_column_header('death_count'),
    'cfr': Terms.get_column_header('cfr')
})
```

---

## 📊 주요 용어 목록

### 핵심 지표

| 용어 | Terms 사용 | 설명 |
|------|-----------|------|
| 치명률 | `Terms.KOREAN.CFR` | Case Fatality Rate (사망+중증부상) |
| 치명률(CFR) | `Terms.KOREAN.CFR_FULL` | 괄호 포함 버전 |
| 사망률 | `Terms.KOREAN.DEATH_RATE` | 사망만 |
| 사망 | `Terms.KOREAN.DEATH_COUNT` | 사망 건수 |
| 중대 피해 | `Terms.KOREAN.SEVERE_HARM` | 사망 + 중증 부상 |
| 중증 부상 | `Terms.KOREAN.SERIOUS_INJURY` | 중증 부상 |
| 경증 부상 | `Terms.KOREAN.MINOR_INJURY` | 경증 부상 |
| 부상 없음 | `Terms.KOREAN.NO_HARM` | 부상 없음 |

### 엔티티

| 용어 | Terms 사용 |
|------|-----------|
| 제조사 | `Terms.KOREAN.MANUFACTURER` |
| 제품군 | `Terms.KOREAN.PRODUCT` |
| 기기 | `Terms.KOREAN.DEVICE` |
| 결함 유형 | `Terms.KOREAN.DEFECT_TYPE` |
| 문제 부품 | `Terms.KOREAN.COMPONENT` |
| 클러스터 | `Terms.KOREAN.CLUSTER` |

### 패턴/분석

| 용어 | Terms 사용 |
|------|-----------|
| 급증 | `Terms.KOREAN.SPIKE` |
| 증가 | `Terms.KOREAN.INCREASE` |
| 감소 | `Terms.KOREAN.DECREASE` |
| 시계열 | `Terms.KOREAN.TIME_SERIES` |
| 추이 | `Terms.KOREAN.TREND` |
| 월별 | `Terms.KOREAN.MONTHLY` |
| 분포 | `Terms.KOREAN.DISTRIBUTION` |

### 섹션 제목

| 용어 | Terms 사용 |
|------|-----------|
| 결함 유형 분석 | `Terms.KOREAN.DEFECT_TYPE_ANALYSIS` |
| 문제 부품 분석 | `Terms.KOREAN.COMPONENT_ANALYSIS` |
| 환자 피해 분포 | `Terms.KOREAN.HARM_DISTRIBUTION` |
| 사건 유형 분포 | `Terms.KOREAN.EVENT_TYPE_DISTRIBUTION` |
| 치명률(CFR) 분석 | `Terms.KOREAN.CFR_ANALYSIS` |
| 리스크 매트릭스 | `Terms.KOREAN.RISK_MATRIX` |

---

## 🔧 용어 추가/수정 방법

### 1. 용어 추가

```yaml
# 1. terminology.yaml에 추가
korean_terms:
  metrics:
    new_metric: '새로운 지표'
```

```python
# 2. constants.py에 추가
class Terms:
    class KOREAN:
        NEW_METRIC = _term.get('korean_terms.metrics.new_metric', '새로운 지표')
```

```python
# 3. 사용
st.metric(Terms.KOREAN.NEW_METRIC, value)
```

### 2. 용어 수정

```yaml
# terminology.yaml만 수정하면 전체 반영!
korean_terms:
  metrics:
    cfr: '치명률'  # -> '위험도'로 변경하면 모든 코드에 자동 반영
```

---

## ✅ 적용 완료 현황

### 수정된 파일

- ✅ `dashboard/overview_tab.py` - 메트릭, 차트 제목
- ✅ `dashboard/cluster_tab.py` - 메트릭 (사망, 중증 부상, 경증 부상)
- ✅ `dashboard/eda_tab.py` - 메트릭 (사망, 중증 부상, 경증 부상, 부상 없음)

### 검증 완료

```bash
$ python apply_terminology.py
✅ 하드코딩된 문자열이 없습니다!
```

---

## 📖 참고 문서

| 문서 | 설명 |
|------|------|
| [terminology.yaml](config/dashboard/terminology.yaml) | 모든 용어 정의 원본 |
| [TERMINOLOGY_GUIDE.md](config/dashboard/TERMINOLOGY_GUIDE.md) | 상세 사용 가이드 |
| [README.md](config/dashboard/README.md) | Config 전체 설명 |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | 마이그레이션 가이드 |
| [constants.py](dashboard/utils/constants.py) | Terms 클래스 구현 |
| [terminology.py](dashboard/utils/terminology.py) | TerminologyManager 구현 |

---

## 🎉 성과

1. **용어 통일** ✅
   - CFR vs 치명률 vs 사망률 혼동 해결
   - 모든 지표/엔티티 명칭 일관성 확보

2. **중앙 관리** ✅
   - terminology.yaml 한 곳에서 모든 용어 관리
   - 변경 시 전체 반영 자동화

3. **개발자 경험 개선** ✅
   - IDE 자동완성으로 오타 방지
   - 명확한 API (`Terms.KOREAN.CFR`)
   - 가독성 향상

4. **유지보수성** ✅
   - 새 용어 추가 쉬움
   - 용어 변경 안전
   - 일관성 보장

5. **문서화** ✅
   - 상세한 가이드 문서
   - 실전 예시 제공
   - 마이그레이션 도구 제공

---

## 🚀 다음 단계

### 추천 작업

1. **기존 코드 점진적 마이그레이션**
   - 새 기능 개발 시 Terms 사용
   - 기존 코드 수정 시 함께 변경

2. **팀 공유**
   - TERMINOLOGY_GUIDE.md 팀원과 공유
   - 코딩 컨벤션에 추가

3. **확장**
   - 필요한 용어 지속 추가
   - 메시지 템플릿 활용 확대

---

## 💡 Tips

1. **자동완성 활용**: `Terms.KOREAN.`까지 타이핑하면 사용 가능한 모든 용어 표시
2. **검색 쉬워짐**: `Terms.KOREAN.CFR`로 검색하면 사용처 한 번에 찾기
3. **리팩토링 안전**: 용어 변경 시 한 곳만 수정하면 됨
4. **신규 팀원**: TERMINOLOGY_GUIDE.md 읽으면 바로 이해

---

**모든 용어가 이제 한 곳에서 관리됩니다! 🎉**

---

## 📞 문의

- 용어 추가/수정 필요 시 → `terminology.yaml` 수정 후 PR
- 사용법 질문 → [TERMINOLOGY_GUIDE.md](config/dashboard/TERMINOLOGY_GUIDE.md) 참고
- 버그 발견 시 → Issue 등록

---

Made with ❤️ for consistent and clean codebase!

# 용어 통일 가이드 (Terminology Guide)

## 📌 개요

대시보드 전체에서 일관된 용어를 사용하기 위한 중앙 집중식 용어 사전입니다.

### 문제점 (Before)
```python
# 코드 곳곳에 하드코딩된 한글
st.metric("치명률", f"{cfr:.2f}%")      # 여기는 "치명률"
st.metric("사망률", f"{rate:.2f}%")    # 여기는 "사망률"
st.metric("치명률(CFR)", ...)           # 여기는 "치명률(CFR)"
st.subheader("기기별 치명률 분석")      # 또 다른 표현

# 같은 의미인데 표현이 달라서 혼란 발생!
```

### 해결 (After)
```python
from dashboard.utils.constants import Terms

# 모든 코드에서 일관된 용어 사용
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")           # "치명률"
st.metric(Terms.KOREAN.DEATH_RATE, f"{rate:.2f}%")  # "사망률"
st.subheader(f"{Terms.KOREAN.DEVICE}별 {Terms.KOREAN.CFR_FULL} 분석")

# 한 곳(terminology.yaml)에서 관리하므로 변경 시 전체 반영 가능!
```

---

## 📂 파일 구조

```
config/dashboard/
  ├── terminology.yaml          # 용어 사전 (모든 용어 정의)
  └── TERMINOLOGY_GUIDE.md      # 이 문서

dashboard/utils/
  ├── terminology.py            # TerminologyManager 클래스
  └── constants.py              # Terms 클래스 (상수와 통합)
```

---

## 🎯 사용 방법

### 1. 기본 사용 (권장)

```python
from dashboard.utils.constants import Terms

# 한국어 용어
cfr_label = Terms.KOREAN.CFR                    # "치명률"
death_rate_label = Terms.KOREAN.DEATH_RATE     # "사망률"
manufacturer = Terms.KOREAN.MANUFACTURER       # "제조사"
spike = Terms.KOREAN.SPIKE                     # "급증"

# 메트릭에 사용
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")

# 차트 제목
st.subheader(f"{Terms.KOREAN.DEVICE}별 {Terms.KOREAN.CFR_FULL} 분석")
```

### 2. 컬럼명 사용

```python
# 계산된 컬럼명 가져오기
death_col = Terms.COLUMN.DEATH_COUNT           # 'death_count'
cfr_col = Terms.COLUMN.CFR                     # 'cfr'

# DataFrame에서 사용
df = df.with_columns([
    pl.col('event_type').filter(pl.col('event_type') == 'Death')
      .count().alias(Terms.COLUMN.DEATH_COUNT)
])
```

### 3. DataFrame 컬럼 헤더 변환

```python
# Pandas DataFrame 컬럼명 한글로 변환
display_df = df.rename(columns={
    'death_count': Terms.get_column_header('death_count'),      # '사망'
    'cfr': Terms.get_column_header('cfr'),                      # '치명률(%)'
    'total_count': Terms.get_column_header('total_count')       # '전체 건수'
})

# 또는 전체 매핑 사용
from dashboard.utils.terminology import get_term_manager
term = get_term_manager()
display_df = df.rename(columns=term.column_headers)
```

### 4. 메시지 템플릿

```python
# 고위험 CFR 경고 메시지
msg = Terms.format_message(
    'high_cfr_alert',
    device='ABC Corp - XYZ Device',
    cfr=12.5,
    count=100
)
st.error(msg)
# 출력: "⚠️ **ABC Corp - XYZ Device**의 치명률이 **12.50%**로 매우 높습니다 (중대 피해 100건)"

# 낮은 CFR 정보
msg = Terms.format_message('low_cfr_info', cfr=0.8)
st.success(msg)
# 출력: "✅ 평균 치명률이 **0.80%**로 양호한 수준입니다"
```

### 5. 용어 설명 (툴팁)

```python
# CFR 설명 가져오기
cfr_description = Terms.get_description('cfr')

with st.expander("ℹ️ 치명률(CFR)이란?"):
    st.markdown(cfr_description)
```

---

## 📖 주요 용어 목록

### 핵심 지표

| 영문 키 | 한글 | 사용법 |
|---------|------|--------|
| cfr | 치명률 | `Terms.KOREAN.CFR` |
| cfr_full | 치명률(CFR) | `Terms.KOREAN.CFR_FULL` |
| death_rate | 사망률 | `Terms.KOREAN.DEATH_RATE` |
| death_count | 사망 | `Terms.KOREAN.DEATH_COUNT` |
| severe_harm | 중대 피해 | `Terms.KOREAN.SEVERE_HARM` |
| serious_injury | 중증 부상 | `Terms.KOREAN.SERIOUS_INJURY` |

**중요:**
- **치명률(CFR)** = (사망 + 중증 부상) / 전체 건수 × 100
- **사망률** = 사망 / 전체 건수 × 100
- 두 용어를 명확히 구분해서 사용!

### 엔티티

| 영문 키 | 한글 | 사용법 |
|---------|------|--------|
| manufacturer | 제조사 | `Terms.KOREAN.MANUFACTURER` |
| product | 제품군 | `Terms.KOREAN.PRODUCT` |
| device | 기기 | `Terms.KOREAN.DEVICE` |
| defect_type | 결함 유형 | `Terms.KOREAN.DEFECT_TYPE` |
| component | 부품 | `Terms.KOREAN.COMPONENT` |
| cluster | 클러스터 | `Terms.KOREAN.CLUSTER` |

### 패턴/분석

| 영문 키 | 한글 | 사용법 |
|---------|------|--------|
| spike | 급증 | `Terms.KOREAN.SPIKE` |
| increase | 증가 | `Terms.KOREAN.INCREASE` |
| decrease | 감소 | `Terms.KOREAN.DECREASE` |

---

## 🔧 용어 수정 방법

### 용어 변경

`config/dashboard/terminology.yaml` 파일을 수정하면 **전체 대시보드에 즉시 반영**됩니다.

```yaml
# terminology.yaml
korean_terms:
  metrics:
    cfr: '치명률'           # 이 값을 변경하면
    death_rate: '사망률'    # 모든 코드에서 자동 반영
```

### 새로운 용어 추가

```yaml
# 1. terminology.yaml에 추가
korean_terms:
  metrics:
    new_metric: '새로운 지표'

# 2. constants.py의 Terms 클래스에 추가
class Terms:
    class KOREAN:
        NEW_METRIC = _term.get('korean_terms.metrics.new_metric', '새로운 지표')

# 3. 코드에서 사용
st.metric(Terms.KOREAN.NEW_METRIC, value)
```

---

## 📋 메시지 템플릿 목록

| 템플릿 키 | 설명 | 사용 예시 |
|-----------|------|-----------|
| `high_cfr_alert` | 고위험 CFR 경고 | `Terms.format_message('high_cfr_alert', device=..., cfr=..., count=...)` |
| `low_cfr_info` | 낮은 CFR 정보 | `Terms.format_message('low_cfr_info', cfr=...)` |
| `spike_detected` | 급증 탐지 | `Terms.format_message('spike_detected', entity=..., period=..., count=..., new_count=...)` |
| `cluster_high_risk` | 클러스터 고위험 | `Terms.format_message('cluster_high_risk', cluster_id=..., cfr=..., count=...)` |
| `cluster_low_risk` | 클러스터 저위험 | `Terms.format_message('cluster_low_risk', cluster_id=..., cfr=...)` |
| `no_data` | 데이터 없음 | `Terms.format_message('no_data')` |
| `loading` | 로딩 중 | `Terms.format_message('loading')` |

### 새 템플릿 추가 방법

```yaml
# terminology.yaml
message_templates:
  my_custom_message: '⚠️ {device}에서 {pattern}이 감지되었습니다 ({count}건)'
```

```python
# 사용
msg = Terms.format_message('my_custom_message',
                           device='ABC Corp',
                           pattern='급증',
                           count=50)
```

---

## 🎨 실제 적용 예시

### Before (문제)

```python
# eda_tab.py
st.metric("치명률", f"{cfr:.2f}%")

# cluster_tab.py
st.metric("치명률 (CFR)", f"{cfr:.2f}%")

# overview_tab.py
st.metric("사망률", f"{cfr:.2f}%")  # 실제로는 CFR인데 사망률로 잘못 표시!

# 같은 지표인데 표현이 다르고, 심지어 잘못된 용어 사용!
```

### After (해결)

```python
from dashboard.utils.constants import Terms

# eda_tab.py
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")

# cluster_tab.py
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")

# overview_tab.py
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")

# 모든 탭에서 일관되게 "치명률" 사용!
# terminology.yaml에서 한 번만 수정하면 전체 반영!
```

---

## ✅ 체크리스트

새로운 코드를 작성할 때 확인하세요:

- [ ] 하드코딩된 한글 용어 대신 `Terms.KOREAN.*` 사용
- [ ] 컬럼명은 `Terms.COLUMN.*` 또는 `ColumnNames.*` 사용
- [ ] 반복되는 메시지는 템플릿으로 만들기
- [ ] DataFrame 표시 시 `Terms.get_column_header()` 또는 `term.column_headers` 사용
- [ ] 새로운 용어는 `terminology.yaml`에 먼저 추가

---

## 🔍 자주 묻는 질문 (FAQ)

### Q1. 기존 하드코딩된 문자열을 모두 바꿔야 하나요?

**A:** 점진적으로 개선하면 됩니다. 새로운 코드부터 `Terms`를 사용하고, 기존 코드는 수정이 필요할 때 함께 변경하세요.

### Q2. 성능에 영향이 있나요?

**A:** 거의 없습니다. `TerminologyManager`는 싱글톤 패턴으로 한 번만 로드되고, 이후는 메모리에서 빠르게 접근합니다.

### Q3. 영문 버전도 지원하나요?

**A:** `terminology.yaml`에 `english_terms` 섹션이 있습니다. 필요 시 확장 가능합니다.

```python
# 향후 다국어 지원 시
english_label = term.english.metrics.cfr  # 'CFR'
```

### Q4. 컬럼명과 표시명을 헷갈려요.

**A:**
- **컬럼명** (`Terms.COLUMN.*`): DataFrame 내부에서 사용하는 실제 컬럼명 (예: `'death_count'`)
- **표시명** (`Terms.KOREAN.*`): 사용자에게 보여주는 한글 이름 (예: `'사망'`)

```python
# 컬럼명 (데이터 처리)
df.select(Terms.COLUMN.DEATH_COUNT)

# 표시명 (UI)
st.metric(Terms.KOREAN.DEATH_COUNT, f"{count:,}건")
```

---

## 📚 관련 파일

- [terminology.yaml](./terminology.yaml) - 용어 사전 원본
- [constants.py](../../dashboard/utils/constants.py) - Terms 클래스
- [terminology.py](../../dashboard/utils/terminology.py) - TerminologyManager 클래스

---

## 🚀 시작하기

```python
# 1. 임포트
from dashboard.utils.constants import Terms

# 2. 사용
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")

# 끝!
```

**모든 용어는 한 곳에서 관리됩니다! 🎉**

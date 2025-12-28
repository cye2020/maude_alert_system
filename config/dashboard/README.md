# Dashboard Configuration

대시보드 설정 파일 모음입니다. 모든 설정은 YAML 파일로 관리되어 코드 수정 없이 변경 가능합니다.

## 📂 파일 구조

```
config/dashboard/
├── README.md                    # 이 문서
├── TERMINOLOGY_GUIDE.md         # 용어 통일 가이드 (상세)
├── terminology.yaml             # 🆕 용어 사전 (한글↔영문 매핑, 컬럼명)
├── defaults.yaml                # 기본 설정 (TOP_N, 차트 높이 등)
├── ui_standards.yaml            # UI 표준 (색상, 레이아웃, 메트릭 라벨)
└── sidebar.yaml                 # 사이드바 구조 (필터 설정)
```

---

## 🎯 각 파일 역할

### 1. `terminology.yaml` 🆕
**용어 통일을 위한 중앙 단어 사전**

- 한국어/영문 용어 매핑
- 컬럼명 표준화
- 메시지 템플릿
- 용어 설명 (툴팁용)

```yaml
korean_terms:
  metrics:
    cfr: '치명률'
    death_rate: '사망률'
    spike: '급증'
```

**사용법:**
```python
from dashboard.utils.constants import Terms

st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")  # "치명률"
```

📖 **자세한 사용법:** [TERMINOLOGY_GUIDE.md](./TERMINOLOGY_GUIDE.md)

---

### 2. `defaults.yaml`
**대시보드 기본 설정값**

- 분석 기본값 (TOP_N, MIN_CASES 등)
- 차트 높이, 색상
- 컬럼명 정의
- 제외 값 설정

```yaml
defaults:
  top_n: 10
  min_cases: 10
  chart_height: 600
```

**사용법:**
```python
from dashboard.utils.constants import Defaults

top_n = Defaults.TOP_N  # 10
```

---

### 3. `ui_standards.yaml`
**UI 표준화 설정**

- 페이지/탭 제목
- 메트릭 라벨 (기존, terminology.yaml로 이전 권장)
- 색상 팔레트
- 레이아웃 설정
- 버튼 라벨

```yaml
colors:
  harm:
    death: "#DC2626"
    serious_injury: "#F59E0B"
```

**사용법:**
```python
from dashboard.utils.constants import HarmColors

color = HarmColors.DEATH  # "#DC2626"
```

---

### 4. `sidebar.yaml`
**사이드바 필터 구조**

- 필터 순서
- 기본값
- 표시 옵션

```yaml
filters:
  date_range:
    enabled: true
    default_years: 3
```

---

## 🔄 설정 변경 흐름

### 기존 방식 (문제)
```
코드 수정 → 여러 파일 찾아서 변경 → 실수 발생 → 불일치
```

### 새로운 방식 (해결)
```
YAML 파일 수정 → 자동으로 전체 반영 → 일관성 보장
```

---

## 📝 용어 통일 전후 비교

### Before (문제점)

**코드 곳곳에 하드코딩:**
```python
# eda_tab.py
st.metric("치명률", f"{cfr:.2f}%")

# cluster_tab.py
st.metric("치명률 (CFR)", f"{cfr:.2f}%")

# overview_tab.py
st.metric("사망률", f"{cfr:.2f}%")  # 잘못된 용어!

# 😱 같은 지표인데 3가지 다른 표현!
```

**문제:**
- 용어 혼용 (치명률 vs 사망률)
- 표현 불일치 (치명률 vs 치명률(CFR))
- 수정 시 전체 검색 필요
- 오타 및 실수 발생

### After (해결)

**중앙 집중식 관리:**
```python
from dashboard.utils.constants import Terms

# 모든 파일에서 동일하게 사용
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")  # ✅ "치명률"

# 용어 변경 시 terminology.yaml만 수정하면 전체 반영!
```

**장점:**
- ✅ 용어 통일 보장
- ✅ 한 곳에서 관리
- ✅ 오타 방지 (IDE 자동완성)
- ✅ 변경 용이

---

## 🎯 실전 예시

### 예시 1: 메트릭 표시

```python
from dashboard.utils.constants import Terms

# Before
st.metric("치명률", f"{cfr:.2f}%")
st.metric("사망", f"{deaths:,}건")

# After
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")
```

### 예시 2: 차트 제목

```python
# Before
st.subheader("기기별 치명률(CFR) 분석")

# After
st.subheader(f"{Terms.KOREAN.DEVICE}별 {Terms.KOREAN.CFR_FULL} 분석")
```

### 예시 3: 경고 메시지

```python
# Before
st.error(f"⚠️ **{device}**의 치명률이 **{cfr:.2f}%**로 매우 높습니다 (중대 피해 {count:,}건)")

# After
msg = Terms.format_message('high_cfr_alert', device=device, cfr=cfr, count=count)
st.error(msg)
```

### 예시 4: DataFrame 컬럼 변환

```python
# Before
display_df = df.rename(columns={
    'death_count': '사망',
    'cfr': '치명률(%)',
    'total_count': '전체 건수'
})

# After
from dashboard.utils.terminology import get_term_manager
term = get_term_manager()
display_df = df.rename(columns=term.column_headers)
```

---

## 🔧 용어 추가/수정 방법

### 1. 용어 추가

```yaml
# terminology.yaml에 추가
korean_terms:
  metrics:
    new_metric: '새로운 지표'
```

```python
# constants.py에 추가
class Terms:
    class KOREAN:
        NEW_METRIC = _term.get('korean_terms.metrics.new_metric', '새로운 지표')
```

### 2. 용어 수정

```yaml
# terminology.yaml만 수정
korean_terms:
  metrics:
    cfr: '치명률'  # -> '위험도'로 변경하면 전체 반영
```

---

## 📚 코드에서 사용법

### 기본 임포트

```python
from dashboard.utils.constants import (
    Terms,           # 용어 통일
    Defaults,        # 기본 설정
    ColumnNames,     # 컬럼명
    HarmColors,      # 색상
)
```

### 자주 사용하는 패턴

```python
# 1. 메트릭
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")

# 2. 제목
st.subheader(f"{Terms.KOREAN.MANUFACTURER}별 분석")

# 3. 메시지
msg = Terms.format_message('high_cfr_alert', device=name, cfr=cfr, count=cnt)

# 4. 컬럼명
df.select(Terms.COLUMN.DEATH_COUNT)

# 5. 설정값
top_n = Defaults.TOP_N
```

---

## ✅ 마이그레이션 체크리스트

기존 코드를 새 시스템으로 마이그레이션할 때:

- [ ] 하드코딩된 한글 문자열 찾기
- [ ] `Terms.KOREAN.*` 로 변경
- [ ] 컬럼명 하드코딩 `Terms.COLUMN.*` 로 변경
- [ ] 반복되는 메시지 템플릿으로 이동
- [ ] DataFrame 표시 시 `column_headers` 사용

---

## 🎓 학습 순서

1. **[TERMINOLOGY_GUIDE.md](./TERMINOLOGY_GUIDE.md)** 읽기 (필수!)
2. `terminology.yaml` 구조 파악
3. 간단한 예시부터 적용
4. 기존 코드 점진적 개선

---

## 📖 참고 자료

| 파일 | 용도 | 설명 |
|------|------|------|
| `terminology.yaml` | 용어 사전 | 모든 용어 정의 |
| `TERMINOLOGY_GUIDE.md` | 가이드 | 상세 사용법 |
| `constants.py` | 코드 | Terms 클래스 정의 |
| `terminology.py` | 코드 | TerminologyManager 구현 |

---

## 🚀 Quick Start

```python
# 1. 임포트
from dashboard.utils.constants import Terms

# 2. 사용
st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")

# 3. 메시지
msg = Terms.format_message('high_cfr_alert',
                           device='ABC',
                           cfr=12.5,
                           count=100)
st.error(msg)
```

**끝! 이제 모든 용어가 통일됩니다! 🎉**

---

## 💡 Tips

1. **자동완성 활용**: `Terms.KOREAN.`까지 타이핑하면 IDE가 사용 가능한 용어를 보여줍니다
2. **점진적 적용**: 새 코드부터 적용하고, 기존 코드는 천천히 개선
3. **팀 공유**: 이 README와 TERMINOLOGY_GUIDE를 팀원과 공유
4. **용어 제안**: 새로운 용어가 필요하면 `terminology.yaml`에 추가 후 PR

---

Made with ❤️ for consistent terminology across the dashboard!

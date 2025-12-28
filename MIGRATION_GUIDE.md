# 용어 통일 마이그레이션 가이드

기존 하드코딩된 한글 텍스트를 `Terms`로 변경하는 가이드입니다.

## 🎯 목표

모든 하드코딩된 한글 문자열을 `Terms` 클래스로 변경하여 용어 통일

## 📝 변경 패턴

### 1. 메트릭 라벨

#### Before
```python
st.metric("치명률", f"{cfr:.2f}%")
st.metric("사망", f"{deaths:,}건")
st.metric("중대 피해", f"{severe:,}건")
```

#### After
```python
from dashboard.utils.constants import Terms

st.metric(Terms.KOREAN.CFR, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")
st.metric(Terms.KOREAN.SEVERE_HARM, f"{severe:,}건")
```

---

### 2. 섹션/차트 제목

#### Before
```python
st.subheader("📈 보고 건수 및 중대 피해율 추이")
st.subheader("🔍 리스크 매트릭스")
st.markdown("#### 환자 피해 분포")
st.markdown("#### 결함 유형별 상위 문제 & 사건 유형 분포")  # ❌ 너무 더티!
```

#### After
```python
from dashboard.utils.constants import Terms

st.subheader(f"📈 {Terms.KOREAN.REPORT_COUNT} 및 {Terms.KOREAN.SEVERE_HARM_RATE} {Terms.KOREAN.TREND}")
st.subheader(f"🔍 {Terms.KOREAN.RISK_MATRIX}")
st.markdown(f"#### {Terms.KOREAN.HARM_DISTRIBUTION}")
st.markdown(f"#### {Terms.section_title('entity_analysis', entity=Terms.KOREAN.DEFECT_TYPE)}")
# 또는
st.markdown(f"#### {Terms.KOREAN.DEFECT_TYPE_ANALYSIS}")
```

---

### 3. 복합 제목 (템플릿 활용)

#### Before
```python
st.subheader(f"{entity}별 상위 {metric}")
st.subheader("제조사별 치명률")
st.markdown("#### 결함 유형 분석")
```

#### After
```python
# 방법 1: 템플릿 사용
st.subheader(Terms.section_title('top_items_by_entity',
                                  entity=Terms.KOREAN.MANUFACTURER,
                                  metric=Terms.KOREAN.REPORT_COUNT))

# 방법 2: 직접 조합
st.subheader(f"{Terms.KOREAN.MANUFACTURER}별 {Terms.KOREAN.CFR}")

# 방법 3: 미리 정의된 섹션 사용
st.markdown(f"#### {Terms.KOREAN.DEFECT_TYPE_ANALYSIS}")
```

---

### 4. DataFrame 컬럼 헤더

#### Before
```python
display_df = df.rename(columns={
    'death_count': '사망',
    'cfr': '치명률(%)',
    'total_count': '전체 건수',
    'manufacturer_product': '제조사-제품군'
})
```

#### After
```python
from dashboard.utils.terminology import get_term_manager

# 방법 1: 전체 매핑 사용 (권장)
term = get_term_manager()
display_df = df.rename(columns=term.column_headers)

# 방법 2: 개별 변환
display_df = df.rename(columns={
    'death_count': Terms.get_column_header('death_count'),
    'cfr': Terms.get_column_header('cfr'),
    'total_count': Terms.get_column_header('total_count')
})
```

---

### 5. 메시지

#### Before
```python
st.error(f"⚠️ **{device}**의 치명률이 **{cfr:.2f}%**로 매우 높습니다 (중대 피해 {count:,}건)")
st.success(f"✅ 평균 치명률이 **{avg_cfr:.2f}%**로 양호한 수준입니다")
st.info("선택한 조건에 해당하는 데이터가 없습니다.")
```

#### After
```python
# 템플릿 사용
msg = Terms.format_message('high_cfr_alert', device=device, cfr=cfr, count=count)
st.error(msg)

msg = Terms.format_message('low_cfr_info', cfr=avg_cfr)
st.success(msg)

st.info(Terms.format_message('no_data'))
```

---

## 📋 주요 용어 매핑표

### 메트릭

| 하드코딩 | Terms 사용 |
|----------|------------|
| `"치명률"` | `Terms.KOREAN.CFR` |
| `"치명률(CFR)"` | `Terms.KOREAN.CFR_FULL` |
| `"사망률"` | `Terms.KOREAN.DEATH_RATE` |
| `"사망"` | `Terms.KOREAN.DEATH_COUNT` |
| `"중대 피해"` | `Terms.KOREAN.SEVERE_HARM` |
| `"중증 부상"` | `Terms.KOREAN.SERIOUS_INJURY` |
| `"보고 건수"` | `Terms.KOREAN.REPORT_COUNT` |

### 엔티티

| 하드코딩 | Terms 사용 |
|----------|------------|
| `"제조사"` | `Terms.KOREAN.MANUFACTURER` |
| `"제품군"` | `Terms.KOREAN.PRODUCT` |
| `"기기"` | `Terms.KOREAN.DEVICE` |
| `"결함 유형"` | `Terms.KOREAN.DEFECT_TYPE` |
| `"문제 부품"` | `Terms.KOREAN.COMPONENT` |
| `"클러스터"` | `Terms.KOREAN.CLUSTER` |
| `"환자 피해"` | `Terms.KOREAN.PATIENT_HARM` (엔티티에는 없음, 직접 추가 필요)` |

### 분석/패턴

| 하드코딩 | Terms 사용 |
|----------|------------|
| `"급증"` | `Terms.KOREAN.SPIKE` |
| `"증가"` | `Terms.KOREAN.INCREASE` |
| `"감소"` | `Terms.KOREAN.DECREASE` |
| `"시계열"` | `Terms.KOREAN.TIME_SERIES` |
| `"추이"` | `Terms.KOREAN.TREND` |
| `"월별"` | `Terms.KOREAN.MONTHLY` |
| `"분포"` | `Terms.KOREAN.DISTRIBUTION` |

### 섹션 제목

| 하드코딩 | Terms 사용 |
|----------|------------|
| `"개요"` | `Terms.KOREAN.OVERVIEW` |
| `"요약"` | `Terms.KOREAN.SUMMARY` |
| `"인사이트"` | `Terms.KOREAN.INSIGHTS` |
| `"결함 유형 분석"` | `Terms.KOREAN.DEFECT_TYPE_ANALYSIS` |
| `"문제 부품 분석"` | `Terms.KOREAN.COMPONENT_ANALYSIS` |
| `"환자 피해 분포"` | `Terms.KOREAN.HARM_DISTRIBUTION` |
| `"사건 유형 분포"` | `Terms.KOREAN.EVENT_TYPE_DISTRIBUTION` |
| `"치명률(CFR) 분석"` | `Terms.KOREAN.CFR_ANALYSIS` |
| `"리스크 매트릭스"` | `Terms.KOREAN.RISK_MATRIX` |

---

## 🔧 실전 예시

### 예시 1: overview_tab.py

#### Before
```python
st.subheader("📈 보고 건수 및 중대 피해율 추이")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("전체 보고 건수", f"{total:,}건")
with col2:
    st.metric("사망", f"{deaths:,}건")
with col3:
    st.metric("중대 피해율", f"{severe_rate:.2f}%")

st.markdown("---")
st.subheader("🔍 리스크 매트릭스")
```

#### After
```python
from dashboard.utils.constants import Terms

st.subheader(f"📈 {Terms.KOREAN.REPORT_COUNT} 및 {Terms.KOREAN.SEVERE_HARM_RATE} {Terms.KOREAN.TREND}")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(Terms.KOREAN.TOTAL_REPORTS, f"{total:,}건")
with col2:
    st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")
with col3:
    st.metric(Terms.KOREAN.SEVERE_HARM_RATE, f"{severe_rate:.2f}%")

st.markdown("---")
st.subheader(f"🔍 {Terms.KOREAN.RISK_MATRIX}")
```

---

### 예시 2: eda_tab.py - 더티한 제목 개선

#### Before (❌ 매우 더티함!)
```python
st.markdown("#### 결함 유형별 상위 문제 & 사건 유형 분포")
st.markdown("#### defect type별 환자 피해 분포")
st.markdown("### 💀 기기별 치명률(CFR) 분석")
```

#### After (✅ 깔끔!)
```python
from dashboard.utils.constants import Terms

# 방법 1: 미리 정의된 섹션 제목 사용
st.markdown(f"#### {Terms.KOREAN.DEFECT_TYPE_ANALYSIS}")
st.markdown(f"#### {Terms.KOREAN.HARM_DISTRIBUTION}")
st.markdown(f"### 💀 {Terms.KOREAN.CFR_ANALYSIS}")

# 방법 2: 템플릿 사용
st.markdown(f"#### {Terms.section_title('entity_analysis', entity=Terms.KOREAN.DEFECT_TYPE)}")
```

---

### 예시 3: cluster_tab.py

#### Before
```python
st.metric("치명률 (CFR)", f"{cfr:.2f}%")
st.metric("사망", f"{deaths:,}건")

st.markdown("#### 💀 클러스터별 치명률")

if cfr > 5.0:
    st.error(f"⚠️ **Cluster {cluster_id}**의 치명률이 **{cfr:.2f}%**로 가장 높습니다")
```

#### After
```python
from dashboard.utils.constants import Terms

st.metric(Terms.KOREAN.CFR_FULL, f"{cfr:.2f}%")
st.metric(Terms.KOREAN.DEATH_COUNT, f"{deaths:,}건")

st.markdown(f"#### 💀 {Terms.KOREAN.CLUSTER}별 {Terms.KOREAN.CFR}")

if cfr > 5.0:
    msg = Terms.format_message('cluster_high_risk',
                               cluster_id=cluster_id,
                               cfr=cfr,
                               count=severe_harm)
    st.error(msg)
```

---

## 🚀 마이그레이션 절차

### 1단계: 임포트 추가
```python
from dashboard.utils.constants import Terms
```

### 2단계: 메트릭부터 변경
```python
# Before
st.metric("치명률", ...)
# After
st.metric(Terms.KOREAN.CFR, ...)
```

### 3단계: 섹션 제목 변경
```python
# Before
st.subheader("결함 유형 분석")
# After
st.subheader(Terms.KOREAN.DEFECT_TYPE_ANALYSIS)
```

### 4단계: 복잡한 제목 템플릿화
```python
# Before
st.markdown(f"#### {entity}별 상위 {metric}")
# After
st.markdown(f"#### {Terms.section_title('metric_by_entity', entity=entity, metric=metric)}")
```

### 5단계: 메시지 템플릿 적용
```python
# Before
st.error(f"⚠️ 경고: {message}")
# After
st.error(Terms.format_message('template_key', ...))
```

---

## ✅ 체크리스트

각 파일을 마이그레이션할 때 확인:

- [ ] `from dashboard.utils.constants import Terms` 임포트 추가
- [ ] `st.metric()` 라벨 변경
- [ ] `st.subheader()`, `st.markdown()` 제목 변경
- [ ] DataFrame 컬럼 헤더 변경
- [ ] 반복되는 메시지는 템플릿으로 이동
- [ ] 하드코딩된 한글 문자열 검색 (정규식: `[\"'][가-힣]+[\"']`)

---

## 🔍 하드코딩 찾기 명령어

```bash
# 하드코딩된 한글 찾기
grep -rn '["'\''][가-힣]' dashboard/*.py

# st.metric, st.subheader 등에서 하드코딩 찾기
grep -rn 'st\.\(metric\|subheader\|markdown\).*["'\''][가-힣]' dashboard/*.py
```

---

## 📚 참고 자료

- [terminology.yaml](config/dashboard/terminology.yaml) - 모든 용어 정의
- [TERMINOLOGY_GUIDE.md](config/dashboard/TERMINOLOGY_GUIDE.md) - 상세 가이드
- [constants.py](dashboard/utils/constants.py) - Terms 클래스

---

## 💡 Tips

1. **한 번에 하나씩**: 파일 단위로 마이그레이션
2. **테스트**: 변경 후 대시보드가 정상 작동하는지 확인
3. **일관성**: 같은 의미는 항상 같은 `Terms` 사용
4. **템플릿 활용**: 반복되는 패턴은 `section_title()` 또는 `format_message()` 활용
5. **새 용어**: 필요한 용어가 없으면 `terminology.yaml`에 추가

---

**점진적으로 개선하되, 새 코드는 반드시 Terms를 사용하세요!** 🎯

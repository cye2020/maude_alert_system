# analysis.py
"""데이터 분석 로직 함수"""

import polars as pl
import streamlit as st
from typing import List, Optional
from .constants import ColumnNames, PatientHarmLevels,Defaults
from .data_utils import get_year_month_expr, apply_basic_filters
from src import BaselineAggregator
from dateutil.relativedelta import relativedelta

@st.cache_data
def get_filtered_products(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    _year_month_expr: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """제조사-제품군 조합을 필터링하여 이상 사례 발생 수 집계

    Args:
        _lf: LazyFrame (언더스코어로 시작하여 캐싱에서 제외)
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군(제품코드) 컬럼명
        date_col: 날짜 컬럼명 (기본: date_received)
        selected_dates: 선택된 년-월 리스트 (예: ['2024-01', '2024-02'])
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        top_n: 상위 N개만 반환 (None이면 전체)
        _year_month_expr: 년-월 컬럼 생성 표현식 (재사용용, 언더스코어로 시작하여 캐싱에서 제외)

    Returns:
        필터링된 결과 DataFrame
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=manufacturer_col,
        product_col=product_col,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=True
    )

    # 집계
    result = (
        filtered_lf
        .group_by("manufacturer_product")
        .agg(pl.len().alias("total_count"))
        .sort("total_count", descending=True)
    )

    # top_n 처리
    if top_n is not None:
        result = result.head(top_n)

    return result.collect()


@st.cache_data
def get_monthly_counts(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    _year_month_expr: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """년-월별로 제조사-제품군 조합의 개수를 집계하여 반환

    Args:
        _lf: LazyFrame
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군(제품코드) 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        _year_month_expr: 년-월 컬럼 생성 표현식 (재사용용, 언더스코어로 시작하여 캐싱에서 제외)

    Returns:
        년-월별 집계 DataFrame (year_month, manufacturer_product, total_count)
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=manufacturer_col,
        product_col=product_col,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=True
    )

    # 년-월별, 제조사-제품군별 집계
    result = (
        filtered_lf
        .group_by(["year_month", "manufacturer_product"])
        .agg(pl.len().alias("total_count"))
        .sort(["year_month", "total_count"], descending=[False, True])
        .collect()
    )

    return result


@st.cache_data
def analyze_manufacturer_defects(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    _year_month_expr: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """제조사-제품군 조합별 결함 분석 (필터 적용)

    Args:
        _lf: LazyFrame
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        제조사-제품군별 결함 분석 결과 DataFrame
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=manufacturer_col,
        product_col=product_col,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=True
    )

    # 결함 분석 집계
    result = (
        filtered_lf
        .group_by(["manufacturer_product", ColumnNames.DEFECT_TYPE])
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .with_columns(
            (pl.col("count") / pl.col("count").sum().over("manufacturer_product") * 100)
            .round(2)
            .alias("percentage")
        )
        .sort(["manufacturer_product", "percentage"], descending=[False, True])
        .collect()
    )

    return result


@st.cache_data
def analyze_defect_components(
    _lf: pl.LazyFrame,
    defect_type: str,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    top_n: int = Defaults.TOP_N,
    _year_month_expr: Optional[pl.Expr] = None
) -> Optional[pl.DataFrame]:
    """특정 결함 종류의 문제 기기 부품 분석

    Args:
        _lf: LazyFrame
        defect_type: 분석할 결함 종류
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        top_n: 상위 N개 문제 부품 표시
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        문제 부품 분포 DataFrame (None이면 데이터 없음)
    """
    # 기본 필터링
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=manufacturer_col,
        product_col=product_col,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=False
    )

    # 결함 유형 필터
    filtered_lf = filtered_lf.filter(pl.col(ColumnNames.DEFECT_TYPE) == defect_type)

    # problem_components가 null이 아닌 데이터만 필터링
    defect_data = filtered_lf.filter(pl.col(ColumnNames.PROBLEM_COMPONENTS).is_not_null())

    # 전체 개수 계산
    total = defect_data.select(pl.len()).collect().item()

    if total == 0:
        return None

    # 문제 부품 분포 집계
    component_dist = (
        defect_data
        .group_by(ColumnNames.PROBLEM_COMPONENTS)
        .agg(pl.len().alias('count'))
        .with_columns(
            (pl.col('count') / total * 100)
            .round(2)
            .alias('percentage')
        )
        .sort('count', descending=True)
        .head(top_n)
        .collect()
    )

    return component_dist


@st.cache_data
def calculate_cfr_by_device(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    event_column: str = ColumnNames.EVENT_TYPE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    top_n: Optional[int] = None,
    min_cases: int = Defaults.MIN_CASES,
    _year_month_expr: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """제조사-제품군 조합별 치명률(Case Fatality Rate)을 계산하는 함수

    치명률(CFR) = (사망 건수 / 해당 기기 총 보고 건수) × 100

    Args:
        _lf: LazyFrame
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군 컬럼명
        event_column: 사건 유형 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        top_n: 상위 N개 기기만 분석 (None이면 전체)
        min_cases: 최소 보고 건수 (이보다 적은 기기는 제외, 통계적 신뢰도 확보)
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        기기별 치명률 결과 DataFrame
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=manufacturer_col,
        product_col=product_col,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=True
    )

    # 제조사-제품군 조합별 전체 건수와 사건 유형별 건수
    device_stats = (
        filtered_lf
        .group_by("manufacturer_product")
        .agg([
            pl.len().alias('total_cases'),
            (pl.col(event_column) == 'Death').sum().alias('death_count'),
            (pl.col(event_column) == 'Injury').sum().alias('injury_count'),
            (pl.col(event_column) == 'Malfunction').sum().alias('malfunction_count')
        ])
        .filter(pl.col('total_cases') >= min_cases)  # 최소 건수 필터
        .with_columns([
            # CFR 계산
            (pl.col('death_count') / pl.col('total_cases') * 100).round(2).alias('cfr'),
            # 부상률
            (pl.col('injury_count') / pl.col('total_cases') * 100).round(2).alias('injury_rate'),
            # 오작동률
            (pl.col('malfunction_count') / pl.col('total_cases') * 100).round(2).alias('malfunction_rate')
        ])
        .sort('cfr', descending=True)
    )

    # Top N만
    if top_n:
        device_stats = device_stats.head(top_n)

    result = device_stats.collect()

    return result


# ==================== Spike Detection Tab 분석 함수 ====================

@st.cache_data(show_spinner=False)
def perform_spike_detection(
    _lf: pl.LazyFrame,
    as_of_month: str,
    window: int = 1,
    min_c_recent: int = 20,
    z_threshold: float = 2.0,
    eps: float = 0.1,
    alpha: float = 0.05,
    correction: str = 'fdr_bh',
    min_methods: int = 2,
) -> pl.DataFrame:
    """스파이크 탐지 분석 수행

    Args:
        _lf: MAUDE 데이터 LazyFrame
        as_of_month: 기준 월 (예: "2025-11")
        window: 윈도우 크기 (1 또는 3)
        min_c_recent: 최소 최근 케이스 수
        z_threshold: Z-score 임계값
        eps: Epsilon 값 (z_log 계산용)
        alpha: 유의수준 (Poisson 검정용)
        correction: 다중검정 보정 방법 ('bonferroni', 'sidak', 'fdr_bh', None)
        min_methods: 앙상블 스파이크 판정 최소 방법 수

    Returns:
        스파이크 탐지 결과 DataFrame (pattern 컬럼 포함)
    """
    # BaselineAggregator 초기화 및 베이스라인 테이블 생성
    aggregator = BaselineAggregator(_lf)

    baseline_lf = aggregator.create_baseline_table(
        as_of_month=as_of_month,
        z_threshold=z_threshold,
        min_c_recent=min_c_recent,
        eps=eps,
        alpha=alpha,
        correction_method=correction if correction != 'none' else None,
        verbose=False
    )

    # 앙상블 스파이크 탐지 (pattern이 이미 포함되어 있음)
    result_df = (
        baseline_lf
        .filter(pl.col("window") == window)
        .with_columns(
            (
                pl.col("is_spike").cast(pl.Int8) +
                pl.col("is_spike_z").cast(pl.Int8) +
                pl.col("is_spike_p").cast(pl.Int8)
            ).alias("n_methods")
        )
        .with_columns(
            (pl.col("n_methods") >= min_methods).alias("is_spike_ensemble")
        )
        .sort(["n_methods", "score_pois"], descending=True)
        .collect()
    )

    return result_df


@st.cache_data(show_spinner=False)
def get_spike_time_series(
    _lf: pl.LazyFrame,
    keywords: List[str],
    start_month: str,
    end_month: str,
    date_col: str = ColumnNames.DATE_RECEIVED,
    window: int = 1,
) -> pl.DataFrame:
    """특정 키워드들의 시계열 데이터 추출

    Args:
        _lf: MAUDE 데이터 LazyFrame
        keywords: 키워드 리스트
        start_month: 시작 월 (예: "2024-01")
        end_month: 종료 월 (예: "2025-11")
        date_col: 날짜 컬럼명
        window: 윈도우 크기 (기준 기간 계산용)

    Returns:
        시계열 데이터 DataFrame (columns: month, keyword, count, ratio)
        ratio = 해당 월 count / 기준 기간(이전 12개월) 평균 count
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    # 월별 키워드 집계
    lf_with_month = _lf.with_columns(
        pl.col(date_col).cast(pl.Date).dt.strftime("%Y-%m").alias("month")
    )

    # 키워드별 월별 집계 (전체 기간)
    keyword_monthly = (
        lf_with_month
        .filter(pl.col(ColumnNames.DEFECT_TYPE).is_in(keywords))
        .group_by(["month", ColumnNames.DEFECT_TYPE])
        .agg(pl.len().alias("count"))
        .sort("month")
        .collect()
    )

    # 각 월별로 ratio 계산 (해당 월 / 이전 12개월 평균)
    result_rows = []

    for keyword in keywords:
        keyword_data = keyword_monthly.filter(pl.col(ColumnNames.DEFECT_TYPE) == keyword)

        for row in keyword_data.iter_rows(named=True):
            month = row["month"]
            count = row["count"]

            # 이전 12개월 범위 계산
            current_date = datetime.strptime(month, "%Y-%m")
            base_end = current_date - relativedelta(months=window)
            base_start = base_end - relativedelta(months=11)

            # 기준 기간 데이터 추출
            base_data = keyword_data.filter(
                (pl.col("month") >= base_start.strftime("%Y-%m")) &
                (pl.col("month") <= base_end.strftime("%Y-%m"))
            )

            # 기준 기간 평균 계산
            if len(base_data) > 0:
                base_avg = base_data["count"].mean()
                ratio = (count + 1) / (base_avg + 1) if base_avg is not None else 1.0
            else:
                ratio = 1.0

            result_rows.append({
                "month": month,
                "keyword": keyword,
                "count": count,
                "ratio": round(ratio, 2)
            })

    result = pl.DataFrame(result_rows)

    # start_month ~ end_month 범위로 필터링
    result = result.filter(
        (pl.col("month") >= start_month) & (pl.col("month") <= end_month)
    )

    return result


# ==================== Overview Tab 분석 함수 ====================

@st.cache_data(show_spinner=False)
def calculate_big_numbers(
    _data: pl.LazyFrame,
    segment: Optional[str] = None,
    segment_value: Optional[str] = None,
    start_date = None,
    end_date = None,
) -> dict:
    """Big Number 4개 계산 (선택된 기간 전체 vs 그 이전 동일 기간 슬라이딩 비교)

    Args:
        _data: LazyFrame 데이터
        segment: 세그먼트 컬럼명 (필터링할 컬럼)
        segment_value: 세그먼트 값 (특정 제조사, 제품 등)
        start_date: 분석 시작 날짜 (datetime 객체)
        end_date: 분석 종료 날짜 (datetime 객체)

    Returns:
        {
            'total_reports': 총 보고서 수 (선택 기간 전체),
            'total_reports_delta': 이전 동일 기간 대비 변동률 (%),
            'severe_harm_rate': 중대 피해 발생률 (%),
            'severe_harm_rate_delta': 이전 동일 기간 대비 변동 (%p),
            'defect_confirmed_rate': 제조사 결함 확정률 (%),
            'defect_confirmed_rate_delta': 이전 동일 기간 대비 변동 (%p),
            'most_critical_defect_type': 가장 치명적인 defect type,
            'most_critical_defect_rate': 해당 defect type의 치명률 (%)
        }

    Example:
        사용자가 2024-01 ~ 2025-12 선택 시 (24개월)
        - 현재 기간: 2024-01 ~ 2025-12
        - 이전 기간: 2022-01 ~ 2023-12 (24개월 전으로 슬라이딩)
    """
    # Segment 필터 적용
    if segment and segment_value:
        _data = _data.filter(pl.col(segment) == segment_value)

    # 날짜 범위가 지정되지 않은 경우 전체 데이터에서 최신 날짜 기준
    if not start_date or not end_date:
        max_date = _data.select(pl.col(ColumnNames.DATE_RECEIVED).max()).collect()[ColumnNames.DATE_RECEIVED][0]
        end_date = max_date
        start_date = max_date - relativedelta(years=1)

    # 선택된 기간의 월 수 계산
    months_diff = 1

    # 현재 기간 (월 기준)
    current_start = start_date.replace(day=1)
    current_end = end_date.replace(day=1)

    # 이전 기간 (바로 직전 months_diff개월)
    prev_start = current_start - relativedelta(months=months_diff)
    prev_end = current_end - relativedelta(months=months_diff)

    # Sparkline용 데이터 (최근 6개월)
    sparkline_start = current_end - relativedelta(months=5)
    sparkline_start = sparkline_start.replace(day=1)

    sparkline_data = _data.filter(
        (pl.col(ColumnNames.DATE_RECEIVED) >= sparkline_start) &
        (pl.col(ColumnNames.DATE_RECEIVED) <= current_end)
    ).with_columns(
        pl.col(ColumnNames.DATE_RECEIVED).dt.truncate("1mo").alias("month")
    ).group_by("month").agg([
        pl.len().alias("total_reports"),
        pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
          .then(1).otherwise(0).sum().alias("severe_harm_count"),
        pl.when(pl.col(ColumnNames.DEFECT_CONFIRMED) == True)
          .then(1).otherwise(0).sum().alias("defect_confirmed_count"),
    ]).with_columns([
        (pl.col("severe_harm_count") / pl.col("total_reports") * 100).alias("severe_harm_rate"),
        (pl.col("defect_confirmed_count") / pl.col("total_reports") * 100).alias("defect_confirmed_rate"),
    ]).sort("month").collect()

    # 현재 기간 데이터 집계
    latest_data = _data.filter(
        (pl.col(ColumnNames.DATE_RECEIVED) >= current_start) &
        (pl.col(ColumnNames.DATE_RECEIVED) <= current_end)
    )
    # st.write(latest_data.select(pl.col(ColumnNames.DATE_RECEIVED)))  
    
    latest_df = latest_data.select([
        pl.len().alias("total"),
        pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
          .then(1).otherwise(0).sum().alias("severe_harm_count"),
        pl.when(pl.col(ColumnNames.DEFECT_CONFIRMED) == True)
          .then(1).otherwise(0).sum().alias("defect_confirmed_count"),
    ]).collect()

    # 이전 동일 기간 데이터 집계 (먼저 계산)
    prev_data = _data.filter(
        (pl.col(ColumnNames.DATE_RECEIVED) >= prev_start) &
        (pl.col(ColumnNames.DATE_RECEIVED) <= prev_end)
    )

    # 현재 기간: 가장 치명적인 defect type 찾기 (치명률 기준)
    current_defect_stats = latest_data.filter(
        ~pl.col(ColumnNames.DEFECT_TYPE).is_in(Defaults.EXCLUDE_DEFECT_TYPES)
    ).group_by(ColumnNames.DEFECT_TYPE).agg([
        pl.len().alias("total_count"),
        pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
          .then(1).otherwise(0).sum().alias("critical_count")
    ]).with_columns(
        (pl.col("critical_count") / pl.col("total_count") * 100).alias("critical_rate")
    ).sort("critical_rate", descending=True).limit(1).collect()

    if len(current_defect_stats) > 0:
        most_critical_defect_type = current_defect_stats[ColumnNames.DEFECT_TYPE][0]
        most_critical_defect_rate = current_defect_stats["critical_rate"][0]
    else:
        most_critical_defect_type = "N/A"
        most_critical_defect_rate = 0.0

    # 이전 기간: 가장 치명적인 defect type 찾기
    prev_defect_stats = prev_data.filter(
        ~pl.col(ColumnNames.DEFECT_TYPE).is_in(Defaults.EXCLUDE_DEFECT_TYPES)
    ).group_by(ColumnNames.DEFECT_TYPE).agg([
        pl.len().alias("total_count"),
        pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
          .then(1).otherwise(0).sum().alias("critical_count")
    ]).with_columns(
        (pl.col("critical_count") / pl.col("total_count") * 100).alias("critical_rate")
    ).sort("critical_rate", descending=True).limit(1).collect()

    if len(prev_defect_stats) > 0:
        prev_most_critical_defect_type = prev_defect_stats[ColumnNames.DEFECT_TYPE][0]
        prev_most_critical_defect_rate = prev_defect_stats["critical_rate"][0]
    else:
        prev_most_critical_defect_type = "N/A"
        prev_most_critical_defect_rate = 0.0

    prev_df = prev_data.select([
        pl.len().alias("total"),
        pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
          .then(1).otherwise(0).sum().alias("severe_harm_count"),
        pl.when(pl.col(ColumnNames.DEFECT_CONFIRMED) == True)
          .then(1).otherwise(0).sum().alias("defect_confirmed_count"),
    ]).collect()

    # 최신 한 달 수치
    latest_total = latest_df["total"][0]
    
    latest_severe_harm = latest_df["severe_harm_count"][0]
    latest_defect_confirmed = latest_df["defect_confirmed_count"][0]

    # 전월 수치
    prev_total = prev_df["total"][0]
    prev_severe_harm = prev_df["severe_harm_count"][0]
    prev_defect_confirmed = prev_df["defect_confirmed_count"][0]

    # 비율 계산
    latest_severe_harm_rate = (latest_severe_harm / latest_total * 100) if latest_total > 0 else 0.0
    latest_defect_confirmed_rate = (latest_defect_confirmed / latest_total * 100) if latest_total > 0 else 0.0

    prev_severe_harm_rate = (prev_severe_harm / prev_total * 100) if prev_total > 0 else 0.0
    prev_defect_confirmed_rate = (prev_defect_confirmed / prev_total * 100) if prev_total > 0 else 0.0

    # 변동률 계산
    total_delta = ((latest_total - prev_total) / prev_total * 100) if prev_total > 0 else None
    severe_harm_rate_delta = latest_severe_harm_rate - prev_severe_harm_rate
    defect_confirmed_rate_delta = latest_defect_confirmed_rate - prev_defect_confirmed_rate

    return {
        "total_reports": latest_total,
        "total_reports_delta": total_delta,
        "total_reports_sparkline": sparkline_data["total_reports"].to_list(),
        "severe_harm_rate": latest_severe_harm_rate,
        "severe_harm_rate_delta": severe_harm_rate_delta,
        "severe_harm_sparkline": sparkline_data["severe_harm_rate"].to_list(),
        "defect_confirmed_rate": latest_defect_confirmed_rate,
        "defect_confirmed_rate_delta": defect_confirmed_rate_delta,
        "defect_sparkline": sparkline_data["defect_confirmed_rate"].to_list(),
        "most_critical_defect_type": most_critical_defect_type,
        "most_critical_defect_rate": most_critical_defect_rate,
        "prev_most_critical_defect_type": prev_most_critical_defect_type,
        "prev_most_critical_defect_rate": prev_most_critical_defect_rate,
    }


# ==================== Phase 2: Treemap & Risk Matrix ====================

@st.cache_data(show_spinner=False)
def get_treemap_data(
    _lf: pl.LazyFrame,
    start_date = None,
    end_date = None,
    segment_col: Optional[str] = None,
    segment_value: Optional[str] = None,
    top_n: int = 10
) -> pl.DataFrame:
    """Treemap용 데이터 집계 (Defect Type → Patient Harm Level)

    Args:
        _lf: LazyFrame 데이터
        start_date: 시작 날짜 (datetime 객체)
        end_date: 종료 날짜 (datetime 객체)
        segment_col: 세그먼트 컬럼명 (예: "manufacturer_name", "product_code")
        segment_value: 세그먼트 값 (예: "MEDTRONIC", "FMH")
        top_n: 상위 N개 Defect Type만 표시

    Returns:
        계층 구조 DataFrame (defect_type, patient_harm, count, severe_harm_rate)
    """
    # 날짜 필터링
    filtered_data = _lf
    if start_date and end_date:
        filtered_data = filtered_data.filter(
            (pl.col(ColumnNames.DATE_RECEIVED) >= start_date) &
            (pl.col(ColumnNames.DATE_RECEIVED) <= end_date)
        )

    # 세그먼트 필터링
    if segment_col and segment_value:
        filtered_data = filtered_data.filter(pl.col(segment_col) == segment_value)

    # Top N Defect Type 추출
    top_defects = (
        filtered_data
        .filter(~pl.col(ColumnNames.DEFECT_TYPE).is_in(Defaults.EXCLUDE_DEFECT_TYPES))
        .group_by(ColumnNames.DEFECT_TYPE)
        .agg(pl.len().alias("total_count"))
        .sort("total_count", descending=True)
        .limit(top_n)
        .select(ColumnNames.DEFECT_TYPE)
        .collect()
    )

    top_defect_list = top_defects[ColumnNames.DEFECT_TYPE].to_list()

    # Defect Type × Patient Harm 집계
    result = (
        filtered_data
        .filter(pl.col(ColumnNames.DEFECT_TYPE).is_in(top_defect_list))
        .group_by([ColumnNames.DEFECT_TYPE, ColumnNames.PATIENT_HARM])
        .agg(pl.len().alias("count"))
        .with_columns([
            # Defect Type별 전체 count
            pl.col("count").sum().over(ColumnNames.DEFECT_TYPE).alias("defect_total"),
            # Defect Type별 severe harm count
            pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
              .then(pl.col("count"))
              .otherwise(0)
              .sum()
              .over(ColumnNames.DEFECT_TYPE)
              .alias("severe_count")
        ])
        .with_columns(
            # 치명률 계산
            (pl.col("severe_count") / pl.col("defect_total") * 100).alias("severe_harm_rate")
        )
        .sort([ColumnNames.DEFECT_TYPE, "count"], descending=[False, True])
        .collect()
    )

    return result


@st.cache_data(show_spinner=False)
def get_risk_matrix_data(
    _lf: pl.LazyFrame,
    start_date = None,
    end_date = None,
    segment_col: Optional[str] = None,
    segment_value: Optional[str] = None,
    view_mode: str = "defect_type",
    top_n: int = 20
) -> pl.DataFrame:
    """Risk Matrix용 데이터 집계

    Args:
        _lf: LazyFrame 데이터
        start_date: 시작 날짜 (datetime 객체)
        end_date: 종료 날짜 (datetime 객체)
        segment_col: 세그먼트 컬럼명
        segment_value: 세그먼트 값
        view_mode: "defect_type", "manufacturer", "product"
        top_n: 상위 N개만 표시

    Returns:
        DataFrame (entity, report_count, severe_harm_rate, defect_confirmed_rate)
    """
    # 날짜 필터링
    filtered_data = _lf
    if start_date and end_date:
        filtered_data = filtered_data.filter(
            (pl.col(ColumnNames.DATE_RECEIVED) >= start_date) &
            (pl.col(ColumnNames.DATE_RECEIVED) <= end_date)
        )

    # view_mode에 따라 group_by 컬럼 결정
    if view_mode == "defect_type":
        group_col = ColumnNames.DEFECT_TYPE
        # 세그먼트 필터 적용
        if segment_col and segment_value:
            filtered_data = filtered_data.filter(pl.col(segment_col) == segment_value)
        # exclude 필터
        filtered_data = filtered_data.filter(~pl.col(group_col).is_in(Defaults.EXCLUDE_DEFECT_TYPES))

    elif view_mode == "manufacturer":
        group_col = ColumnNames.MANUFACTURER
        # 특정 제품코드의 제조사들 비교
        if segment_value:
            filtered_data = filtered_data.filter(pl.col(ColumnNames.PRODUCT_CODE) == segment_value)

    elif view_mode == "product":
        group_col = ColumnNames.PRODUCT_CODE
        # 특정 제조사의 제품들 비교
        if segment_value:
            filtered_data = filtered_data.filter(pl.col(ColumnNames.MANUFACTURER) == segment_value)

    else:
        group_col = ColumnNames.DEFECT_TYPE

    # 집계
    result = (
        filtered_data
        .group_by(group_col)
        .agg([
            pl.len().alias("report_count"),
            pl.when(pl.col(ColumnNames.PATIENT_HARM).is_in(PatientHarmLevels.SERIOUS))
              .then(1).otherwise(0).sum().alias("severe_harm_count"),
            pl.when(pl.col(ColumnNames.DEFECT_CONFIRMED) == True)
              .then(1).otherwise(0).sum().alias("defect_confirmed_count")
        ])
        .with_columns([
            (pl.col("severe_harm_count") / pl.col("report_count") * 100).alias("severe_harm_rate"),
            (pl.col("defect_confirmed_count") / pl.col("report_count") * 100).alias("defect_confirmed_rate")
        ])
        .sort("report_count", descending=True)
        .limit(top_n)
        .collect()
    )

    # 컬럼명을 'entity'로 통일
    result = result.rename({group_col: "entity"})

    return result

# analysis_cluster.py
"""Cluster/Defect Type 분석 관련 함수"""

import polars as pl
import streamlit as st
import ast
from typing import List, Optional
from .constants import ColumnNames, Defaults
from .data_utils import apply_basic_filters


@st.cache_data
def get_available_clusters(
    _lf: pl.LazyFrame,
    cluster_col: str = ColumnNames.CLUSTER,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    exclude_minus_one: bool = True,
    _year_month_expr: Optional[pl.Expr] = None
) -> List:
    """필터링된 데이터에서 사용 가능한 cluster 목록 반환

    Args:
        _lf: LazyFrame
        cluster_col: cluster 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        exclude_minus_one: -1 제외 여부
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        cluster 리스트 (정수형)
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=ColumnNames.MANUFACTURER,
        product_col=ColumnNames.PRODUCT_CODE,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=False
    )

    filtered_lf = filtered_lf.filter(pl.col(cluster_col).is_not_null())

    # -1 제외 옵션
    if exclude_minus_one:
        filtered_lf = filtered_lf.filter(pl.col(cluster_col) != -1)

    clusters = (
        filtered_lf
        .select(pl.col(cluster_col))
        .unique()
        .sort(cluster_col)
        .collect()
    )[cluster_col].to_list()

    return clusters


@st.cache_data
def cluster_keyword_unpack(
    _lf: pl.LazyFrame,
    col_name: str = ColumnNames.PROBLEM_COMPONENTS,
    cluster_col: str = ColumnNames.DEFECT_TYPE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    top_n: int = Defaults.TOP_N,
    _year_month_expr: Optional[pl.Expr] = None
) -> pl.DataFrame:
    """결함 유형 별로 col_name마다 있는 리스트를 열어서 키워드 종류를 추출하고 count

    Args:
        _lf: LazyFrame
        col_name: 리스트가 들어있는 열 이름 (예: 'problem_components')
        cluster_col: 결함 유형 열 이름
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        top_n: 상위 N개 키워드만 반환
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        결함 유형별 키워드, count, ratio를 포함한 DataFrame
    """
    # 기본 필터 적용
    lf_temp = apply_basic_filters(
        _lf,
        manufacturer_col=ColumnNames.MANUFACTURER,
        product_col=ColumnNames.PRODUCT_CODE,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=False
    )

    # 필요한 컬럼만 선택
    lf_temp = lf_temp.select([cluster_col, col_name])

    # 1. 문자열을 리스트로 변환 (필요한 경우)
    schema = lf_temp.collect_schema()
    if schema[col_name] == pl.Utf8:
        def safe_literal_eval(x):
            if not x or x == 'null' or x == 'None':
                return []
            try:
                result = ast.literal_eval(x)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []

        lf_temp = lf_temp.with_columns(
            pl.col(col_name)
            .map_elements(safe_literal_eval, return_dtype=pl.List(pl.Utf8))
        )

    # 2. 전체 데이터를 한 번에 explode (벡터화)
    exploded_lf = (lf_temp
                  .explode(col_name)
                  .filter(pl.col(col_name).is_not_null())
                  .filter(pl.col(col_name) != "")  # 빈 문자열 제거
                 )

    # 3. 결함 유형별로 그룹화하여 카운트 (벡터화)
    keyword_counts = (exploded_lf
                      .with_columns(
                          pl.col(col_name).str.to_lowercase().str.strip_chars()  # 소문자 + 공백 제거
                          )
                      .group_by([cluster_col, col_name])
                      .agg(pl.len().alias('count'))
                     )

    # 4. 결함 유형별 전체 키워드 수 계산
    cluster_totals = (keyword_counts
                      .group_by(cluster_col)
                      .agg(pl.col('count').sum().alias('total_count'))
                     )

    # 5. ratio 계산 및 정렬
    result_lf = (keyword_counts
                 .join(cluster_totals, on=cluster_col)
                 .with_columns(
                     (pl.col('count') / pl.col('total_count') * 100).round(2).alias('ratio')
                 )
                 .select([cluster_col, col_name, 'count', 'ratio'])
                 .sort([cluster_col, 'count'], descending=[False, True])
                )

    # 6. 결함 유형별 상위 N개만 선택
    result_df = (
        result_lf
        .with_columns(
            pl.col('count').rank('dense', descending=True).over(cluster_col).alias('rank')
        )
        .filter(pl.col('rank') <= top_n)
        .drop('rank')
        .collect()
    )

    return result_df


@st.cache_data
def get_patient_harm_summary(
    _lf: pl.LazyFrame,
    event_column: str = ColumnNames.PATIENT_HARM,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    selected_defect_types: Optional[List[str]] = None,
    _year_month_expr: Optional[pl.Expr] = None
) -> dict:
    """환자 피해 분포 계산 (파이 차트용)

    Args:
        _lf: LazyFrame
        event_column: 환자 피해 컬럼명 (patient_harm)
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        selected_defect_types: 선택된 결함 유형 리스트
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        {
            'total_deaths': 사망 건수,
            'total_serious_injuries': 중증 부상 건수,
            'total_minor_injuries': 경증 부상 건수,
            'total_no_injuries': 부상 없음 건수
        }
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=ColumnNames.MANUFACTURER,
        product_col=ColumnNames.PRODUCT_CODE,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=False
    )

    # defect_type 필터 추가
    if selected_defect_types:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.DEFECT_TYPE).is_in(selected_defect_types))

    # 환자 피해별 집계 (No Harm과 No Apparent Injury 모두 체크)
    result = filtered_lf.select([
        (pl.col(event_column) == 'Death').sum().alias('death_count'),
        (pl.col(event_column) == 'Serious Injury').sum().alias('serious_injury_count'),
        (pl.col(event_column) == 'Minor Injury').sum().alias('minor_injury_count'),
        (
            (pl.col(event_column) == 'No Apparent Injury') |
            (pl.col(event_column) == 'No Harm')
        ).sum().alias('no_injury_count'),
        (pl.col(event_column) == 'Unknown').sum().alias('unknown_count')
    ]).collect()

    total_deaths = result['death_count'][0] if len(result) > 0 else 0
    total_serious = result['serious_injury_count'][0] if len(result) > 0 else 0
    total_minor = result['minor_injury_count'][0] if len(result) > 0 else 0
    total_none = result['no_injury_count'][0] if len(result) > 0 else 0
    total_unknown = result['unknown_count'][0] if len(result) > 0 else 0

    return {
        'total_deaths': total_deaths,
        'total_serious_injuries': total_serious,
        'total_minor_injuries': total_minor,
        'total_no_injuries': total_none,
        'total_unknown': total_unknown
    }


@st.cache_data
def cluster_check(
    _lf: pl.LazyFrame,
    cluster_name: int = 0,
    cluster_col: str = ColumnNames.CLUSTER,
    component_col: str = ColumnNames.PROBLEM_COMPONENTS,
    event_col: str = ColumnNames.PATIENT_HARM,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    top_n: int = Defaults.TOP_N,
    _year_month_expr: Optional[pl.Expr] = None,
    manufacturers: tuple = None,  # Cache key parameter
    products: tuple = None,       # Cache key parameter
) -> dict:
    """클러스터별로 분포와 top_n problem_component 차트를 확인

    Args:
        _lf: LazyFrame
        cluster_name: 클러스터 번호 (정수형)
        cluster_col: 클러스터 컬럼명
        component_col: 부품 컬럼명
        event_col: 이벤트 유형 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        top_n: 상위 N개 부품
        _year_month_expr: 년-월 컬럼 생성 표현식

    Returns:
        {
            'harm_summary': 환자 피해 분포,
            'top_components': 상위 부품 DataFrame,
            'total_count': 전체 케이스 수,
            'time_series': 시계열 데이터 (월별 케이스 수)
        }
    """
    # 기본 필터 적용
    filtered_lf = apply_basic_filters(
        _lf,
        manufacturer_col=ColumnNames.MANUFACTURER,
        product_col=ColumnNames.PRODUCT_CODE,
        date_col=date_col,
        selected_dates=selected_dates,
        selected_manufacturers=selected_manufacturers,
        selected_products=selected_products,
        year_month_expr=_year_month_expr,
        add_combo=False
    )

    # 클러스터 필터링
    cluster_lf = filtered_lf.filter(pl.col(cluster_col) == cluster_name)

    # 1. 전체 케이스 수
    total_count = cluster_lf.select(pl.len()).collect()[0, 0]

    # 2. 환자 피해 분포 (patient_harm 컬럼 사용)
    harm_summary = cluster_lf.select([
        (pl.col(event_col) == 'Death').sum().alias('death_count'),
        (pl.col(event_col) == 'Serious Injury').sum().alias('serious_injury_count'),
        (pl.col(event_col) == 'Minor Injury').sum().alias('minor_injury_count'),
        (
            (pl.col(event_col) == 'No Apparent Injury') |
            (pl.col(event_col) == 'No Harm')
        ).sum().alias('no_injury_count'),
        (pl.col(event_col) == 'Unknown').sum().alias('unknown_count')
    ]).collect()

    event_dict = {
        'total_deaths': harm_summary['death_count'][0] if len(harm_summary) > 0 else 0,
        'total_serious_injuries': harm_summary['serious_injury_count'][0] if len(harm_summary) > 0 else 0,
        'total_minor_injuries': harm_summary['minor_injury_count'][0] if len(harm_summary) > 0 else 0,
        'total_no_injuries': harm_summary['no_injury_count'][0] if len(harm_summary) > 0 else 0,
        'total_unknown': harm_summary['unknown_count'][0] if len(harm_summary) > 0 else 0
    }

    # 3. 상위 부품 추출 (cluster_keyword_unpack과 유사한 로직)
    lf_temp = cluster_lf.select([component_col])

    # 문자열을 리스트로 변환 (필요한 경우)
    schema = lf_temp.collect_schema()
    if schema[component_col] == pl.Utf8:
        def safe_literal_eval(x):
            if not x or x == 'null' or x == 'None':
                return []
            try:
                result = ast.literal_eval(x)
                return result if isinstance(result, list) else []
            except (ValueError, SyntaxError):
                return []

        lf_temp = lf_temp.with_columns(
            pl.col(component_col)
            .map_elements(safe_literal_eval, return_dtype=pl.List(pl.Utf8))
        )

    # 리스트 explode 및 카운트
    top_components_df = (
        lf_temp
        .explode(component_col)
        .filter(pl.col(component_col).is_not_null())
        .filter(pl.col(component_col) != "")
        .with_columns(
            pl.col(component_col).str.to_lowercase().str.strip_chars()
        )
        .group_by(component_col)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .head(top_n)
        .with_columns(
            (pl.col('count') / total_count * 100).round(2).alias('ratio')
        )
        .collect()
    )

    # 4. 시계열 데이터 (월별 케이스 수)
    if _year_month_expr is None:
        from .data_utils import get_year_month_expr
        _year_month_expr = get_year_month_expr(_lf, date_col)

    time_series_df = (
        cluster_lf
        .with_columns(_year_month_expr)
        .group_by('year_month')
        .agg(pl.len().alias('count'))
        .sort('year_month')
        .collect()
    )

    # 5. Defect Type 분포 (상위 N개)
    defect_type_df = (
        cluster_lf
        .filter(pl.col(ColumnNames.DEFECT_TYPE).is_not_null())
        .filter(pl.col(ColumnNames.DEFECT_TYPE) != "")
        .group_by(ColumnNames.DEFECT_TYPE)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .head(top_n)
        .with_columns(
            (pl.col('count') / total_count * 100).round(2).alias('ratio')
        )
        .collect()
    )

    # 6. Defect Confirmed 분포 (한글로 매핑)
    defect_confirmed_df = (
        cluster_lf
        .filter(pl.col(ColumnNames.DEFECT_CONFIRMED).is_not_null())
        .with_columns(
            pl.when(pl.col(ColumnNames.DEFECT_CONFIRMED).eq(True))
            .then(pl.lit('결함 있음'))
            .when(pl.col(ColumnNames.DEFECT_CONFIRMED).eq(False))
            .then(pl.lit('결함 없음'))
            .when(pl.col(ColumnNames.DEFECT_CONFIRMED) == 'Unknown')
            .then(pl.lit('알 수 없음'))
            .otherwise(pl.col(ColumnNames.DEFECT_CONFIRMED))
            .alias(ColumnNames.DEFECT_CONFIRMED)
        )
        .group_by(ColumnNames.DEFECT_CONFIRMED)
        .agg(pl.len().alias('count'))
        .sort('count', descending=True)
        .with_columns(
            (pl.col('count') / total_count * 100).round(2).alias('ratio')
        )
        .collect()
    )

    return {
        'harm_summary': event_dict,
        'top_components': top_components_df,
        'total_count': total_count,
        'time_series': time_series_df,
        'defect_types': defect_type_df,
        'defect_confirmed': defect_confirmed_df
    }

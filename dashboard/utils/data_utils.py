# data_utils.py
"""Polars 데이터 처리 유틸리티 함수"""

import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import List, Tuple, Optional
from .constants import ColumnNames, Defaults


def get_year_month_expr(lf: pl.LazyFrame, date_col: str = ColumnNames.DATE_RECEIVED) -> pl.Expr:
    """년-월 컬럼 생성 표현식을 반환 (날짜 타입에 따라 자동 처리)

    Args:
        lf: LazyFrame
        date_col: 날짜 컬럼명

    Returns:
        polars 표현식 (year_month 컬럼)
    """
    try:
        schema = lf.collect_schema()
        date_dtype = None
        for name, dtype in schema.items():
            if name == date_col:
                date_dtype = dtype
                break

        if date_dtype == pl.Date:
            # 이미 Date 타입인 경우
            return (
                pl.col(date_col)
                .dt.strftime(Defaults.DATE_FORMAT)
                .alias("year_month")
            )
        else:
            # 문자열인 경우 (YYYYMMDD 형식)
            return (
                pl.col(date_col)
                .cast(pl.Utf8)
                .str.strptime(pl.Date, format="%Y%m%d", strict=False)
                .dt.strftime(Defaults.DATE_FORMAT)
                .alias("year_month")
            )
    except:
        # 기본값: 문자열로 가정
        return (
            pl.col(date_col)
            .cast(pl.Utf8)
            .str.strptime(pl.Date, format="%Y%m%d", strict=False)
            .dt.strftime(Defaults.DATE_FORMAT)
            .alias("year_month")
        )


def create_manufacturer_product_combo(
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    missing_label: str = Defaults.MISSING_VALUE_LABEL
) -> pl.Expr:
    """제조사-제품군 조합 컬럼 생성 표현식

    Args:
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군 컬럼명
        missing_label: null 값 대체 레이블

    Returns:
        polars 표현식 (manufacturer_product 컬럼)
    """
    return (
        pl.when(pl.col(manufacturer_col).is_not_null() & pl.col(product_col).is_not_null())
        .then(
            pl.col(manufacturer_col).cast(pl.Utf8)
            + pl.lit(" - ")
            + pl.col(product_col).cast(pl.Utf8)
        )
        .otherwise(pl.lit(missing_label))
        .alias("manufacturer_product")
    )


def get_window_dates(
    available_dates: List[str],
    window_size: int = Defaults.WINDOW_SIZE,
    as_of_month: Optional[str] = None,
    include_overlap: bool = False
) -> Tuple[List[str], List[str]]:
    """윈도우 기반 날짜 범위 계산 (최근 k개월과 직전 k개월)

    Args:
        available_dates: 사용 가능한 년-월 리스트 (내림차순 정렬된 것 가정)
        window_size: 윈도우 크기 (1, 3, 6, 12 등 임의의 크기 지원)
        as_of_month: 기준 월 (None이면 가장 최근 월 사용)
        include_overlap: True이면 겹치는 기간 포함 (base가 1개월 앞에서 시작)

    Returns:
        tuple: (recent_months, base_months)

    Examples:
        >>> # window_size=3, include_overlap=False
        >>> # recent: [2024-03, 2024-02, 2024-01]
        >>> # base: [2023-12, 2023-11, 2023-10]

        >>> # window_size=6, include_overlap=True
        >>> # recent: [2024-03, ..., 2023-10]
        >>> # base: [2023-10, ..., 2023-05] (1개월 겹침)
    """
    if not available_dates or len(available_dates) == 0:
        return [], []

    # 기준 월 설정
    if as_of_month is None:
        as_of_month = available_dates[0]  # 가장 최근 월

    # datetime 객체로 변환
    as_of_date = datetime.strptime(as_of_month, Defaults.DATE_FORMAT)

    # 최근 기간 계산 (0부터 window_size-1개월 전까지)
    recent_months = [
        (as_of_date - relativedelta(months=i)).strftime(Defaults.DATE_FORMAT)
        for i in range(window_size)
    ]

    # 비교 기간 계산
    if include_overlap:
        # 겹치는 기간 포함: recent의 마지막 달부터 시작
        base_start = window_size - 1
    else:
        # 겹치지 않는 기간: recent 다음 달부터 시작
        base_start = window_size

    base_months = [
        (as_of_date - relativedelta(months=i)).strftime(Defaults.DATE_FORMAT)
        for i in range(base_start, base_start + window_size)
    ]

    # available_dates에 존재하는 월만 필터링
    recent_months = [m for m in recent_months if m in available_dates]
    base_months = [m for m in base_months if m in available_dates]

    return recent_months, base_months


def apply_basic_filters(
    lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    year_month_expr: Optional[pl.Expr] = None,
    add_combo: bool = True,
    filter_nulls: bool = True,
    custom_filters: Optional[List[pl.Expr]] = None
) -> pl.LazyFrame:
    """기본 필터 적용 (제조사, 제품군, 날짜) + 커스텀 필터 지원

    Args:
        lf: LazyFrame
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군 컬럼명
        date_col: 날짜 컬럼명
        selected_dates: 선택된 년-월 리스트
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품군 리스트
        year_month_expr: 년-월 컬럼 생성 표현식 (재사용)
        add_combo: 제조사-제품군 조합 컬럼 추가 여부
        filter_nulls: null 값 필터링 여부 (add_combo=True일 때만 적용)
        custom_filters: 추가 커스텀 필터 리스트 (예: [pl.col('severity') >= 3])

    Returns:
        필터링된 LazyFrame

    Examples:
        >>> # 커스텀 필터 사용 예시
        >>> apply_basic_filters(
        ...     lf,
        ...     custom_filters=[
        ...         pl.col('event_type') == 'Death',
        ...         pl.col('date_occurred') > '2020-01-01'
        ...     ]
        ... )
    """
    # 조합 컬럼 추가
    if add_combo:
        combo_expr = create_manufacturer_product_combo(manufacturer_col, product_col)
        filtered_lf = lf.with_columns([combo_expr])

        # null 필터 (선택적)
        if filter_nulls:
            filtered_lf = filtered_lf.filter(
                pl.col(manufacturer_col).is_not_null() &
                pl.col(product_col).is_not_null()
            )
    else:
        filtered_lf = lf

    # 날짜 필터
    if selected_dates and len(selected_dates) > 0:
        if year_month_expr is None:
            year_month_expr = get_year_month_expr(lf, date_col)

        filtered_lf = (
            filtered_lf
            .with_columns(year_month_expr)
            .filter(pl.col("year_month").is_in(selected_dates))
        )

    # 제조사 필터
    if selected_manufacturers and len(selected_manufacturers) > 0:
        filtered_lf = filtered_lf.filter(pl.col(manufacturer_col).is_in(selected_manufacturers))

    # 제품군 필터
    if selected_products and len(selected_products) > 0:
        filtered_lf = filtered_lf.filter(pl.col(product_col).is_in(selected_products))

    # 커스텀 필터 적용
    if custom_filters:
        for custom_filter in custom_filters:
            filtered_lf = filtered_lf.filter(custom_filter)

    return filtered_lf

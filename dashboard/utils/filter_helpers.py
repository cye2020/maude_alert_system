# filter_helpers.py
"""필터링 관련 헬퍼 함수"""

import polars as pl
import streamlit as st
from typing import List, Tuple, Optional
from .constants import ColumnNames, Defaults
from .data_utils import get_year_month_expr


@st.cache_data
def get_available_filters(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    _year_month_expr: Optional[pl.Expr] = None
) -> Tuple[List[str], List[str], List[str]]:
    """필터에 사용할 unique 값들을 추출

    Args:
        _lf: LazyFrame (언더스코어로 시작하여 캐싱에서 제외)
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군(제품코드) 컬럼명
        date_col: 날짜 컬럼명
        _year_month_expr: 년-월 컬럼 생성 표현식 (재사용용, 언더스코어로 시작하여 캐싱에서 제외)

    Returns:
        tuple: (available_dates, available_manufacturers, available_products)
    """
    # 년-월 리스트 (재사용 또는 새로 생성)
    year_month_expr = _year_month_expr if _year_month_expr is not None else get_year_month_expr(_lf, date_col)

    try:
        available_dates = (
            _lf
            .filter(pl.col(date_col).is_not_null())
            .with_columns(year_month_expr)
            .select("year_month")
            .filter(pl.col("year_month").is_not_null())
            .unique()
            .sort("year_month", descending=True)
            .collect()
        )["year_month"].to_list()
    except Exception:
        available_dates = []

    # 제조사 리스트
    available_manufacturers = (
        _lf
        .select(pl.col(manufacturer_col))
        .filter(pl.col(manufacturer_col).is_not_null())
        .unique()
        .sort(manufacturer_col)
        .collect()
    )[manufacturer_col].to_list()

    # 제품군 리스트
    available_products = (
        _lf
        .select(pl.col(product_col))
        .filter(pl.col(product_col).is_not_null())
        .unique()
        .sort(product_col)
        .collect()
    )[product_col].to_list()

    return available_dates, available_manufacturers, available_products


@st.cache_data
def get_manufacturers_by_dates(
    _lf: pl.LazyFrame,
    selected_dates: List[str],
    date_col: str = ColumnNames.DATE_RECEIVED,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    _year_month_expr: Optional[pl.Expr] = None
) -> List[str]:
    """선택된 년-월에 존재하는 제조사 목록을 반환

    Args:
        _lf: LazyFrame
        selected_dates: 선택된 년-월 리스트
        date_col: 날짜 컬럼명
        manufacturer_col: 제조사 컬럼명
        _year_month_expr: 년-월 컬럼 생성 표현식 (재사용용, 언더스코어로 시작하여 캐싱에서 제외)

    Returns:
        선택된 년-월에 존재하는 제조사 목록
    """
    if not selected_dates or len(selected_dates) == 0:
        return []

    year_month_expr = _year_month_expr if _year_month_expr is not None else get_year_month_expr(_lf, date_col)

    manufacturers = (
        _lf
        .filter(pl.col(date_col).is_not_null())
        .filter(pl.col(manufacturer_col).is_not_null())
        .with_columns(year_month_expr)
        .filter(pl.col("year_month").is_in(selected_dates))
        .select(pl.col(manufacturer_col))
        .unique()
        .sort(manufacturer_col)
        .collect()
    )[manufacturer_col].to_list()

    return manufacturers


@st.cache_data
def get_products_by_manufacturers(
    _lf: pl.LazyFrame,
    selected_manufacturers: List[str],
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE
) -> List[str]:
    """선택된 제조사에 해당하는 제품군 목록을 반환

    Args:
        _lf: LazyFrame (언더스코어로 시작하여 캐싱에서 제외)
        selected_manufacturers: 선택된 제조사 리스트
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군(제품코드) 컬럼명

    Returns:
        선택된 제조사에 해당하는 제품군 리스트
    """
    if not selected_manufacturers or len(selected_manufacturers) == 0:
        return []

    products = (
        _lf
        .filter(pl.col(manufacturer_col).is_not_null())
        .filter(pl.col(product_col).is_not_null())
        .filter(pl.col(manufacturer_col).is_in(selected_manufacturers))
        .select(pl.col(product_col))
        .unique()
        .sort(product_col)
        .collect()
    )[product_col].to_list()

    return products


@st.cache_data
def get_available_defect_types(
    _lf: pl.LazyFrame,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    date_col: str = ColumnNames.DATE_RECEIVED,
    selected_dates: Optional[List[str]] = None,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    _year_month_expr: Optional[pl.Expr] = None
) -> List[str]:
    """필터링된 데이터에서 사용 가능한 결함 유형 목록 반환

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
        결함 유형 리스트
    """
    year_month_expr = _year_month_expr if _year_month_expr is not None else get_year_month_expr(_lf, date_col)

    filtered_lf = _lf.filter(pl.col(ColumnNames.DEFECT_TYPE).is_not_null())

    # 날짜 필터 적용
    if selected_dates and len(selected_dates) > 0:
        filtered_lf = (
            filtered_lf
            .with_columns(year_month_expr)
            .filter(pl.col("year_month").is_in(selected_dates))
        )

    # 제조사 필터 적용
    if selected_manufacturers and len(selected_manufacturers) > 0:
        filtered_lf = filtered_lf.filter(pl.col(manufacturer_col).is_in(selected_manufacturers))

    # 제품군 필터 적용
    if selected_products and len(selected_products) > 0:
        filtered_lf = filtered_lf.filter(pl.col(product_col).is_in(selected_products))

    defect_types = (
        filtered_lf
        .select(pl.col(ColumnNames.DEFECT_TYPE))
        .unique()
        .sort(ColumnNames.DEFECT_TYPE)
        .collect()
    )[ColumnNames.DEFECT_TYPE].to_list()

    return defect_types


@st.cache_data
def get_devices_by_filters(
    _lf: pl.LazyFrame,
    selected_manufacturers: Optional[List[str]] = None,
    selected_products: Optional[List[str]] = None,
    manufacturer_col: str = ColumnNames.MANUFACTURER,
    product_col: str = ColumnNames.PRODUCT_CODE,
    device_col: str = ColumnNames.UDI_DI
) -> List[str]:
    """선택된 제조사/제품군에 해당하는 기기 목록을 반환 (cascade filter)

    Args:
        _lf: LazyFrame (언더스코어로 시작하여 캐싱에서 제외)
        selected_manufacturers: 선택된 제조사 리스트
        selected_products: 선택된 제품 리스트
        manufacturer_col: 제조사 컬럼명
        product_col: 제품군(제품코드) 컬럼명
        device_col: 기기(UDI-DI) 컬럼명

    Returns:
        선택된 조건에 해당하는 기기 리스트
    """
    filtered_lf = _lf.filter(pl.col(device_col).is_not_null())

    # 제조사 필터 적용
    if selected_manufacturers and len(selected_manufacturers) > 0:
        filtered_lf = filtered_lf.filter(pl.col(manufacturer_col).is_in(selected_manufacturers))

    # 제품군 필터 적용
    if selected_products and len(selected_products) > 0:
        filtered_lf = filtered_lf.filter(pl.col(product_col).is_in(selected_products))

    devices = (
        filtered_lf
        .select(pl.col(device_col))
        .unique()
        .sort(device_col)
        .collect()
    )[device_col].to_list()

    return devices


@st.cache_data
def get_available_clusters(
    _lf: pl.LazyFrame,
    cluster_col: str = ColumnNames.CLUSTER,
    exclude_minus_one: bool = True
) -> List[int]:
    """사용 가능한 클러스터 목록 반환

    Args:
        _lf: LazyFrame
        cluster_col: 클러스터 컬럼명
        exclude_minus_one: -1 클러스터 제외 여부

    Returns:
        클러스터 번호 리스트
    """
    filtered_lf = _lf.filter(pl.col(cluster_col).is_not_null())

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


def apply_common_filters(
    lf: pl.LazyFrame,
    manufacturers: list = None,
    products: list = None,
    devices: list = None,
    defect_types: list = None,
    clusters: list = None
) -> pl.LazyFrame:
    """공통 필터를 LazyFrame에 적용

    Args:
        lf: 원본 LazyFrame
        manufacturers: 제조사 필터 (빈 리스트면 전체)
        products: 제품군 필터 (빈 리스트면 전체)
        devices: 기기 필터 (빈 리스트면 전체)
        defect_types: 결함 유형 필터 (빈 리스트면 전체)
        clusters: 클러스터 필터 (빈 리스트면 전체)

    Returns:
        필터링된 LazyFrame
    """
    filtered_lf = lf

    # 제조사 필터
    if manufacturers and len(manufacturers) > 0:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.MANUFACTURER).is_in(manufacturers))

    # 제품군 필터
    if products and len(products) > 0:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.PRODUCT_CODE).is_in(products))

    # 기기 필터
    if devices and len(devices) > 0:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.UDI_DI).is_in(devices))

    # 결함 유형 필터
    if defect_types and len(defect_types) > 0:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.DEFECT_TYPE).is_in(defect_types))

    # 클러스터 필터
    if clusters and len(clusters) > 0:
        filtered_lf = filtered_lf.filter(pl.col(ColumnNames.CLUSTER).is_in(clusters))

    return filtered_lf

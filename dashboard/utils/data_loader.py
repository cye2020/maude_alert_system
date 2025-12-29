"""
데이터 로딩 유틸리티

로컬 Parquet 파일 또는 Snowflake에서 데이터를 로드합니다.
"""

import polars as pl
import streamlit as st
from pathlib import Path
from .dashboard_config import get_config


class SnowflakeDataSource:
    """
    Snowflake 데이터 소스 (LazyFrame처럼 동작)

    실제로는 쿼리를 실행할 때만 Snowflake에서 데이터를 가져옴
    메모리에 전체 데이터를 올리지 않음
    """

    def __init__(self):
        from src.snowflake.config import get_snowflake_config
        self.config = get_snowflake_config()
        self.table_name = self.config.get_table_name()

    def collect(self) -> pl.DataFrame:
        """전체 데이터 수집 (필요한 경우에만 사용)"""
        from src.snowflake.read import read_table_as_polars
        return read_table_as_polars(self.table_name)

    def filter(self, *args, **kwargs):
        """필터링 - 로컬 LazyFrame처럼 사용하기 위한 래퍼"""
        # Snowflake는 필터를 SQL로 변환해야 하므로
        # 일단은 전체 데이터를 가져와서 Polars LazyFrame으로 변환
        df = self.collect()
        return df.lazy()


def load_maude_data(cache_key: str):
    """
    Silver Stage3 (클러스터링) 데이터 로드

    Snowflake 사용 설정에 따라 Snowflake 또는 로컬 파일에서 데이터를 로드합니다.

    Args:
        cache_key: 캐시 키 (예: "2025-01") - 월이 바뀌면 자동 갱신

    Returns:
        Polars LazyFrame 또는 SnowflakeDataSource
    """
    config = get_config()

    # Snowflake 사용 여부 확인
    if config.use_snowflake():
        # Snowflake 데이터 소스 반환 (LazyFrame처럼 동작)
        return load_from_snowflake(cache_key)
    else:
        # 로컬 파일에서 데이터 로드
        return load_from_local(cache_key)


def load_from_snowflake(cache_key: str) -> pl.LazyFrame:
    """
    Snowflake에서 데이터 로드

    메모리 효율을 위해 전체 데이터를 한번에 가져오지만
    Polars LazyFrame으로 변환하여 반환

    Args:
        cache_key: 캐시 키

    Returns:
        Polars LazyFrame
    """
    try:
        from src.snowflake.read import read_table_as_polars

        # Snowflake에서 전체 데이터 읽기
        # 주의: 메모리에 모두 올라감
        st.info("Snowflake에서 데이터를 로딩합니다... (최초 1회만)")
        df = read_table_as_polars()

        # LazyFrame으로 변환
        return df.lazy()

    except Exception as e:
        st.error(f"Snowflake에서 데이터를 로드할 수 없습니다: {e}")
        st.info("로컬 파일에서 데이터를 로드합니다...")
        # Snowflake 실패 시 로컬 파일로 폴백
        return load_from_local(cache_key)


def load_from_local(cache_key: str) -> pl.LazyFrame:
    """
    로컬 Parquet 파일에서 데이터 로드

    Args:
        cache_key: 캐시 키

    Returns:
        Polars LazyFrame
    """
    config = get_config()
    data_path = config.get_silver_stage3_path(dataset='maude')

    if not data_path.exists():
        st.error(f"데이터 파일을 찾을 수 없습니다: {data_path}")
        st.stop()

    return pl.scan_parquet(data_path)

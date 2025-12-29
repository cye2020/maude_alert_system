"""
Snowflake 데이터 읽기 모듈

Snowflake 테이블에서 데이터를 읽어 Polars DataFrame으로 반환합니다.
"""

import logging
from typing import Optional
import polars as pl
import snowflake.connector
from snowflake.connector import DictCursor

from src.snowflake.config import get_snowflake_config

logger = logging.getLogger(__name__)


def read_table_as_polars(
    table_name: Optional[str] = None,
    limit: Optional[int] = None
) -> pl.DataFrame:
    """
    Snowflake 테이블을 Polars DataFrame으로 읽어옵니다.

    Args:
        table_name: 테이블명 (None이면 config에서 가져옴)
        limit: 읽어올 행 수 제한 (None이면 전체)

    Returns:
        Polars DataFrame
    """
    config = get_snowflake_config()
    conn_params = config.get_connection_params()

    if table_name is None:
        table_name = config.get_table_name()

    logger.info(f"Snowflake에서 테이블 읽기: {table_name}")

    # Snowflake 연결
    conn = snowflake.connector.connect(**conn_params)

    try:
        # SQL 쿼리 생성
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"쿼리 실행: {query}")

        # 데이터 읽기
        cursor = conn.cursor()
        cursor.execute(query)

        # 결과를 Polars DataFrame으로 변환
        # fetch_pandas_all()로 Pandas DataFrame을 가져온 후 Polars로 변환
        df_pandas = cursor.fetch_pandas_all()
        df_polars = pl.from_pandas(df_pandas)

        logger.info(f"데이터 로드 완료: {len(df_polars):,} rows, {len(df_polars.columns)} columns")

        cursor.close()
        return df_polars

    finally:
        conn.close()


def read_table_as_lazy(
    table_name: Optional[str] = None,
    limit: Optional[int] = None
) -> pl.LazyFrame:
    """
    Snowflake 테이블을 Polars LazyFrame으로 읽어옵니다.

    Args:
        table_name: 테이블명 (None이면 config에서 가져옴)
        limit: 읽어올 행 수 제한 (None이면 전체)

    Returns:
        Polars LazyFrame
    """
    df = read_table_as_polars(table_name, limit)
    return df.lazy()


def main():
    """테스트용 메인 함수"""
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Snowflake 데이터 읽기 테스트")
    print("=" * 60)

    # 1. 샘플 데이터 읽기 (10행)
    print("\n1. 샘플 데이터 읽기 (10행)...")
    df_sample = read_table_as_polars(limit=10)
    print(f"\n✓ 데이터 로드 완료: {len(df_sample)} rows")
    print(f"\n컬럼 목록 ({len(df_sample.columns)}개):")
    for i, col in enumerate(df_sample.columns, 1):
        print(f"  {i}. {col}")

    print("\n데이터 미리보기:")
    print(df_sample.head())

    # 2. 전체 데이터 통계
    print("\n" + "=" * 60)
    print("2. 전체 데이터 통계...")
    df_full = read_table_as_polars()
    print(f"\n✓ 전체 데이터: {len(df_full):,} rows × {len(df_full.columns)} columns")

    # 3. 기본 통계
    print("\n기본 통계:")
    print(df_full.describe())

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

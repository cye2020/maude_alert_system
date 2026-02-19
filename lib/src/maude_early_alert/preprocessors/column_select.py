"""
컬럼 선택 SQL 생성 모듈
컬럼 리스트 → SELECT SQL 생성
"""
from typing import List

import structlog

from maude_early_alert.utils.sql_builder import build_cte_sql


logger: structlog.stdlib.BoundLogger = structlog.get_logger()


def build_select_columns_sql(cols: List[str], table_name: str) -> str:
    """
    SELECT SQL 생성

    Args:
        cols: 선택할 컬럼 리스트
        table_name: 테이블명

    Returns:
        생성된 SELECT SQL 문자열

    Examples:
        >>> sql = build_select_columns_sql(['col1', 'col2'], "MAUDE.SILVER.EVENT")
    """
    if not cols:
        logger.debug("선택할 컬럼이 없습니다. SELECT * 반환")
        return build_cte_sql(ctes=[], from_clause=table_name)

    select_cols = [
        f"{col['name']} AS {col['alias']}"
        for col in cols
    ]

    sql = build_cte_sql(
        ctes=[],
        from_clause=table_name,
        select_cols=select_cols,
        distinct=True
    )

    logger.debug("SELECT SQL 생성 완료", table_name=table_name, column_count=len(cols))

    return sql


if __name__ == '__main__':
    from maude_early_alert.logging_config import configure_logging
    configure_logging()

    print("=" * 80)
    print("컬럼 필터링 SQL 테스트")
    print("=" * 80)

    pass


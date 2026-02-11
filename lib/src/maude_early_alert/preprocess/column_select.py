"""
컬럼 선택 SQL 생성 모듈
컬럼 리스트 → SELECT SQL 생성
"""
from typing import List

import structlog

from maude_early_alert.utils.sql_builder import build_cte_sql


logger = structlog.get_logger()


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
    from maude_early_alert.utils.config_loader import load_config
    
    configure_logging()

    print("=" * 80)
    print("컬럼 필터링 SQL 테스트")
    print("=" * 80)

    columns_config = load_config('preprocess/columns')
    event_columns = columns_config['event']['cols']
    udi_columns = columns_config['udi']['cols']

    sql_maude = build_select_columns_sql(event_columns, "MAUDE.SILVER.EVENT_STAGE_02")
    print("\n[MAUDE SQL]")
    print(sql_maude)

    sql_udi = build_select_columns_sql(udi_columns, "MAUDE.SILVER.UDI_STAGE_02")
    print("\n[UDI SQL]")
    print(sql_udi)
    
    final_event_columns = [
        {**d, 'name': d['alias']}
        for d in event_columns
        if d['final']
    ]
    
    print(final_event_columns)
    
    sql_final_event = build_select_columns_sql(final_event_columns, "EVENT_STAGE_11")
    print('\nFINAL EVENT SQL')
    print(sql_final_event)
    
    

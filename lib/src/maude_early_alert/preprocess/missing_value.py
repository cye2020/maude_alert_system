"""
Missing Value 처리 SQL 생성 모듈
그룹별 최빈값(MODE)으로 NULL 값 대체하는 SELECT SQL 생성
"""
from textwrap import dedent
from typing import Dict, List

import structlog

from maude_early_alert.utils import build_cte_sql


logger = structlog.get_logger()


def build_mode_fill_sql(
    group_to_target: Dict[str, str],
    table_name: str,
    table_alias: str
) -> str:
    """
    그룹별 최빈값으로 NULL 대체 SQL 생성

    Args:
        group_to_target: {그룹_컬럼: 대상_컬럼} 딕셔너리
            예: {'product_code': 'device_name', 'postal_code': 'manufacturer_name'}
        table_name: 테이블명
        other_cols: NULL 처리 안 하고 그냥 가져올 컬럼들

    Returns:
        생성된 SELECT SQL 문자열

    Examples:
        >>> sql = build_mode_fill_sql(
        ...     group_to_target={
        ...         'product_code': 'device_name',
        ...         'postal_code': 'manufacturer_name'
        ...     },
        ...     table_name="MAUDE.SILVER.EVENT",
        ...     other_cols=['mdr_report_key', 'date_received']
        ... )
    """
    if not group_to_target:
        raise ValueError("group_to_target 딕셔너리가 비어있습니다")

    # CTE 생성
    ctes = []
    for i, (group_col, target_col) in enumerate(group_to_target.items()):
        cte_name = f"mode_{i+1}"
        query = dedent(f"""\
            SELECT
                {group_col},
                MODE({target_col}) AS mode_{target_col}
            FROM
                {table_name}
            WHERE
                {group_col} IS NOT NULL
                AND {target_col} IS NOT NULL
            GROUP BY
                {group_col}""")
        ctes.append({'name': cte_name, 'query': query})

    # SELECT 컬럼 생성
    replace_cols = []
    
    for i, (group_col, target_col) in enumerate(group_to_target.items()):
        cte_name = f"mode_{i+1}"
        replace_cols.append(
            f"COALESCE({table_alias}.{target_col}, {cte_name}.mode_{target_col}) AS {target_col}"
        )

    # JOIN 생성
    joins = [
        f"LEFT JOIN mode_{i+1} ON {table_alias}.{group_col} = mode_{i+1}.{group_col}"
        for i, group_col in enumerate(group_to_target.keys())
    ]

    sql = build_cte_sql(
        ctes=ctes,
        table_alias=table_alias,
        replace_cols=replace_cols,
        from_clause=f"{table_name} AS {table_alias}",
        joins=joins,
    )

    logger.debug("MODE fill SQL 생성 완료", mapping_count=len(group_to_target))

    return sql


if __name__ == '__main__':
    from maude_early_alert.logging_config import configure_logging
    
    configure_logging()
    
    print("=" * 80)
    print("Missing Value 처리 SQL 테스트")
    print("=" * 80)

    sql = build_mode_fill_sql(
        group_to_target={
            'PRODUCT_CODE': 'PRODUCT_NAME',
            'MANUFACTURER_POSTAL_CODE': 'MANUFACTURER_NAME'
        },
        table_name="MAUDE.SILVER.EVENT_STAGE_04",
        table_alias='t'
    )
    print(sql)
"""
타입 변환 SQL 생성 모듈
"""
from typing import List, Dict
import structlog

from maude_early_alert.utils.sql_builder import build_cte_sql

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def build_type_cast_sql(
    columns: List[Dict],
    input_table: str,
    table_alias: str = "t"
) -> str:
    """타입 변환 SQL 생성

    Args:
        columns: 컬럼 정보 리스트
        input_table: 입력 테이블명
        table_alias: 테이블 alias

    Returns:
        생성된 SQL 문자열
    """
    if not columns:
        logger.warning("No columns to cast")
        return f"SELECT * FROM {input_table}"

    select_exprs = [
        _build_cast_expr(col, table_alias)
        for col in columns
    ]
    sql = build_cte_sql(
        ctes=[],
        from_clause=f"{input_table} {table_alias}",
        select_cols=select_exprs,
    )

    logger.debug("Type cast SQL generated", column_count=len(columns))
    return sql


def _build_cast_expr(col: Dict, table_alias: str) -> str:
    """단일 컬럼 CAST 표현식 생성"""
    col_name = col.get("alias") or col["name"]
    col_type = col.get("type", "VARCHAR").upper()
    q = f'"{col_name}"'

    if col_type == "DATE":
        sql =  (
            f"COALESCE("
            f"\n    TRY_TO_DATE({table_alias}.{q}, 'YYYYMMDD'), "
            f"\n    TRY_TO_DATE({table_alias}.{q}, 'YYYY-MM-DD')"
            f"\n) AS {q}"
        )
        return sql

    if col_type in ["INTEGER", "INT"]:
        return f"TRY_TO_NUMBER({table_alias}.{q})::INTEGER AS {q}"

    if col_type == "BOOLEAN":
        true_val, false_val = _parse_boolean_mapping(col)
        return (
            f"CASE WHEN {table_alias}.{q} = '{true_val}' THEN TRUE "
            f"WHEN {table_alias}.{q} = '{false_val}' THEN FALSE "
            f"ELSE NULL END AS {q}"
        )
    
    # 기본
    return f"{table_alias}.{q} AS {q}"


def _parse_boolean_mapping(col: Dict) -> tuple:
    """YAML에서 boolean true/false 매핑 파싱"""
    true_val = col.get("true_value")
    false_val = col.get("false_value")
    if not true_val or not false_val:
        raise ValueError(
            f"BOOLEAN 컬럼 '{col.get('name')}'에 true_value/false_value가 필요합니다"
        )
    return true_val, false_val


if __name__ == "__main__":
    from maude_early_alert.logging_config import configure_logging
    from maude_early_alert.utils.config_loader import load_config

    configure_logging()

    print("=" * 80)
    print("타입 변환 SQL 테스트")
    print("=" * 80)

    columns_config = load_config('preprocess/columns')
    event_columns = columns_config['event']['cols']
    udi_columns = columns_config['udi']['cols']

    sql = build_type_cast_sql(
        columns=event_columns,
        input_table="EVENT_STAGE_07",
    )
    print(sql)
    
    sql = build_type_cast_sql(
        columns=udi_columns,
        input_table="UDI_STAGE_05",
    )
    print(sql)


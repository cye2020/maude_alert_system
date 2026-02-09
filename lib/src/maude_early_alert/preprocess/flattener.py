"""Snowflake Flatten SQL 생성"""

from ast import Dict
from pathlib import Path
from typing import List, Dict, Union
import sys


# ============================================================
# SQL 생성 함수 (순수 함수 - Snowflake 연결 불필요)
# ============================================================

def sanitize(name: str) -> str:
    """컬럼명을 SQL 안전 이름으로 변환"""
    return name.replace(":", "_").replace("-", "_").replace(".", "_").replace(" ", "_")


def _flatten_to_entries(sub_keys) -> list:
    """nested dict/list를 (json_path, dtype) 리스트로 평탄화"""
    if isinstance(sub_keys, dict):
        stack = [(k, sub_keys[k]) for k in sorted(sub_keys.keys())]
    else:
        stack = [(k, "VARCHAR") for k in sorted(sub_keys)]

    entries = []
    while stack:
        path, val = stack.pop(0)
        if isinstance(val, dict):
            for ck in sorted(val.keys()):
                stack.append((f"{path}:{ck}", val[ck]))
        else:
            entries.append((path, (val or "VARCHAR").upper()))
    return entries


def build_flatten_sql(
    table_name: str,
    raw_column: str = "raw_data",
    scalar_keys: Dict[str, str] = None,
    first_element_keys: Dict[str, Union[List[str], Dict[str, str]]] = None,
    transform_keys: Dict[str, List[str]] = None,
    flatten_keys: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = None,
    flatten_outer: bool = True,
) -> str:
    """Snowflake Flatten SQL 생성

    Args:
        table_name: 테이블명 (예: MAUDE.BRONZE.EVENT)
        raw_column: JSON 컬럼명
        scalar_keys: 최상위 키들 {key: dtype} (ARRAY 포함, 전략 미지정 키들)
        first_element_keys: 배열의 첫 번째 요소만 선택 (예: patient[0]).
                           {"patient": ["key1", "key2"]} 또는
                           {"patient": {"key1": "VARCHAR", "key2": "ARRAY"}}
        transform_keys: 배열을 TRANSFORM으로 리스트화 (예: mdr_text).
                       {"mdr_text": ["text", "text_type_code"]}
        flatten_keys: LATERAL FLATTEN으로 행 분리 (예: device).
                     값이 dict이면 nested object로 간주하여 하위 키를 펼침.
                     {"device": {"brand_name": "VARCHAR", "openfda": {"device_name": "VARCHAR"}}}
        flatten_outer: LATERAL FLATTEN 시 OUTER JOIN 사용 여부

    Returns:
        SELECT SQL 문자열
    """
    sections = []
    from_parts = [f"FROM {table_name}"]

    # 1) 최상위 키 (전략 미지정 - 캐스팅 없이 VARIANT 그대로)
    if scalar_keys:
        scalar_cols = sorted(
            f"    {raw_column}:{key}::{t} AS {sanitize(key)}"
            for key, t in scalar_keys.items()
        )
        sections.append(
            "    -- ================================================\n"
            "    -- Top-level Columns\n"
            "    -- ================================================\n"
            + ",\n".join(scalar_cols)
        )

    # 2) first_element: 배열의 첫 번째 요소 선택 (array[0])
    if first_element_keys:
        for array_name, sub_keys in first_element_keys.items():
            entries = _flatten_to_entries(sub_keys)
            cols = []
            for path, dtype in entries:
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                if dtype == "ARRAY":
                    cols.append(
                        f"    TRANSFORM({raw_column}:{array_name}[0]:{path}, x -> x::STRING)"
                        f" AS {col_alias}"
                    )
                else:
                    cols.append(
                        f"    {raw_column}:{array_name}[0]:{path}::STRING AS {col_alias}"
                    )
            if cols:
                sections.append(
                    f"\n    -- ================================================\n"
                    f"    -- {array_name} (first element [0])\n"
                    f"    -- ================================================\n"
                    + ",\n".join(cols)
                )

    # 3) transform: 배열을 TRANSFORM으로 리스트화
    if transform_keys:
        for array_name, sub_keys in transform_keys.items():
            entries = _flatten_to_entries(sub_keys)
            cols = []
            for path, dtype in entries:
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                cols.append(
                    f"    TRANSFORM({raw_column}:{array_name}, x -> x:{path}::STRING)"
                    f" AS {col_alias}"
                )
            if cols:
                sections.append(
                    f"\n    -- ================================================\n"
                    f"    -- {array_name} (TRANSFORM)\n"
                    f"    -- ================================================\n"
                    + ",\n".join(cols)
                )

    # 4) flatten: LATERAL FLATTEN으로 행 분리
    if flatten_keys:
        outer = "TRUE" if flatten_outer else "FALSE"
        for i, (array_name, sub_keys) in enumerate(flatten_keys.items()):
            alias = f"f{i}"
            entries = _flatten_to_entries(sub_keys)
            cols = []
            for path, dtype in entries:
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                if dtype == "ARRAY":
                    cols.append(
                        f"    TRANSFORM({alias}.value:{path}, x -> x::STRING) AS {col_alias}"
                    )
                else:
                    cols.append(
                        f"    {alias}.value:{path}::STRING AS {col_alias}"
                    )

            if cols:
                sections.append(
                    f"\n    -- ================================================\n"
                    f"    -- {array_name} (LATERAL FLATTEN)\n"
                    f"    -- ================================================\n"
                    + ",\n".join(cols)
                )

            from_parts.append(
                f"    , LATERAL FLATTEN(input => {raw_column}:{array_name}, OUTER => {outer}) AS {alias}"
            )

    # SQL 조립
    from_clause = "\n".join(from_parts)
    select_body = ",\n".join(sections)

    return f"""-- ====================================================================
-- Snowflake Flatten SQL
-- Table: {table_name}
-- ====================================================================

SELECT
{select_body}
{from_clause};
"""


# ============================================================
# Snowflake 스키마 조회 (snowflake_load 활용)
# ============================================================

def fetch_schema_and_build_sql(
    table_name: str,
    raw_column: str = "raw_data",
    flatten: List[str] = None,
    transform: List[str] = None,
    first_element: List[str] = None,
) -> str:
    """Snowflake에서 스키마 조회 후 SQL 생성

    Args:
        table_name: 테이블명
        raw_column: JSON 컬럼명
        flatten: LATERAL FLATTEN 전략을 적용할 배열 이름 목록 (예: ["device"])
        transform: TRANSFORM 전략을 적용할 배열 이름 목록 (예: ["mdr_text"])
        first_element: 첫 번째 요소([0]) 전략을 적용할 배열 이름 목록 (예: ["patient"])

    Returns:
        생성된 SQL 문자열
    """
    # snowflake_load 임포트
    try:
        from maude_early_alert.loading.snowflake_load import SnowflakeLoader
        import snowflake.connector
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
        
    from maude_early_alert.utils.secrets import get_secret
    # Snowflake 연결
    try:
        secret = get_secret('snowflake/bronze/credentials')
        conn = snowflake.connector.connect(
            user=secret['user'],
            password=secret['password'],
            account=secret['account'],
            warehouse=secret['warehouse'],
            database=secret['database'],
            schema=secret['schema']
        )
        cursor = conn.cursor()
    except Exception as e:
        raise
    
    # 스키마 조회: top-level 키 + 각 배열의 nested 구조를 한 번에
    top = SnowflakeLoader.top_keys_with_type(cursor, table_name, raw_column)

    array_names = [k for k, t in top.items() if t.upper() == 'ARRAY']
    array_schemas = {
        name: SnowflakeLoader.array_keys_with_type(cursor, table_name, name, raw_column)
        for name in array_names
    }

    cursor.close()
    conn.close()

    # 전략 매핑 (인자로 받은 배열 이름 → 전략)
    flatten_keys = {k: array_schemas[k] for k in (flatten or []) if k in array_schemas}
    transform_keys = {k: array_schemas[k] for k in (transform or []) if k in array_schemas}
    first_element_keys = {k: array_schemas[k] for k in (first_element or []) if k in array_schemas}

    # 전략이 명시된 배열만 제외 (나머지는 타입 그대로 유지)
    exclude = set((flatten or []) + (transform or []) + (first_element or []))
    scalar_keys = {k: t.upper() for k, t in top.items() if k not in exclude}

    return build_flatten_sql(
        table_name=table_name,
        raw_column=raw_column,
        scalar_keys=scalar_keys,
        first_element_keys=first_element_keys,
        transform_keys=transform_keys,
        flatten_keys=flatten_keys,
    )


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    event_table_name = "MAUDE.BRONZE.EVENT"
    udi_table_name = "MAUDE.BRONZE.UDI"
    RAW_COLUMN = "raw_data"
    
    sql = fetch_schema_and_build_sql(
        event_table_name, RAW_COLUMN,
        flatten=["device"],
        transform=["mdr_text"],
        first_element=["patient"],
    )
    print(sql)
    
    sql = fetch_schema_and_build_sql(
        udi_table_name, RAW_COLUMN,
        flatten=["identifiers"],
        transform=['product_codes']
    )
    print(sql)
    
    
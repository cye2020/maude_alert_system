"""Snowflake Flatten SQL 생성"""

import re
from typing import List, Dict, Union

from maude_early_alert.utils.sql_builder import build_cte_sql


# ============================================================
# 스키마 조회용 SQL 빌더 (순수 함수)
# ============================================================

def build_top_keys_sql(table_name: str, raw_column: str = "raw_data") -> str:
    """최상위 키와 타입 조회 SQL 생성"""
    return f"""
    SELECT DISTINCT f.key, TYPEOF(f.value)
    FROM {table_name},
        LATERAL FLATTEN(input => {raw_column}) AS f
    ORDER BY f.key;
    """


def build_array_keys_sql(table_name: str, array_path: str, raw_column: str = "raw_data") -> str:
    """배열 내 모든 키를 RECURSIVE로 조회하는 SQL 생성"""
    return f"""
    SELECT DISTINCT f.path::STRING, f.key::STRING, TYPEOF(f.value)
    FROM {table_name},
        LATERAL FLATTEN(input => {raw_column}:{array_path}, RECURSIVE => TRUE) f
    WHERE f.key IS NOT NULL
      AND TYPEOF(f.value) != 'OBJECT'
    ORDER BY 1, 2
    """


def parse_array_keys_result(rows: List[tuple]) -> Dict[str, Union[str, dict]]:
    """array_keys SQL 결과를 nested dict로 변환

    Args:
        rows: [(path, key, typeof_value), ...] 형태의 결과 행

    Returns:
        nested dict (예: {"brand_name": "VARCHAR", "openfda": {"device_name": "VARCHAR"}})
    """
    result = {}
    for raw_path, key, typ in rows:
        inner = re.sub(r'^\[\d+\]\.?', '', str(raw_path))
        if re.search(r'\[\d+\]', inner):
            continue

        clean = re.sub(r'\[\d+\]\.?', '', str(raw_path)).strip('.')
        parts = [p for p in clean.split('.') if p]

        parent_parts = parts[:-1] if parts else []

        node = result
        for part in parent_parts:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[key] = typ

    return result


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
    select_cols = []
    lateral_parts = []

    if scalar_keys:
        select_cols.extend(sorted(
            f"{raw_column}:{key}::{t} AS {sanitize(key)}"
            for key, t in scalar_keys.items()
        ))

    if first_element_keys:
        for array_name, sub_keys in first_element_keys.items():
            for path, dtype in _flatten_to_entries(sub_keys):
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                if dtype == "ARRAY":
                    select_cols.append(
                        f"TRANSFORM({raw_column}:{array_name}[0]:{path}, x -> x::STRING)"
                        f" AS {col_alias}"
                    )
                else:
                    select_cols.append(
                        f"{raw_column}:{array_name}[0]:{path}::STRING AS {col_alias}"
                    )

    if transform_keys:
        for array_name, sub_keys in transform_keys.items():
            for path, dtype in _flatten_to_entries(sub_keys):
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                select_cols.append(
                    f"TRANSFORM({raw_column}:{array_name}, x -> x:{path}::STRING)"
                    f" AS {col_alias}"
                )

    if flatten_keys:
        outer = "TRUE" if flatten_outer else "FALSE"
        for i, (array_name, sub_keys) in enumerate(flatten_keys.items()):
            alias = f"f{i}"
            for path, dtype in _flatten_to_entries(sub_keys):
                col_alias = f"{sanitize(array_name)}_{sanitize(path)}"
                if dtype == "ARRAY":
                    select_cols.append(
                        f"TRANSFORM({alias}.value:{path}, x -> x::STRING) AS {col_alias}"
                    )
                else:
                    select_cols.append(
                        f"{alias}.value:{path}::STRING AS {col_alias}"
                    )
            lateral_parts.append(
                f", LATERAL FLATTEN(input => {raw_column}:{array_name}, OUTER => {outer}) AS {alias}"
            )

    from_clause = table_name
    if lateral_parts:
        from_clause += "\n" + "\n".join(lateral_parts)

    return build_cte_sql(
        ctes=[],
        from_clause=from_clause,
        select_cols=select_cols,
    )

# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    print("=== build_top_keys_sql ===")
    print(build_top_keys_sql("my_table"))

    print("\n=== build_array_keys_sql ===")
    print(build_array_keys_sql("my_table", "items"))

    print("\n=== parse_array_keys_result ===")
    rows = [
        ("[0].name", "name", "VARCHAR"),
        ("[0].meta.code", "code", "VARCHAR"),
        ("[0][1].skip", "skip", "VARCHAR"),  # 중첩 배열 → 제외
    ]
    print(parse_array_keys_result(rows))
    # 예상: {'name': 'VARCHAR', 'meta': {'code': 'VARCHAR'}}

    print("\n=== build_flatten_sql ===")
    sql = build_flatten_sql(
        table_name="my_table",
        scalar_keys={"id": "VARCHAR", "created_at": "DATE"},
        first_element_keys={"patient": {"age": "VARCHAR", "gender": "VARCHAR"}},
        transform_keys={"notes": {"text": "VARCHAR", "type": "VARCHAR"}},
        flatten_keys={"devices": {"brand": "VARCHAR", "info": {"model": "VARCHAR"}}},
    )
    print(sql)
# ======================
# 표준 라이브러리
# ======================
import json
from pathlib import Path
from textwrap import dedent, indent
from typing import Any, Dict, List, Optional, Union

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.utils.helpers import validate_identifier
from maude_early_alert.utils.sql_builder import build_cte_sql, build_insert_sql, build_join_clause

_INDENT = "    "


# ============================================================================
# MDR 텍스트 추출 SQL
# ============================================================================

def build_mdr_text_extract_sql(
    table_name: str,
    columns: Optional[List[str]] = None,
    where: Optional[str] = None,
    limit: Optional[int] = None,
) -> str:
    """대상 테이블에서 MDR_TEXT 추출용 SELECT SQL 생성.

    Args:
        table_name: 대상 테이블 (FQDN 가능, 예: MAUDE.SILVER.EVENT_STAGE_12)
        columns: SELECT 컬럼 목록 (None이면 MDR_TEXT 만)
        where: WHERE 절 (선택, 예: "MDR_TEXT IS NOT NULL")
        limit: LIMIT 값 (None이면 미포함)

    Returns:
        실행 가능한 SELECT SQL 문자열
    """
    validate_identifier(table_name)
    select_cols = columns if columns is not None else ["MDR_TEXT"]
    for c in select_cols:
        validate_identifier(c)

    select_clause = ", ".join(select_cols)
    sql = f"SELECT {select_clause}\nFROM {table_name}"
    if where:
        sql += f"\nWHERE {dedent(where).strip()}"
    if limit is not None:
        sql += f"\nLIMIT {limit}"
    return sql


# ============================================================================
# 추출 결과 테이블 SQL 빌더 (loaders/snowflake_load.py에서 이동, 파라미터화)
# ============================================================================

def build_ensure_extracted_table_sql(
    table_name: str,
    columns: List[dict],
) -> str:
    """추출 결과 전용 테이블 CREATE IF NOT EXISTS SQL 생성.

    Args:
        table_name: 테이블명
        columns: [{'name': ..., 'type': ..., 'primary_key': True/False}, ...]
    """
    col_defs = []
    pk_cols = []
    for col in columns:
        not_null = " NOT NULL" if col.get('primary_key') else ""
        col_defs.append(f"{col['name']} {col['type']}{not_null}")
        if col.get('primary_key'):
            pk_cols.append(col['name'])

    col_clause = indent(",\n".join(col_defs), _INDENT)
    pk_clause = ""
    if pk_cols:
        pk_clause = f",\n{_INDENT}PRIMARY KEY ({', '.join(pk_cols)})"

    return f"CREATE TABLE IF NOT EXISTS {table_name} (\n{col_clause}{pk_clause}\n)"


def build_create_extract_temp_sql(
    table_name: str,
    columns: List[dict],
) -> str:
    """추출 결과용 임시 스테이징 테이블 CREATE SQL 생성."""
    col_defs = [f"{col['name']} {col['type']}" for col in columns]
    col_clause = indent(",\n".join(col_defs), _INDENT)
    return f"CREATE TEMPORARY TABLE {table_name} (\n{col_clause}\n)"


def build_extract_stage_insert_sql(
    table_name: str,
    columns: List[dict],
) -> str:
    """추출 결과 임시 테이블 INSERT SQL 생성 (executemany용)."""
    col_names = [col['name'] for col in columns]
    return build_insert_sql(
        table_name=table_name,
        columns=col_names,
        num_rows=1,
    )


def build_extract_merge_sql(
    target: str,
    source: str,
    pk_column: str,
    non_pk_columns: List[str],
) -> str:
    """추출 결과 MERGE SQL 생성.

    Args:
        target: 대상 테이블명
        source: 소스 (임시) 테이블명
        pk_column: PRIMARY KEY 컬럼명
        non_pk_columns: PK 외 컬럼명 목록
    """
    update_set = ",\n        ".join(
        f"target.{col} = source.{col}" for col in non_pk_columns
    )
    all_cols = [pk_column] + non_pk_columns
    insert_cols = ", ".join(all_cols)
    insert_vals = ", ".join(f"source.{col}" for col in all_cols)

    return dedent(f"""\
        MERGE INTO {target} AS target
        USING {source} AS source
        ON target.{pk_column} = source.{pk_column}
        WHEN MATCHED THEN UPDATE SET
            {update_set}
        WHEN NOT MATCHED THEN INSERT (
            {insert_cols}
        ) VALUES (
            {insert_vals}
        )""")


def build_extracted_join_sql(
    base_table: str,
    extracted_table: str,
    non_pk_columns: List[str],
    pk_column: str = "MDR_TEXT",
    base_alias: str = "e",
    extract_alias: str = "ex",
) -> str:
    """원본 테이블과 추출 결과 테이블을 LEFT JOIN하는 SELECT SQL 생성.

    Args:
        base_table: 원본 테이블명 (e.g. 'DB.SCHEMA.EVENT_CURRENT')
        extracted_table: 추출 결과 테이블명 (e.g. 'DB.SCHEMA.EVENT_EXTRACTED')
        non_pk_columns: 추출 테이블에서 가져올 컬럼 목록 (PK 제외)
        pk_column: JOIN 키 컬럼
        base_alias: 원본 테이블 alias
        extract_alias: 추출 테이블 alias
    """
    extract_cols = [f"{extract_alias}.{col}" for col in non_pk_columns]
    select_columns = [f"{base_alias}.*"] + extract_cols

    join_clause = build_join_clause(
        left_table=base_table,
        right_table=extracted_table,
        on_columns=pk_column,
        join_type="LEFT",
        left_alias=base_alias,
        right_alias=extract_alias,
    )

    return build_cte_sql(
        ctes=[],
        from_clause=f"{base_table} {base_alias}",
        select_cols=select_columns,
        joins=[join_clause],
    )


# ============================================================================
# 추출 결과 데이터 변환
# ============================================================================

# vLLM 결과 키 → 추출 결과 컬럼 매핑
_RESULT_KEY_MAP = {
    "MDR_TEXT":           lambda r: r.get('_mdr_text', ''),
    "PATIENT_HARM":      lambda r: r.get('incident_details', {}).get('patient_harm'),
    "PROBLEM_COMPONENTS": lambda r: json.dumps(
        r.get('incident_details', {}).get('problem_components', [])
    ),
    "DEFECT_CONFIRMED":  lambda r: r.get('manufacturer_inspection', {}).get('defect_confirmed'),
    "DEFECT_TYPE":       lambda r: r.get('manufacturer_inspection', {}).get('defect_type'),
}


def prepare_insert_data(
    results: List[Dict[str, Any]],
    columns: List[dict],
) -> List[tuple]:
    """vLLM 추출 결과를 INSERT용 데이터로 변환.

    Args:
        results: MAUDEExtractor 추출 결과 리스트
        columns: YAML extracted.columns 스키마

    Returns:
        [(col1_val, col2_val, ...), ...] 성공한 레코드만 포함
    """
    col_names = [col['name'] for col in columns]
    insert_data = []

    for result in results:
        if not result.get('_success', False):
            continue

        row = tuple(
            _RESULT_KEY_MAP[name](result)
            for name in col_names
        )
        insert_data.append(row)

    return insert_data


# ============================================================================
# 리스트 기반 추출기 파사드
# ============================================================================

def records_to_rows(
    records: List[Union[str, dict]],
) -> List[dict]:
    """리스트 입력을 추출기 내부에서 쓰는 행 리스트로 변환.

    - str 이면 mdr_text 로만 사용, product_problems 는 빈 문자열.
    - dict 이면 'mdr_text' 필수, 'product_problems' / 'product_problems_str' 선택.
    """
    rows = []
    for i, r in enumerate(records):
        if isinstance(r, str):
            rows.append({"mdr_text": r, "product_problems": ""})
        elif isinstance(r, dict):
            text = r.get("mdr_text") or r.get("MDR_TEXT")
            if text is None:
                raise ValueError(f"record[{i}] has no mdr_text")
            probs = r.get("product_problems") or r.get("product_problems_str") or ""
            rows.append({"mdr_text": str(text), "product_problems": str(probs)})
        else:
            raise TypeError(f"record[{i}] must be str or dict, got {type(r)}")
    return rows


def get_extractor_class():
    """MAUDEExtractor를 lazy import (지연 로딩).

    이 함수가 호출되기 전까지 vLLM, torch 등 무거운 라이브러리가
    로딩되지 않습니다.
    """
    try:
        from maude_early_alert.preprocessors.mdr_extractor import MAUDEExtractor
        return MAUDEExtractor
    except ImportError:
        raise ImportError(
            "MAUDEExtractor를 찾을 수 없습니다. "
            "maude_early_alert.preprocessors.mdr_extractor 경로를 확인하세요."
        )


class MDRExtractor:
    """MDR 텍스트 리스트를 받아 추출 결과를 반환하는 파사드.

    파이프라인 예:
        extractor = MDRExtractor(model_path=..., ...)
        results = extractor.process_batch(unique_mdr_text)
    """

    def __init__(self, **kwargs):
        """MAUDEExtractor 와 동일한 인자 전달 (model_path, tensor_parallel_size 등)."""
        self._kwargs = kwargs
        self._extractor = None

    def _get_extractor(self):
        if self._extractor is None:
            Klass = get_extractor_class()
            self._extractor = Klass(**self._kwargs)
        return self._extractor

    def process_batch(
        self,
        records: List[Union[str, dict]],
        checkpoint_dir: Optional[Union[str, Path]] = None,
        checkpoint_interval: int = 5000,
        checkpoint_prefix: str = "checkpoint",
    ):
        """리스트(또는 unique_mdr_text)를 받아 추출 결과 반환.

        Args:
            records: MDR 텍스트 문자열 리스트, 또는 dict 리스트
            checkpoint_dir: None 이면 체크포인트 없이 process_with_retry 만 수행
            checkpoint_interval: 체크포인트 간격
            checkpoint_prefix: 체크포인트 파일 접두사

        Returns:
            추출 결과 dict 리스트
        """
        rows = records_to_rows(records)

        impl = self._get_extractor()
        if checkpoint_dir is None:
            return impl.process_with_retry(rows)
        return impl.process_batch(
            rows,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            checkpoint_prefix=checkpoint_prefix,
        )


if __name__ == "__main__":
    cols = [
        {"name": "ID",    "type": "VARCHAR(100)", "primary_key": True},
        {"name": "COL_A", "type": "VARCHAR(50)"},
        {"name": "COL_B", "type": "BOOLEAN"},
    ]
    pk      = "ID"
    non_pk  = ["COL_A", "COL_B"]

    print("=== build_mdr_text_extract_sql ===")
    print(build_mdr_text_extract_sql("SRC_TABLE", ["ID", "COL_A"], where="ID IS NOT NULL", limit=10))

    print("\n=== build_ensure_extracted_table_sql ===")
    print(build_ensure_extracted_table_sql("DST_TABLE", cols))

    print("\n=== build_create_extract_temp_sql ===")
    print(build_create_extract_temp_sql("TMP_TABLE", cols))

    print("\n=== build_extract_stage_insert_sql ===")
    print(build_extract_stage_insert_sql("TMP_TABLE", cols))

    print("\n=== build_extract_merge_sql ===")
    print(build_extract_merge_sql("DST_TABLE", "TMP_TABLE", pk, non_pk))

    print("\n=== build_extracted_join_sql ===")
    print(build_extracted_join_sql("SRC_TABLE", "SRC_TABLE_EXTRACTED", non_pk, pk))

    print("\n=== records_to_rows ===")
    for r in records_to_rows([
        "plain string",
        {"mdr_text": "dict input", "product_problems": "some problem"},
    ]):
        print(r)

    print("\n=== prepare_insert_data ===")
    sample_results = [{
        '_mdr_text': 'hello',
        'incident_details': {'patient_harm': 'Minor', 'problem_components': ['x']},
        'manufacturer_inspection': {'defect_confirmed': True, 'defect_type': 'Material'},
    }]
    for row in prepare_insert_data(sample_results, cols):
        print(row)

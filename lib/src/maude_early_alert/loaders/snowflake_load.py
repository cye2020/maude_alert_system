# ======================
# 표준 라이브러리
# ======================
from typing import Any, Dict, List, Union

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.utils.helpers import (
    ensure_list,
    format_sql_literal,
)

# =====================
# text sql 적재 필요 파일
# =====================
from maude_early_alert.utils.sql_builder import build_cte_sql, build_insert_sql, build_join_clause
from textwrap import dedent
import json

import structlog

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def get_staging_table_name(table_name: str) -> str:
    """원본 테이블명에서 STG_ 접두사가 붙은 스테이징 테이블명 생성"""
    parts = table_name.split('.')
    parts[-1] = f"STG_{parts[-1].upper()}"
    return '.'.join(parts)


def get_raw_table_name(s3_stage_path: str) -> str:
    """S3 스테이지 경로에서 RAW_TEMP 테이블명 생성"""
    return f"RAW_TEMP_{s3_stage_path.replace('/', '_')}"


class SnowflakeLoader:
    """S3→Snowflake 적재용 SQL 빌더 (실행하지 않음)"""

    # ==================== Bronze SQL 빌더 ====================

    @staticmethod
    def build_create_staging_sql(
        table_name: str,
        table_schema: List[tuple],
    ) -> str:
        """원본 테이블 구조로 임시 스테이징 테이블 CREATE SQL 생성

        Args:
            table_name: 원본 테이블명
            table_schema: [(col_name, data_type), ...] 리스트
        """
        stg_table_name = get_staging_table_name(table_name)
        column_defs = ", ".join([f"{col} {data_type}" for col, data_type in table_schema])
        return dedent(f"""\
            CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (
                {column_defs}
            )""")

    @staticmethod
    def build_create_raw_temp_sql(s3_stage_path: str) -> str:
        """Raw JSON 임시 테이블 CREATE SQL 생성"""
        raw_table_name = get_raw_table_name(s3_stage_path)
        return dedent(f"""\
            CREATE OR REPLACE TEMPORARY TABLE {raw_table_name} (
                src_file VARCHAR,
                raw_json VARIANT
            )""")

    @staticmethod
    def build_copy_into_sql(
        raw_table_name: str,
        s3_stage_path: str,
    ) -> str:
        """S3에서 Raw JSON을 임시 테이블에 COPY INTO SQL 생성"""
        return dedent(f"""\
            COPY INTO {raw_table_name} (src_file, raw_json)
            FROM (
                SELECT METADATA$FILENAME, $1
                FROM @{s3_stage_path}
            )
            FILE_FORMAT = (TYPE = 'JSON', STRIP_OUTER_ARRAY = FALSE)
            ON_ERROR = 'CONTINUE'""")

    @staticmethod
    def build_flatten_insert_sql(
        raw_table_name: str,
        stg_table_name: str,
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None,
        json_path: str = 'results',
    ) -> str:
        """Raw 테이블의 JSON 배열을 FLATTEN하여 스테이징 테이블에 INSERT하는 SQL 생성

        Args:
            raw_table_name: Raw JSON 임시 테이블명
            stg_table_name: 스테이징 테이블명
            metadata: 메타데이터 (컬럼으로 추가됨)
            business_primary_key: 비즈니스 기본 키
            json_path: FLATTEN 대상 JSON 경로
        """
        bpk_list = ensure_list(business_primary_key) if business_primary_key else None

        column_names = []
        select_items = []

        if metadata:
            for key, value in metadata.items():
                column_names.append(key)
                select_items.append(format_sql_literal(key, value))

        if bpk_list:
            column_names.extend(bpk_list)
            select_items.extend([
                f"value:{key}::STRING AS {key}" for key in bpk_list
            ])
            object_delete_keys = ",".join([f"'{key}'" for key in bpk_list])
        else:
            object_delete_keys = None

        column_names.extend(['source_file', 'record_hash', 'raw_data'])
        select_items.extend([
            "src_file AS source_file",
            f"HASH(OBJECT_DELETE(value, {object_delete_keys})) AS record_hash" if bpk_list else "HASH(value) AS record_hash",
            "value::VARIANT AS raw_data"
        ])
        select_clause = ',\n'.join(select_items)

        return dedent(f"""\
            INSERT INTO {stg_table_name} ({', '.join(column_names)})
            SELECT {select_clause}
            FROM {raw_table_name},
            LATERAL FLATTEN(input => raw_json:{json_path})""")

    @staticmethod
    def build_merge_sql(
        table_name: str,
        stg_table_name: str,
        primary_key: Union[str, List[str]],
        column_names: List[str],
    ) -> str:
        """스테이징 → 목적 테이블 MERGE (UPSERT) SQL 생성

        Args:
            table_name: 목적 테이블명
            stg_table_name: 스테이징 테이블명
            primary_key: MERGE 조건 키
            column_names: 대상 컬럼 리스트
        """
        pk_list = ensure_list(primary_key)
        on_conditions = " AND ".join([f"t.{pk} = s.{pk}" for pk in pk_list])

        update_column_names = [col for col in column_names if col not in pk_list]
        update_set = ", ".join([f"t.{col} = s.{col}" for col in update_column_names])

        insert_column_names = ", ".join(column_names)
        insert_values = ", ".join([f"s.{col}" for col in column_names])

        return dedent(f"""\
            MERGE INTO {table_name} t
            USING {stg_table_name} s
            ON {on_conditions}
            WHEN MATCHED THEN
            UPDATE SET {update_set}
            WHEN NOT MATCHED THEN
            INSERT ({insert_column_names}) VALUES ({insert_values})""")

    # ==================== Silver (extractor) SQL 빌더 ====================

    @staticmethod
    def build_dataframe_load_sql(
        table_name: str,
        columns: List[str],
    ) -> str:
        """dataframe 적재 sql"""
        return build_insert_sql(
            table_name=table_name,
            columns=columns,
            num_rows=1,
        )

    @staticmethod
    def prepare_insert_data(
        results: List[Dict[str, Any]],
    ) -> List[tuple]:
        """vLLM 추출 결과를 MERGE용 데이터로 변환."""
        insert_data = []

        for result in results:
            if not result.get('_success', False):
                continue

            incident = result.get('incident_details', {})
            manufacturer = result.get('manufacturer_inspection', {})

            row_data = (
                result.get('_mdr_text', ''),
                incident.get('patient_harm'),
                json.dumps(incident.get('problem_components', [])),
                manufacturer.get('defect_confirmed'),
                manufacturer.get('defect_type'),
            )

            insert_data.append(row_data)

        return insert_data

    @staticmethod
    def build_ensure_extracted_table_sql(extracted_table: str) -> str:
        """추출 결과 전용 테이블 CREATE IF NOT EXISTS SQL 생성"""
        return dedent(f"""\
            CREATE TABLE IF NOT EXISTS {extracted_table} (
                MDR_TEXT           VARCHAR(16777216) NOT NULL,
                PATIENT_HARM       VARCHAR(50),
                PROBLEM_COMPONENTS VARCHAR(16777216),
                DEFECT_CONFIRMED   BOOLEAN,
                DEFECT_TYPE        VARCHAR(50),
                PRIMARY KEY (MDR_TEXT)
            )""")

    @staticmethod
    def build_create_extract_temp_sql(temp_table: str) -> str:
        """추출 결과용 임시 스테이징 테이블 CREATE SQL 생성"""
        return dedent(f"""\
            CREATE TEMPORARY TABLE {temp_table} (
                MDR_TEXT           VARCHAR(16777216),
                PATIENT_HARM       VARCHAR(50),
                PROBLEM_COMPONENTS VARCHAR(16777216),
                DEFECT_CONFIRMED   BOOLEAN,
                DEFECT_TYPE        VARCHAR(50)
            )""")

    @staticmethod
    def build_extract_stage_insert_sql(temp_table: str) -> str:
        """추출 결과 임시 테이블 INSERT SQL 생성 (executemany용)"""
        return build_insert_sql(
            table_name=temp_table,
            columns=[
                'MDR_TEXT', 'PATIENT_HARM', 'PROBLEM_COMPONENTS',
                'DEFECT_CONFIRMED', 'DEFECT_TYPE',
            ],
            num_rows=1,
        )

    @staticmethod
    def build_extract_merge_sql(
        extracted_table: str,
        temp_table: str,
    ) -> str:
        """추출 결과 MERGE SQL 생성"""
        return dedent(f"""\
            MERGE INTO {extracted_table} AS target
            USING {temp_table} AS source
            ON target.MDR_TEXT = source.MDR_TEXT
            WHEN MATCHED THEN UPDATE SET
                target.PATIENT_HARM        = source.PATIENT_HARM,
                target.PROBLEM_COMPONENTS  = source.PROBLEM_COMPONENTS,
                target.DEFECT_CONFIRMED    = source.DEFECT_CONFIRMED,
                target.DEFECT_TYPE         = source.DEFECT_TYPE
            WHEN NOT MATCHED THEN INSERT (
                MDR_TEXT, PATIENT_HARM, PROBLEM_COMPONENTS,
                DEFECT_CONFIRMED, DEFECT_TYPE
            ) VALUES (
                source.MDR_TEXT, source.PATIENT_HARM, source.PROBLEM_COMPONENTS,
                source.DEFECT_CONFIRMED, source.DEFECT_TYPE
            )""")

    @staticmethod
    def build_extracted_join_sql(
        base_table_name: str = 'EVENT_STAGE_12',
        extracted_suffix: str = '_EXTRACTED',
        base_alias: str = 'e',
        extract_alias: str = 'ex',
        select_columns: List[str] = None,
    ) -> str:
        """원본 테이블과 추출 결과 테이블을 LEFT JOIN하는 SELECT SQL 생성."""
        extracted_table = f"{base_table_name}{extracted_suffix}"
        extract_cols = [
            f"{extract_alias}.PATIENT_HARM",
            f"{extract_alias}.PROBLEM_COMPONENTS",
            f"{extract_alias}.DEFECT_CONFIRMED",
            f"{extract_alias}.DEFECT_TYPE",
        ]

        if select_columns is None:
            select_columns = [f"{base_alias}.*"] + extract_cols

        join_clause = build_join_clause(
            left_table=base_table_name,
            right_table=extracted_table,
            on_columns="MDR_TEXT",
            join_type="LEFT",
            left_alias=base_alias,
            right_alias=extract_alias,
        )

        return build_cte_sql(
            ctes=[],
            from_clause=f"{base_table_name} {base_alias}",
            select_cols=select_columns,
            joins=[join_clause],
        )

    # ==================== key & type 조회 (유틸) ====================

    @staticmethod
    def _fetch_map(cursor, sql: str) -> Dict[str, str]:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return {str(k): str(t) for k, t in rows if k is not None}

    @staticmethod
    def top_keys_with_type(cursor, table_fq: str, raw_column: str = "raw_data"):
        """최상위 키와 타입 조회"""
        sql = f"""
        SELECT DISTINCT f.key, TYPEOF(f.value)
        FROM {table_fq},
            LATERAL FLATTEN(input => {raw_column}) AS f
        ORDER BY f.key;
        """
        return SnowflakeLoader._fetch_map(cursor, sql)

    @staticmethod
    def array_keys_with_type(cursor, table_fq: str, array_path: str, raw_column: str = "raw_data"):
        """배열 내 모든 키를 RECURSIVE로 조회하여 nested dict 반환"""
        import re
        sql = f"""
        SELECT DISTINCT f.path::STRING, f.key::STRING, TYPEOF(f.value)
        FROM {table_fq},
            LATERAL FLATTEN(input => {raw_column}:{array_path}, RECURSIVE => TRUE) f
        WHERE f.key IS NOT NULL
          AND TYPEOF(f.value) != 'OBJECT'
        ORDER BY 1, 2
        """
        cursor.execute(sql)

        result = {}
        for raw_path, key, typ in cursor.fetchall():
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

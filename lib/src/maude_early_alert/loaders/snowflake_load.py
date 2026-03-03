# ======================
# 표준 라이브러리
# ======================
from textwrap import dedent, indent
from typing import Any, Dict, List, Union

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.utils.helpers import (
    ensure_list,
    format_sql_literal,
)
from maude_early_alert.utils.sql_builder import _INDENT

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
        column_defs = indent(",\n".join([f"{col} {data_type}" for col, data_type in table_schema]), _INDENT)
        return f"CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (\n{column_defs}\n)"

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
        select_clause = indent(",\n".join(select_items), _INDENT)

        return (
            f"INSERT INTO {stg_table_name} ({', '.join(column_names)})\n"
            f"SELECT\n{select_clause}\n"
            f"FROM {raw_table_name},\n"
            f"LATERAL FLATTEN(input => raw_json:{json_path})"
        )

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



if __name__ == '__main__':
    import pendulum

    # ── 테스트용 파라미터 ──
    table_name = 'EVENT'
    s3_stage_path = 'BRONZE_S3_STAGE/202602/device/event'
    table_schema = [
        ('SOURCE_SYSTEM', 'VARCHAR(50)'),
        ('INGEST_TIME', 'TIMESTAMP_NTZ'),
        ('BATCH_ID', 'VARCHAR(100)'),
        ('MDR_REPORT_KEY', 'VARCHAR(255)'),
        ('SOURCE_FILE', 'VARCHAR(500)'),
        ('RECORD_HASH', 'NUMBER'),
        ('RAW_DATA', 'VARIANT'),
    ]
    primary_key = ['SOURCE_SYSTEM', 'SOURCE_FILE', 'MDR_REPORT_KEY']
    business_primary_key = 'MDR_REPORT_KEY'
    metadata = {
        'source_system': 's3',
        'ingest_time': pendulum.now(),
        'batch_id': 'batch_test_001',
    }
    column_names = [col for col, _ in table_schema]
    stg_table_name = get_staging_table_name(table_name)
    raw_table_name = get_raw_table_name(s3_stage_path)

    # ── Bronze SQL 빌더 출력 ──
    print('=' * 60)
    print('[1] CREATE STAGING TABLE')
    print('=' * 60)
    print(SnowflakeLoader.build_create_staging_sql(table_name, table_schema))

    print('\n' + '=' * 60)
    print('[2] CREATE RAW TEMP TABLE')
    print('=' * 60)
    print(SnowflakeLoader.build_create_raw_temp_sql(s3_stage_path))

    print('\n' + '=' * 60)
    print('[3] COPY INTO')
    print('=' * 60)
    print(SnowflakeLoader.build_copy_into_sql(raw_table_name, s3_stage_path))

    print('\n' + '=' * 60)
    print('[4] FLATTEN INSERT')
    print('=' * 60)
    print(SnowflakeLoader.build_flatten_insert_sql(
        raw_table_name, stg_table_name,
        metadata=metadata,
        business_primary_key=business_primary_key,
    ))

    print('\n' + '=' * 60)
    print('[5] MERGE')
    print('=' * 60)
    print(SnowflakeLoader.build_merge_sql(
        table_name, stg_table_name, primary_key, column_names,
    ))

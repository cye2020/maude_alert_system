# ======================
# 표준 라이브러리
# ======================
import datetime
from typing import Any, Dict, List, Tuple, Union

# ======================
# 서드파티 라이브러리
# ======================
from snowflake.connector.cursor import SnowflakeCursor

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.loading.snowflake_base import SnowflakeBase
from maude_early_alert.utils.helpers import (
    ensure_list,
    format_sql_literal,
    validate_identifier,
)


def get_staging_table_name(table_name: str) -> str:
    """원본 테이블명에서 STG_ 접두사가 붙은 스테이징 테이블명 생성"""
    parts = table_name.split('.')
    parts[-1] = f"STG_{parts[-1].upper()}"
    return '.'.join(parts)


class SnowflakeLoader(SnowflakeBase):
    """S3에서 Snowflake로 데이터 적재 (COPY INTO → FLATTEN INSERT → MERGE)"""

    def load_from_s3(
        self, cursor: SnowflakeCursor,
        table_name: str, s3_stg_table_name: str,
        primary_key: Union[str, List[str]],
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None,
        s3_folder: str = None
    ) -> Dict[str, Any]:
        """S3에서 Snowflake 테이블로 데이터 적재

        Args:
            cursor: Snowflake cursor
            table_name: 목적 테이블명
            s3_stg_table_name: S3 스테이지명
            primary_key: MERGE 기본 키
            metadata: 메타데이터 (ingest_time 등)
            business_primary_key: 비즈니스 기본 키
            s3_folder: S3 폴더 경로 (None이면 ingest_time에서 자동 생성)
        """
        validate_identifier(table_name)
        validate_identifier(s3_stg_table_name)
        for pk in ensure_list(primary_key):
            validate_identifier(pk)
        if business_primary_key:
            for bpk in ensure_list(business_primary_key):
                validate_identifier(bpk)

        # S3 폴더 경로 결정: 명시적 지정 > metadata의 ingest_time > 없음
        if s3_folder is None and metadata and 'ingest_time' in metadata:
            ingest_time = metadata['ingest_time']
            if isinstance(ingest_time, datetime.datetime):
                s3_folder = ingest_time.strftime('%Y%m')

        s3_stage_path = f"{s3_stg_table_name}/{s3_folder}" if s3_folder else s3_stg_table_name

        self.logger.info('S3 데이터 적재 시작',
                        table_name=table_name, s3_stage=s3_stage_path)

        table_schema = self.get_table_schema(cursor, table_name)

        # 1. 스테이징 테이블 생성
        stg_table_name = self.create_temporary_staging_table(cursor, table_name, table_schema)

        # 2. S3에서 Raw JSON을 임시 테이블에 COPY INTO
        raw_table_name, copy_result = self.copy_raw_to_temp(cursor, s3_stage_path)

        # 3. FLATTEN으로 results 배열을 풀어서 스테이징 테이블에 INSERT
        insert_result = self.flatten_and_insert(
            cursor, raw_table_name, stg_table_name,
            metadata=metadata, business_primary_key=business_primary_key
        )

        # 4. MERGE로 목적 테이블로 적재
        column_names = [col_name for col_name, _ in table_schema]
        merge_count = self.load_merge(cursor, table_name, stg_table_name, primary_key, column_names)

        self.logger.info('S3 데이터 적재 완료',
                        table_name=table_name,
                        rows_inserted=insert_result.get('rows_inserted', 0),
                        total_rows=merge_count)

        return {
            'files_loaded': copy_result.get('files_loaded', 0),
            'rows_inserted': insert_result.get('rows_inserted', 0),
            'total_rows': merge_count
        }

    def create_temporary_staging_table(
        self, cursor: SnowflakeCursor,
        table_name: str, table_schema: list = None
    ) -> str:
        """원본 테이블 구조로 임시 스테이징 테이블 생성

        Args:
            cursor: Snowflake cursor
            table_name: 원본 테이블명
            table_schema: 스키마 (미지정 시 자동 조회)
        """
        stg_table_name = get_staging_table_name(table_name)
        self.logger.debug( '임시 스테이징 테이블 생성 시작', stg_table_name=stg_table_name)

        with self._error_logging('임시 테이블 생성', stg_table_name=stg_table_name):
            if not table_schema:
                table_schema = self.get_table_schema(cursor, table_name)

            column_defs = ", ".join([f"{col} {data_type}" for col, data_type in table_schema])

            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (
                    {column_defs}
                )
            """)

            self.logger.debug( '임시 스테이징 테이블 생성 완료', stg_table_name=stg_table_name)
            return stg_table_name

    def copy_raw_to_temp(
        self, cursor: SnowflakeCursor,
        s3_stg_table_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """S3에서 Raw JSON을 임시 테이블에 COPY INTO

        Args:
            cursor: Snowflake cursor
            s3_stg_table_name: S3 스테이지 경로
        """
        raw_table_name = f"RAW_TEMP_{s3_stg_table_name.replace('/', '_')}"
        self.logger.debug( 'Raw COPY INTO 시작',
                  raw_table_name=raw_table_name, s3_stage=s3_stg_table_name)

        with self._error_logging('Raw COPY INTO', raw_table_name=raw_table_name):
            self._set_context(cursor)

            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {raw_table_name} (
                    src_file VARCHAR,
                    raw_json VARIANT
                )
            """)
            self.logger.debug( 'Raw 임시 테이블 생성 완료', raw_table_name=raw_table_name)

            copy_query = f"""
            COPY INTO {raw_table_name} (src_file, raw_json)
            FROM (
                SELECT METADATA$FILENAME, $1
                FROM @{s3_stg_table_name}
            )
            FILE_FORMAT = (TYPE = 'JSON', STRIP_OUTER_ARRAY = FALSE)
            ON_ERROR = 'CONTINUE'
            """
            self.logger.debug( 'COPY INTO 쿼리 실행', query=copy_query)

            cursor.execute(copy_query)
            results = cursor.fetchall()

            # COPY INTO 결과 합산: (file, status, rows_parsed, rows_loaded, error_limit, errors_seen, ...)
            files_loaded = sum(1 for r in results if r[1] == 'LOADED')
            rows_loaded = sum(r[3] for r in results if r and len(r) > 3)
            errors_seen = sum(r[5] for r in results if r and len(r) > 5)

            self.logger.debug( 'Raw COPY INTO 완료',
                      raw_table_name=raw_table_name,
                      files_loaded=files_loaded, rows_loaded=rows_loaded,
                      errors_seen=errors_seen)

            if errors_seen > 0:
                self.logger.warning( 'Raw COPY INTO 중 에러 발생',
                          raw_table_name=raw_table_name, errors_seen=errors_seen)

                for r in results:
                    if r[5] > 0:
                        self.logger.warning( 'COPY INTO 에러 상세',
                                  file=r[0], status=r[1],
                                  errors_seen=r[5], first_error=r[6],
                                  first_error_line=r[7])

            return raw_table_name, {
                'files_loaded': files_loaded,
                'rows_loaded': rows_loaded,
                'errors_seen': errors_seen
            }

    def flatten_and_insert(
        self, cursor: SnowflakeCursor,
        raw_table_name: str,
        stg_table_name: str,
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None,
        json_path: str = 'results'
    ) -> Dict[str, Any]:
        """Raw 테이블의 JSON 배열을 FLATTEN하여 스테이징 테이블에 INSERT

        Args:
            cursor: Snowflake cursor
            raw_table_name: Raw JSON 임시 테이블명
            stg_table_name: 스테이징 테이블명
            metadata: 메타데이터 (컬럼으로 추가됨)
            business_primary_key: 비즈니스 기본 키
            json_path: FLATTEN 대상 JSON 경로
        """
        self.logger.debug( 'FLATTEN INSERT 시작',
                  raw_table_name=raw_table_name, stg_table_name=stg_table_name)

        with self._error_logging('FLATTEN INSERT', stg_table_name=stg_table_name):
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

            insert_query = f"""
            INSERT INTO {stg_table_name} ({', '.join(column_names)})
            SELECT {select_clause}
            FROM {raw_table_name},
            LATERAL FLATTEN(input => raw_json:{json_path})
            """
            self.logger.debug( 'INSERT 쿼리 실행', query=insert_query)

            cursor.execute(insert_query)
            rows_inserted = cursor.rowcount

            self.logger.debug( 'FLATTEN INSERT 완료',
                      stg_table_name=stg_table_name, rows_inserted=rows_inserted)

            return {'rows_inserted': rows_inserted}

    def load_merge(
        self, cursor: SnowflakeCursor,
        table_name: str, stg_table_name: str,
        primary_key: Union[str, List[str]], column_names: List[str]
    ) -> int:
        """스테이징 → 목적 테이블 MERGE (UPSERT)

        Args:
            cursor: Snowflake cursor
            table_name: 목적 테이블명
            stg_table_name: 스테이징 테이블명
            primary_key: MERGE 조건 키
            column_names: 대상 컬럼 리스트
        """
        self.logger.debug( 'MERGE 시작',
                  table_name=table_name, stg_table_name=stg_table_name)

        with self._error_logging('MERGE', table_name=table_name):
            pk_list = ensure_list(primary_key)
            on_conditions = " AND ".join([f"t.{pk} = s.{pk}" for pk in pk_list])

            update_column_names = [col for col in column_names if col not in pk_list]
            update_set = ", ".join([f"t.{col} = s.{col}" for col in update_column_names])

            insert_column_names = ", ".join(column_names)
            insert_values = ", ".join([f"s.{col}" for col in column_names])

            query = f"""
                MERGE INTO {table_name} t
                USING {stg_table_name} s
                ON {on_conditions}
                WHEN MATCHED THEN
                UPDATE SET {update_set}
                WHEN NOT MATCHED THEN
                INSERT ({insert_column_names}) VALUES ({insert_values})
            """

            with self.transaction(cursor):
                cursor.execute(query)

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_count = cursor.fetchone()[0]

            self.logger.debug( 'MERGE 완료',
                      table_name=table_name, total_rows=total_count)

            return total_count

    # key & type 값 불러오기

    def _fetch_map(cursor, sql: str) -> Dict[str, str]:
        cursor.execute(sql)
        rows = cursor.fetchall()
        return {str(k): str(t) for k, t in rows if k is not None}

    def top_keys_with_type(cursor, table_fq: str, raw_column: str = "raw_data"):
        """최상위 키와 타입 조회"""
        sql = f"""
        SELECT DISTINCT f.key, TYPEOF(f.value)
        FROM {table_fq},
            LATERAL FLATTEN(input => {raw_column}) AS f
        ORDER BY f.key;
        """
        return SnowflakeLoader._fetch_map(cursor, sql)

    def array_keys_with_type(cursor, table_fq: str, array_path: str, raw_column: str = "raw_data"):
        """배열 내 모든 키를 RECURSIVE로 조회하여 nested dict 반환

        OBJECT 타입은 중간 노드이므로 제외하고, 하위 키가 자동으로
        nested dict로 구성된다.

        Args:
            cursor: Snowflake cursor
            table_fq: 테이블 전체 경로
            array_path: 배열 경로 (예: "device", "patient", "mdr_text")
            raw_column: JSON 컬럼명

        Returns:
            nested dict. 예:
            {"brand_name": "VARCHAR", "openfda": {"device_name": "VARCHAR", ...}}
        """
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
            # 첫 번째 배열 인덱스([0] 등) 제거 후, 내부에 [N]이 남아 있으면
            # 배열 원소 전개이므로 스킵 (예: openfda.registration_number[0])
            inner = re.sub(r'^\[\d+\]\.?', '', str(raw_path))
            if re.search(r'\[\d+\]', inner):
                continue

            clean = re.sub(r'\[\d+\]\.?', '', str(raw_path)).strip('.')
            parts = [p for p in clean.split('.') if p]

            # path의 마지막 요소는 key 자체이므로 부모 경로만 탐색
            # 예: path="[0].openfda.device_name", key="device_name"
            #   → parts=["openfda", "device_name"] → 탐색은 ["openfda"]만
            parent_parts = parts[:-1] if parts else []

            node = result
            for part in parent_parts:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[key] = typ

        return result

if __name__=='__main__':
    import pendulum
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret

    secret = get_secret('snowflake/bronze/credentials')
    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
        database=secret['database'],
        schema=secret['schema']
    )

    snowflake_loader = SnowflakeLoader(secret['database'], secret['schema'], log_level='DEBUG')
    table_name = 'UDI'
    s3_stg_table_name = 'BRONZE_S3_STAGE'
    metadata = {
        'source_system': 's3',
        'ingest_time': pendulum.now(),
        'batch_id': 'batch_001'
    }
    business_primary_key = 'public_device_record_key'
    primary_key = ['source_system', 'source_file', 'public_device_record_key']
    
    ym = pendulum.now().strftime('%Y%m')
    s3_folder = f'{ym}/device/udi'

    snowflake_loader.load_from_s3(
        cursor=conn.cursor(),
        table_name=table_name,
        s3_stg_table_name=s3_stg_table_name,
        primary_key=primary_key,
        business_primary_key=business_primary_key,
        metadata=metadata,
        s3_folder=s3_folder
    )
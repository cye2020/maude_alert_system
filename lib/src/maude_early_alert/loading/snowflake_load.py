from typing import Any, Dict, List, Tuple, Union
import datetime
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError

from .snowflake_base import SnowflakeBase
from ..utils.helpers import ensure_list, format_sql_literal


def get_staging_table_name(table_name: str) -> str:
    """Staging 테이블명 생성

    Args:
        table_name: 원본 테이블명

    Returns:
        Staging 테이블명
    """
    parts = table_name.split('.')
    parts[-1] = f"STG_{parts[-1].upper()}"
    return '.'.join(parts)


class SnowflakeLoader(SnowflakeBase):
    """S3에서 Snowflake로 데이터를 적재하는 클래스

    SnowflakeBase의 인프라(로깅, 컨텍스트, 스키마 조회, 트랜잭션)를 상속받아
    COPY INTO, FLATTEN INSERT, MERGE 등 데이터 적재 비즈니스 로직을 수행합니다.
    """

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
            cursor: Snowflake cursor 객체
            table_name: 목적 테이블명
            s3_stg_table_name: S3 스테이지명
            primary_key: 기본 키 (단일 또는 복수)
            metadata: 메타데이터 딕셔너리
            business_primary_key: 비즈니스 기본 키
            s3_folder: S3 스테이지 내 폴더 경로 (None이면 metadata의 ingest_time에서 %Y%m 형식으로 자동 생성)

        Returns:
            dict: 적재 결과 정보

        Raises:
            DatabaseError: 데이터베이스 작업 실패 시
            ProgrammingError: SQL 구문 오류 시
        """
        # S3 폴더 경로 결정: 명시적 지정 > metadata의 ingest_time > 없음
        if s3_folder is None and metadata and 'ingest_time' in metadata:
            ingest_time = metadata['ingest_time']
            if isinstance(ingest_time, datetime.datetime):
                s3_folder = ingest_time.strftime('%Y%m')

        # S3 스테이지 전체 경로 구성
        s3_stage_path = f"{s3_stg_table_name}/{s3_folder}" if s3_folder else s3_stg_table_name

        self.logger.info( 'S3 데이터 적재 시작',
                  table_name=table_name, s3_stage=s3_stage_path)

        try:
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

            self.logger.info( 'S3 데이터 적재 완료',
                      table_name=table_name,
                      rows_inserted=insert_result.get('rows_inserted', 0),
                      total_rows=merge_count)

            return {
                'files_loaded': copy_result.get('files_loaded', 0),
                'rows_inserted': insert_result.get('rows_inserted', 0),
                'total_rows': merge_count
            }

        except (ProgrammingError, DatabaseError):
            # 하위 메서드에서 이미 로깅 완료 — 재로깅 없이 전파
            raise
        except Exception as e:
            self.logger.error( '예상치 못한 오류 발생',
                      table_name=table_name, error=str(e))
            raise

    def create_temporary_staging_table(
        self, cursor: SnowflakeCursor,
        table_name: str, table_schema: list = None
    ) -> str:
        """임시 스테이징 테이블 생성

        Args:
            cursor: Snowflake cursor 객체
            table_name: 원본 테이블명
            table_schema: 테이블 스키마 (없으면 조회)

        Returns:
            str: 생성된 스테이징 테이블명

        Raises:
            ProgrammingError: 테이블 생성 실패 시
        """
        stg_table_name = get_staging_table_name(table_name)
        self.logger.debug( '임시 스테이징 테이블 생성 시작', stg_table_name=stg_table_name)

        try:
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

        except ProgrammingError as e:
            self.logger.error( '임시 테이블 생성 실패',
                      stg_table_name=stg_table_name, error=str(e))
            raise

    def copy_raw_to_temp(
        self, cursor: SnowflakeCursor,
        s3_stg_table_name: str
    ) -> Tuple[str, Dict[str, Any]]:
        """S3에서 Raw JSON을 임시 테이블에 COPY INTO

        Args:
            cursor: Snowflake cursor 객체
            s3_stg_table_name: S3 스테이지 경로

        Returns:
            Tuple[str, dict]: (임시 테이블명, {'files_loaded': 로드된 파일 수, 'rows_loaded': 로드된 행 수, 'errors_seen': 에러 수})

        Raises:
            ProgrammingError: COPY INTO 실패 시
        """
        raw_table_name = f"RAW_TEMP_{s3_stg_table_name.replace('/', '_')}"
        self.logger.debug( 'Raw COPY INTO 시작',
                  raw_table_name=raw_table_name, s3_stage=s3_stg_table_name)

        try:
            self._set_context(cursor)

            # Raw 임시 테이블 생성
            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {raw_table_name} (
                    src_file VARCHAR,
                    raw_json VARIANT
                )
            """)
            self.logger.debug( 'Raw 임시 테이블 생성 완료', raw_table_name=raw_table_name)

            # COPY INTO 실행
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

            # COPY INTO 결과: (file, status, rows_parsed, rows_loaded, error_limit, errors_seen, ...)
            # 파일별로 한 행씩 반환되므로 전체 합산
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

                # COPY INTO 결과에서 에러 상세 추출
                # 컬럼: file, status, rows_parsed, rows_loaded, error_limit, errors_seen,
                #        first_error, first_error_line, first_error_character, first_error_column_name
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

        except ProgrammingError as e:
            self.logger.error( 'Raw COPY INTO 실패',
                      raw_table_name=raw_table_name, error=str(e))
            raise

    def flatten_and_insert(
        self, cursor: SnowflakeCursor,
        raw_table_name: str,
        stg_table_name: str,
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None,
        json_path: str = 'results'
    ) -> Dict[str, Any]:
        """Raw 테이블에서 FLATTEN하여 스테이징 테이블에 INSERT

        Args:
            cursor: Snowflake cursor 객체
            raw_table_name: Raw JSON이 담긴 임시 테이블명
            stg_table_name: 스테이징 테이블명
            metadata: 메타데이터 딕셔너리
            business_primary_key: 비즈니스 기본 키
            json_path: FLATTEN할 JSON 경로 (기본: 'results')

        Returns:
            dict: {'rows_inserted': 삽입된 행 수}

        Raises:
            ProgrammingError: INSERT 실패 시
        """
        self.logger.debug( 'FLATTEN INSERT 시작',
                  raw_table_name=raw_table_name, stg_table_name=stg_table_name)

        try:
            # Business Primary Key List
            bpk_list = ensure_list(business_primary_key) if business_primary_key else None

            column_names = []
            select_items = []

            if metadata:
                for key, value in metadata.items():
                    column_names.append(key)
                    select_items.append(format_sql_literal(key, value))

            # Business Primary Key (value는 FLATTEN된 배열의 각 요소)
            if bpk_list:
                column_names.extend(bpk_list)
                select_items.extend([
                    f"value:{key}::STRING AS {key}" for key in bpk_list
                ])
                object_delete_keys = ",".join([f"'{key}'" for key in bpk_list])
            else:
                object_delete_keys = None

            # 고정 컬럼
            column_names.extend(['source_file', 'record_hash', 'raw_data'])
            select_items.extend([
                "src_file AS source_file",
                f"HASH(OBJECT_DELETE(value, {object_delete_keys})) AS record_hash" if bpk_list else "HASH(value) AS record_hash",
                "value::VARIANT AS raw_data"
            ])
            select_clause = ',\n'.join(select_items)

            # INSERT 쿼리 실행
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

        except ProgrammingError as e:
            self.logger.error( 'FLATTEN INSERT 실패',
                      stg_table_name=stg_table_name, error=str(e))
            raise

    def load_merge(
        self, cursor: SnowflakeCursor,
        table_name: str, stg_table_name: str,
        primary_key: Union[str, List[str]], column_names: List[str]
    ) -> int:
        """MERGE를 통해 스테이징 테이블에서 목적 테이블로 데이터 적재

        Args:
            cursor: Snowflake cursor 객체
            table_name: 목적 테이블명
            stg_table_name: 스테이징 테이블명
            primary_key: 기본 키 (단일 또는 복수)
            column_names: 컬럼명 리스트

        Returns:
            int: 최종 테이블 행 수

        Raises:
            ProgrammingError: MERGE 실패 시
            DatabaseError: 트랜잭션 실패 시
        """
        self.logger.debug( 'MERGE 시작',
                  table_name=table_name, stg_table_name=stg_table_name)

        try:
            # primary_key를 리스트로 정규화
            pk_list = ensure_list(primary_key)

            # ON 절 (복수 키를 AND로 연결)
            on_conditions = " AND ".join([f"t.{pk} = s.{pk}" for pk in pk_list])

            # UPDATE SET 절 (primary_key 제외)
            update_column_names = [col for col in column_names if col not in pk_list]
            update_set = ", ".join([f"t.{col} = s.{col}" for col in update_column_names])

            # INSERT 절
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

        except ProgrammingError as e:
            self.logger.error( 'MERGE 실패',
                      table_name=table_name, error=str(e))
            raise
        except DatabaseError as e:
            self.logger.error( 'MERGE 트랜잭션 실패',
                      table_name=table_name, error=str(e))
            raise


if __name__=='__main__':
    import pendulum
    import snowflake.connector
    import sys
    from pathlib import Path

    src = Path(__file__).parent.parent.parent
    if str(src) in sys.path:
        sys.path.remove(str(src))
    sys.path.insert(0, str(src))
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
    table_name = 'EVENT'
    s3_stg_table_name = 'BRONZE_S3_STAGE'
    metadata = {
        'source_system': 's3',
        'ingest_time': pendulum.now(),
        'batch_id': 'batch_001'
    }
    business_primary_key = 'mdr_report_key'
    primary_key = ['source_system', 'source_file', 'mdr_report_key']

    snowflake_loader.load_from_s3(
        cursor=conn.cursor(),
        table_name=table_name,
        s3_stg_table_name=s3_stg_table_name,
        primary_key=primary_key,
        business_primary_key=business_primary_key,
        metadata=metadata,
        s3_folder=pendulum.now().format('YYYYMM')
    )

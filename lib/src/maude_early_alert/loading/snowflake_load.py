from typing import Any, Dict, List, Literal, Tuple, Union
from contextlib import contextmanager
import logging
import snowflake.connector
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError
import pendulum
import structlog

LogLevel = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', None]

LOG_LEVEL_MAP = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

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

class SnowflakeLoader:
    def __init__(
        self, database: str, schema: str,
        log_level: LogLevel = 'INFO'
    ):
        self.database = database
        self.schema = schema
        self._setup_logger(log_level)

    def _setup_logger(self, log_level: LogLevel) -> None:
        """로거 설정

        Args:
            log_level: 로그 레벨 ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', None)
                       None이면 로그 비활성화
        """
        self.logger = structlog.get_logger(__name__)

        if log_level is None:
            # 로그 비활성화
            self._log_enabled = False
        else:
            self._log_enabled = True
            self._log_level = LOG_LEVEL_MAP.get(log_level, logging.INFO)

    def _log(self, level: str, message: str, **kwargs) -> None:
        """조건부 로그 출력

        Args:
            level: 로그 레벨 ('debug', 'info', 'warning', 'error', 'critical')
            message: 로그 메시지
            **kwargs: 추가 컨텍스트
        """
        if not self._log_enabled:
            return

        level_value = LOG_LEVEL_MAP.get(level.upper(), logging.INFO)
        if level_value >= self._log_level:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message, **kwargs)
    
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
            if isinstance(ingest_time, pendulum.DateTime):
                s3_folder = ingest_time.format('YYYYMM')

        # S3 스테이지 전체 경로 구성
        s3_stage_path = f"{s3_stg_table_name}/{s3_folder}" if s3_folder else s3_stg_table_name

        self._log('info', 'S3 데이터 적재 시작',
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

            self._log('info', 'S3 데이터 적재 완료',
                      table_name=table_name,
                      rows_inserted=insert_result.get('rows_inserted', 0),
                      total_rows=merge_count)

            return {
                'files_loaded': copy_result.get('files_loaded', 0),
                'rows_inserted': insert_result.get('rows_inserted', 0),
                'total_rows': merge_count
            }

        except ProgrammingError as e:
            self._log('error', 'SQL 구문 오류',
                      table_name=table_name, error=str(e))
            raise
        except DatabaseError as e:
            self._log('error', '데이터베이스 작업 실패',
                      table_name=table_name, error=str(e))
            raise
        except Exception as e:
            self._log('error', '예상치 못한 오류 발생',
                      table_name=table_name, error=str(e))
            raise
        finally:
            cursor.close()
    
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
        self._log('debug', '임시 스테이징 테이블 생성 시작', stg_table_name=stg_table_name)

        try:
            if not table_schema:
                table_schema = self.get_table_schema(cursor, table_name)

            column_defs = ", ".join([f"{col} {data_type}" for col, data_type in table_schema])

            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (
                    {column_defs}
                )
            """)

            self._log('debug', '임시 스테이징 테이블 생성 완료', stg_table_name=stg_table_name)
            return stg_table_name

        except ProgrammingError as e:
            self._log('error', '임시 테이블 생성 실패',
                      stg_table_name=stg_table_name, error=str(e))
            raise
    
    def _format_metadata_value(self, key: str, value: Any) -> str:
        """메타데이터 값을 SQL SELECT 표현식으로 변환

        Args:
            key: 컬럼명
            value: 메타데이터 값

        Returns:
            SQL SELECT 표현식 (예: "'value' AS column_name")
        """
        if value is None:
            return f"NULL AS {key}"
        elif isinstance(value, pendulum.DateTime):
            formatted = value.format('YYYY-MM-DD HH:mm:ss')
            return f"'{formatted}'::TIMESTAMP_TZ AS {key}"
        elif isinstance(value, bool):
            return f"{str(value).upper()} AS {key}"
        elif isinstance(value, (int, float)):
            return f"{value} AS {key}"
        else:
            # 문자열 및 기타 타입
            escaped = str(value).replace("'", "''")
            return f"'{escaped}' AS {key}"

    def copy_raw_to_temp(
        self, cursor: SnowflakeCursor,
        s3_stg_table_name: str,
        file_format: str = 'JSON'
    ) -> Tuple[str, Dict[str, Any]]:
        """S3에서 Raw JSON을 임시 테이블에 COPY INTO

        Args:
            cursor: Snowflake cursor 객체
            s3_stg_table_name: S3 스테이지 경로
            file_format: 파일 형식 (JSON, CSV, AVRO, ORC, PARQUET, XML)

        Returns:
            Tuple[str, dict]: (임시 테이블명, {'files_loaded': 로드된 파일 수, 'rows_loaded': 로드된 행 수, 'errors_seen': 에러 수})

        Raises:
            ProgrammingError: COPY INTO 실패 시
        """
        raw_table_name = f"RAW_TEMP_{s3_stg_table_name.replace('/', '_')}"
        self._log('debug', 'Raw COPY INTO 시작',
                  raw_table_name=raw_table_name, s3_stage=s3_stg_table_name)

        try:
            cursor.execute(f"USE DATABASE {self.database}")
            cursor.execute(f"USE SCHEMA {self.schema}")

            # Raw 임시 테이블 생성
            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {raw_table_name} (
                    src_file VARCHAR,
                    raw_json VARIANT
                )
            """)
            self._log('debug', 'Raw 임시 테이블 생성 완료', raw_table_name=raw_table_name)

            # COPY INTO 실행
            copy_query = f"""
            COPY INTO {raw_table_name} (src_file, raw_json)
            FROM (
                SELECT METADATA$FILENAME, $1
                FROM @{s3_stg_table_name}
            )
            FILE_FORMAT = (TYPE = '{file_format}', STRIP_OUTER_ARRAY = FALSE)
            ON_ERROR = 'CONTINUE'
            """
            self._log('debug', 'COPY INTO 쿼리 실행', query=copy_query)

            cursor.execute(copy_query)
            results = cursor.fetchall()

            # COPY INTO 결과: (file, status, rows_parsed, rows_loaded, error_limit, errors_seen, ...)
            # 파일별로 한 행씩 반환되므로 전체 합산
            files_loaded = sum(1 for r in results if r[1] == 'LOADED')
            rows_loaded = sum(r[3] for r in results if r and len(r) > 3)
            errors_seen = sum(r[5] for r in results if r and len(r) > 5)

            self._log('debug', 'Raw COPY INTO 완료',
                      raw_table_name=raw_table_name,
                      files_loaded=files_loaded, rows_loaded=rows_loaded,
                      errors_seen=errors_seen)

            if errors_seen > 0:
                self._log('warning', 'Raw COPY INTO 중 에러 발생',
                          raw_table_name=raw_table_name, errors_seen=errors_seen)

                # COPY INTO 결과에서 에러 상세 추출
                # 컬럼: file, status, rows_parsed, rows_loaded, error_limit, errors_seen,
                #        first_error, first_error_line, first_error_character, first_error_column_name
                for r in results:
                    if r[5] > 0:
                        self._log('warning', 'COPY INTO 에러 상세',
                                  file=r[0], status=r[1],
                                  errors_seen=r[5], first_error=r[6],
                                  first_error_line=r[7])

            return raw_table_name, {
                'files_loaded': files_loaded,
                'rows_loaded': rows_loaded,
                'errors_seen': errors_seen
            }

        except ProgrammingError as e:
            self._log('error', 'Raw COPY INTO 실패',
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
        self._log('debug', 'FLATTEN INSERT 시작',
                  raw_table_name=raw_table_name, stg_table_name=stg_table_name)

        try:
            # Business Primary Key List
            bpk_list = None
            if business_primary_key:
                bpk_list = (
                    [business_primary_key]
                    if isinstance(business_primary_key, str)
                    else business_primary_key
                )

            column_names = []
            select_items = []

            if metadata:
                for key, value in metadata.items():
                    column_names.append(key)
                    select_items.append(self._format_metadata_value(key, value))

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
            self._log('debug', 'INSERT 쿼리 실행', query=insert_query)

            cursor.execute(insert_query)
            rows_inserted = cursor.rowcount

            self._log('debug', 'FLATTEN INSERT 완료',
                      stg_table_name=stg_table_name, rows_inserted=rows_inserted)

            return {'rows_inserted': rows_inserted}

        except ProgrammingError as e:
            self._log('error', 'FLATTEN INSERT 실패',
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
        self._log('debug', 'MERGE 시작',
                  table_name=table_name, stg_table_name=stg_table_name)

        try:
            # primary_key를 리스트로 정규화
            pk_list = [primary_key] if isinstance(primary_key, str) else primary_key

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

            self._log('debug', 'MERGE 완료',
                      table_name=table_name, total_rows=total_count)

            return total_count

        except ProgrammingError as e:
            self._log('error', 'MERGE 실패',
                      table_name=table_name, error=str(e))
            raise
        except DatabaseError as e:
            self._log('error', 'MERGE 트랜잭션 실패',
                      table_name=table_name, error=str(e))
            raise

    
    def get_table_schema(
        self, cursor: SnowflakeCursor,
        table_name: str
    ) -> List[Tuple[str, str]]:
        """Snowflake 테이블 스키마 조회

        Args:
            cursor: Snowflake cursor 객체
            table_name: 테이블 이름

        Returns:
            List[Tuple[str, str]]: (컬럼명, 데이터타입) 튜플 리스트

        Raises:
            ValueError: 스키마 조회 실패 시
            ProgrammingError: SQL 실행 실패 시
        """
        self._log('debug', '테이블 스키마 조회 시작', table_name=table_name)

        try:
            query = f"""
            SELECT
                column_name,
                data_type
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = '{self.schema}'
                AND table_name = '{table_name}'
            """

            # INFORMATION_SCHEMA 접근 전 세션 스키마를 명시적으로 설정
            cursor.execute(f"USE DATABASE {self.database}")
            cursor.execute(f"USE SCHEMA {self.schema}")
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                error_msg = f"테이블 {self.database}.{self.schema}.{table_name} 스키마 조회 실패"
                self._log('error', error_msg, table_name=table_name)
                raise ValueError(error_msg)

            self._log('debug', '테이블 스키마 조회 완료',
                      table_name=table_name, column_count=len(rows))
            return rows

        except ProgrammingError as e:
            self._log('error', '스키마 조회 SQL 실행 실패',
                      table_name=table_name, error=str(e))
            raise
    
    @contextmanager
    def transaction(self, cursor: SnowflakeCursor):
        """트랜잭션 컨텍스트

        Args:
            cursor: Snowflake cursor 객체

        Yields:
            cursor 객체

        Note:
            - 성공 시 COMMIT
            - 예외 발생 시 ROLLBACK 후 재발생
        """
        self._log('debug', '트랜잭션 시작')
        cursor.execute("BEGIN")
        try:
            yield cursor
            cursor.execute("COMMIT")
            self._log('debug', '트랜잭션 커밋 완료')
        except Exception as e:
            cursor.execute("ROLLBACK")
            self._log('warning', '트랜잭션 롤백', error=str(e))
            raise

if __name__=='__main__':
    import snowflake.connector
    import sys
    from pathlib import Path
    
    src = Path(__file__).parent.parent.parent
    if str(src) in sys.path:
        sys.path.remove(str(src))
    sys.path.insert(0, str(src))
    from maude_early_alert.utils.secrets import get_secret
    
    secret = get_secret('snowflake/bronze/credentials')
    try:
        conn = snowflake.connector.connect(
            user=secret['user'],
            password=secret['password'],
            account=secret['account'],
            warehouse=secret['warehouse'], # Optional
            database=secret['database'],   # Optional
            schema=secret['schema']       # Optional
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
        primary_key = ['source_system', 'source_file' , 'mdr_report_key']
        
        snowflake_loader.load_from_s3(
            cursor = conn.cursor(),
            table_name = table_name,
            s3_stg_table_name = s3_stg_table_name,
            primary_key = primary_key,
            business_primary_key = business_primary_key,
            metadata = metadata,
            s3_folder=pendulum.now().strftime('%Y%m')
        )
        
    except Exception as e:
        raise Exception(e)

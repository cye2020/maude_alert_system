# ======================
# 표준 라이브러리
# ======================
from typing import List, Tuple
from contextlib import contextmanager
import logging

# ======================
# 서드파티 라이브러리
# ======================
import structlog
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import DatabaseError, ProgrammingError

# ======================
# 내부 라이브러리
# ======================
from maude_early_alert.utils.helpers import validate_identifier



class SnowflakeBase:
    """Snowflake 세션 컨텍스트, 스키마 조회, 트랜잭션 등 공통 인프라"""

    def __init__(
        self, database: str, schema: str,
        log_level: str = 'INFO'
    ):
        self.database = validate_identifier(database)
        self.schema = validate_identifier(schema)
        self.logger = structlog.get_logger(__name__)
        # structlog은 stdlib logging을 래핑 — stdlib 레벨 설정으로 필터링
        level = getattr(logging, log_level, logging.CRITICAL + 1) if log_level else logging.CRITICAL + 1
        logging.getLogger(__name__).setLevel(level)

    def _set_context(self, cursor: SnowflakeCursor) -> None:
        """Snowflake 세션의 DATABASE/SCHEMA 컨텍스트 설정"""
        cursor.execute(f"USE DATABASE {self.database}")
        cursor.execute(f"USE SCHEMA {self.schema}")

    @contextmanager
    def _error_logging(self, operation: str, **context):
        """SQL 에러 로깅 후 재발생시키는 컨텍스트 매니저"""
        try:
            yield
        except (ProgrammingError, DatabaseError) as e:
            self.logger.error(f'{operation} 실패', error=str(e), **context)
            raise

    def get_table_schema(
        self, cursor: SnowflakeCursor,
        table_name: str
    ) -> List[Tuple[str, str]]:
        """테이블 스키마(컬럼명, 데이터타입) 조회

        Args:
            cursor: Snowflake cursor
            table_name: 테이블 이름

        Raises:
            ValueError: 스키마 조회 결과 없음
            ProgrammingError: SQL 실행 실패
        """
        self.logger.debug('테이블 스키마 조회 시작', table_name=table_name)

        with self._error_logging('스키마 조회 SQL 실행', table_name=table_name):
            query = """
            SELECT
                column_name,
                data_type
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = %s
                AND table_name = %s
            """

            self._set_context(cursor)
            cursor.execute(query, (self.schema, table_name))
            rows = cursor.fetchall()

            if not rows:
                error_msg = f"테이블 {self.database}.{self.schema}.{table_name} 스키마 조회 실패"
                self.logger.error(error_msg, table_name=table_name)
                raise ValueError(error_msg)

            self.logger.debug('테이블 스키마 조회 완료',
                             table_name=table_name, column_count=len(rows))
            return rows

    @contextmanager
    def transaction(self, cursor: SnowflakeCursor):
        """BEGIN/COMMIT/ROLLBACK을 관리하는 트랜잭션 컨텍스트"""
        self.logger.debug('트랜잭션 시작')
        cursor.execute("BEGIN")
        try:
            yield cursor
            cursor.execute("COMMIT")
            self.logger.debug('트랜잭션 커밋 완료')
        except Exception as e:
            cursor.execute("ROLLBACK")
            self.logger.warning('트랜잭션 롤백', error=str(e))
            raise

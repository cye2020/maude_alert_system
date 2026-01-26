from typing import List, Tuple
from contextlib import contextmanager
import logging
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.errors import ProgrammingError
import structlog


class SnowflakeBase:
    """Snowflake 인프라 기반 클래스

    세션 컨텍스트, 스키마 조회, 트랜잭션 관리 등
    Snowflake 작업의 공통 인프라를 제공합니다.
    """

    def __init__(
        self, database: str, schema: str,
        log_level: str = 'INFO'
    ):
        self.database = database
        self.schema = schema
        self.logger = structlog.get_logger(__name__)
        # structlog은 stdlib logging을 래핑 — stdlib 레벨 설정으로 필터링
        level = getattr(logging, log_level, logging.CRITICAL + 1) if log_level else logging.CRITICAL + 1
        logging.getLogger(__name__).setLevel(level)

    def _set_context(self, cursor: SnowflakeCursor) -> None:
        """Snowflake 세션의 DATABASE/SCHEMA 컨텍스트 설정"""
        cursor.execute(f"USE DATABASE {self.database}")
        cursor.execute(f"USE SCHEMA {self.schema}")

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
        self.logger.debug('테이블 스키마 조회 시작', table_name=table_name)

        try:
            query = """
            SELECT
                column_name,
                data_type
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE table_schema = %s
                AND table_name = %s
            """

            # INFORMATION_SCHEMA 접근 전 세션 컨텍스트 설정
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

        except ProgrammingError as e:
            self.logger.error('스키마 조회 SQL 실행 실패',
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

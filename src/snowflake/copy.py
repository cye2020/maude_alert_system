"""
Snowflake COPY INTO 모듈

Snowflake 공식 튜토리얼 그대로 구현:
https://docs.snowflake.com/ko/user-guide/tutorials/script-data-load-transform-parquet
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import snowflake.connector
from snowflake.connector import DictCursor

from src.snowflake.config import get_snowflake_config

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class SnowflakeCopyLoader:
    """Snowflake COPY INTO 로더"""

    def __init__(self):
        self.config = get_snowflake_config()
        self.connection = None

    def connect(self):
        """Snowflake 연결"""
        logger.info("Snowflake 연결 중...")
        conn_params = self.config.get_connection_params()
        self.connection = snowflake.connector.connect(**conn_params)
        logger.info("연결 완료")

    def disconnect(self):
        """연결 종료"""
        if self.connection:
            self.connection.close()
            logger.info("연결 종료")

    def execute(self, query: str, fetch: bool = False):
        """쿼리 실행"""
        cursor = self.connection.cursor(DictCursor)
        # 타임아웃 없음으로 설정
        cursor.execute(query, timeout=None)

        if fetch:
            result = cursor.fetchall()
            cursor.close()
            return result

        cursor.close()
        return None

    def run(self):
        """튜토리얼 그대로 실행"""
        try:
            self.connect()

            # 1. 파일 포맷 생성
            logger.info("1. 파일 포맷 생성")
            self.execute("""
                CREATE OR REPLACE FILE FORMAT my_parquet_format
                TYPE = parquet
            """)

            # 2. 스테이지 생성
            logger.info("2. 스테이지 생성")
            self.execute("""
                CREATE OR REPLACE STAGE my_parquet_stage
                FILE_FORMAT = my_parquet_format
            """)

            # 3. 파일 업로드
            logger.info("3. 파일 업로드")
            local_file = self.config.get_data_file_path()
            logger.info(f"파일: {local_file}")

            self.execute(f"""
                PUT file://{local_file.as_posix()}
                @my_parquet_stage
                AUTO_COMPRESS = TRUE
            """)

            # 4. 스테이지 파일 확인
            logger.info("4. 스테이지 파일 확인")
            files = self.execute("LIST @my_parquet_stage", fetch=True)
            logger.info(f"파일 수: {len(files)}")

            # 5. 테이블 생성 (CTAS)
            logger.info("5. 테이블 생성")
            table_name = self.config.get_table_name()

            # 테이블이 이미 있으면 스킵
            check = self.execute(f"""
                SELECT COUNT(*) as cnt
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = '{table_name}'
            """, fetch=True)

            if check[0]['CNT'] == 0:
                self.execute(f"""
                    CREATE TABLE {table_name}
                    USING TEMPLATE (
                        SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
                        FROM TABLE(
                            INFER_SCHEMA(
                                LOCATION => '@my_parquet_stage',
                                FILE_FORMAT => 'my_parquet_format'
                            )
                        )
                    )
                """)
                logger.info(f"테이블 생성 완료: {table_name}")
            else:
                logger.info(f"테이블이 이미 존재: {table_name}")

            # 6. COPY INTO
            logger.info("6. COPY INTO 실행")
            result = self.execute(f"""
                COPY INTO {table_name}
                FROM @my_parquet_stage
                FILE_FORMAT = (FORMAT_NAME = 'my_parquet_format')
                MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
            """, fetch=True)

            # 결과
            if result:
                summary = {
                    'rows_loaded': sum(r.get('rows_loaded', 0) for r in result),
                    'rows_parsed': sum(r.get('rows_parsed', 0) for r in result),
                    'files_loaded': len(result)
                }
                logger.info(f"완료: {summary}")
                return summary

            return {}

        finally:
            self.disconnect()


def main():
    """메인"""
    try:
        loader = SnowflakeCopyLoader()
        result = loader.run()
        logger.info(f"최종 결과: {result}")
    except Exception as e:
        logger.error(f"실패: {e}")
        raise


if __name__ == "__main__":
    main()

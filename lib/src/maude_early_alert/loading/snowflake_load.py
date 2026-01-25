<<<<<<< HEAD
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass
import snowflake.connector
from snowflake.connector import SnowflakeConnection

@dataclass
class ColumnInfo:
    """테이블 컬럼 정보"""
    name: str
    data_type: str
    is_nullable: bool
    ordinal_position: int
    
logger = logging.getLogger(__name__)

class SnowflakeLoader:
    def __init__(self, database: str, schema: str, client: None):
        self.database = database
        self.schema = schema
        self.client = client

    def table_schema(self, table_name:str) -> List[ColumnInfo]:
        """
        Snowflake 테이블 스키마 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            List[ColumnInfo]: 컬럼 정보 리스트
        """
        query = f"""
        select
            column_name,
            data_type,
            is_nullable,
            ordinal_position
        from {self.database}.INFORMATION_SCHEMA.COLUMNS
        where table_schema = '{self.schema}'
        and table_name = '{table_name}'
        order by ordinal_position
        """
        cursor = self.client.cursor()

        try:
            logger.info(f"스키마 조회:{self.database}.{self.schema}.{table_name}")
            cursor.execute(query)
            rows = cursor.fetchall()

            if not rows:
                raise ValueError(f"테이블 {self.database}.{self.schema}.{table_name} 스키마 조회 실패")

            columns = [
                ColumnInfo(
                    name=row[0],
                    type=row[1],
                    is_nullable=(row[2] == 'YES'),
                    ordinal_default=row[3]
                )
                for row in rows
            ]
            logger.info(f"{len(columns)}개의 컬럼 조회 완료")
            return columns
        finally:
            cursor.close()

    def load_from_s3(self, table_name: str, stg_name: str):
        #1. 임시테이블생성
        #2. copy into 문 실행
        #3. merge
=======
from contextlib import contextmanager

def get_staging_table_name(table_name: str) -> str:
    """Staging 테이블명 생성

    Args:
        table_name: 원본 테이블명 (예: maude.bronze.event)

    Returns:
        Staging 테이블명 (예: maude.bronze.stg_event)
    """
    parts = table_name.split('.')
    parts[-1] = f"stg_{parts[-1]}"
    return '.'.join(parts)


class SnowflakeLoader:
    def __init__(self, database, schema, cursor = None):
        self.database = database
        self.schema = schema
        self.cursor = cursor
    
    def load_from_s3(self, table_name: str, stg_name: str):
        # 1. 임시 테이블 생성
        # 2. COPY INTO로 S3에서 임시 테이블에 적재
        # 3. MERGE로 목적 테이블로 적재
        pass
    
    def create_temporary_staging_table(self):
        pass
    
    def copy_into(self):
        pass
    
    def load_merge(self):
        pass
    
    @contextmanager
    def snowflake_transaction(self):
        """Snowflake 트랜잭션 컨텍스트
        
        Args:
            cursor: Snowflake cursor 객체
            
        Yields:
            cursor 객체
            
        Note:
            - 성공 시 COMMIT
            - 예외 발생 시 ROLLBACK 후 재발생
        """
        self.cursor.execute("BEGIN")
        try:
            yield self.cursor
            self.cursor.execute("COMMIT")
        except Exception:
            self.cursor.execute("ROLLBACK")
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
        # Create a cursor object
        cur = conn.cursor()
        
    except Exception as e:
        raise Exception(e)
>>>>>>> dffd07776c36b87873b051e2ee72204d4a0bbb35

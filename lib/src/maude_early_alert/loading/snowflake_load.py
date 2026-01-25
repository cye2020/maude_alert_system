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
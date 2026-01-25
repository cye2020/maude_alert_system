from typing import List, Tuple
from contextlib import contextmanager
import snowflake.connector
from snowflake.connector import SnowflakeConnection

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
    def __init__(self, database, schema, client: SnowflakeConnection = None):
        self.database = database
        self.schema = schema
        self.client = client
    
    def load_from_s3(self, table_name: str, stg_name: str):
        # 1. 임시 테이블 생성
        # 2. COPY INTO로 S3에서 임시 테이블에 적재
        # 3. MERGE로 목적 테이블로 적재
        pass
    
    def create_temporary_staging_table(self, table_name: str):
        stg_table_name = get_staging_table_name(table_name)
        
        table_schema = self.get_table_schema(table_name)
        column_defs = ", ".join([f"{col} {data_type}" for col, data_type in table_schema])
        
        try:
            cursor = self.client.cursor()
            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (
                    {column_defs}
                )
            """)
        finally:
            cursor.close()
        
        return stg_table_name
    
    def copy_into(self):
        pass
    
    def load_merge(self, table_name: str, stg_table_name: str, primary_key: str, columns: str):
        # UPDATE SET 절 (primary_key 제외)
        update_columns = [col for col in columns if col != primary_key]
        update_set = ", ".join([f"t.{col} = s.{col}" for col in update_columns])
        
        # INSERT 절
        insert_columns = ", ".join(columns)
        insert_values = ", ".join([f"s.{col}" for col in columns])
        
        query = f"""
            MERGE INTO {table_name} t
            USING {stg_table_name} s
            ON t.{primary_key} = s.{primary_key}
            WHEN MATCHED THEN
            UPDATE SET {update_set}
            WHEN NOT MATCHED THEN
            INSERT ({insert_columns}) VALUES ({insert_values})
        """
        
        try:
            cursor = self.client.cursor()
            
            with self.transaction(cursor):
                cursor.execute(query)
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            return cursor.fetchone()[0]
        finally:
            cursor.close()
    
    def get_table_schema(self, table_name:str) -> List[Tuple[str, str]]:
        """
        Snowflake 테이블 스키마 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            List[ColumnInfo]: 컬럼 정보 리스트
        """
        query = f"""
        SELECT
            column_name,
            data_type
        FROM {self.database}.INFORMATION_SCHEMA.COLUMNS
        WHERE table_schema = '{self.schema}'
            AND table_name = '{table_name}'
        """
        cursor = self.client.cursor()
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            
            if not rows:
                raise ValueError(f"테이블 {self.database}.{self.schema}.{table_name} 스키마 조회 실패")
        finally:
            cursor.close()

        return rows
    
    @contextmanager
    def transaction(self):
        """트랜잭션 컨텍스트
        
        Args:
            cursor: Snowflake cursor 객체
            
        Yields:
            cursor 객체
            
        Note:
            - 성공 시 COMMIT
            - 예외 발생 시 ROLLBACK 후 재발생
        """
        cursor = self.client.cursor()
        cursor.execute("BEGIN")
        try:
            yield cursor
            cursor.execute("COMMIT")
        except Exception:
            cursor.execute("ROLLBACK")
            raise
        finally:
            cursor.close()

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
        
        snowflake_loader = SnowflakeLoader(secret['database'], secret['schema'], conn)
        stg_table_name = snowflake_loader.create_temporary_staging_table('EVENT')
        print(stg_table_name)
        stg_table_schema = snowflake_loader.get_table_schema(stg_table_name)
        print(stg_table_schema)
        
    except Exception as e:
        raise Exception(e)

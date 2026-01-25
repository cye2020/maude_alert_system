from typing import Any, Dict, List, Tuple, Union
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
    def __init__(self, database, schema, conn: SnowflakeConnection = None):
        self.database = database
        self.schema = schema
        self.conn = conn
    
    def load_from_s3(
        self, table_name: str, s3_stg_table_name: str, 
        primary_key: Union[str, List[str]],
        metadata: Dict[str, Any] = None,
        business_primary_key: Union[str, List[str]] = None
    ):
        table_schema = self.get_table_schema(table_name)
        
        # 1. 임시 테이블 생성
        stg_table_name = self.create_temporary_staging_table(table_name, table_schema)
        
        # 2. COPY INTO로 S3에서 임시 테이블에 적재
        self.copy_into(stg_table_name, s3_stg_table_name, table_schema)
        
        # 3. MERGE로 목적 테이블로 적재
        column_names = [col_name for col_name, _ in table_schema]
        self.load_merge(table_name, stg_table_name, primary_key, column_names)
    
    def create_temporary_staging_table(self, table_name: str, table_schema: list = None):
        stg_table_name = get_staging_table_name(table_name)
        
        if not table_schema:
            table_schema = self.get_table_schema(table_name)
        
        column_defs = ", ".join([f"{col} {data_type}" for col, data_type in table_schema])
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"""
                CREATE OR REPLACE TEMPORARY TABLE {stg_table_name} (
                    {column_defs}
                )
            """)
        finally:
            cursor.close()
        
        return stg_table_name
    
    def copy_into(
        self,
        stg_table_name: str, # 임시 스테이지 테이블명
        s3_stg_table_name : str,  # S3 스테이지 경로
        table_schema: List[Tuple[str, str]] = None,
        file_format : str = 'JSON' # 파일 형식
    ) -> dict:

        """
        S3에서 Snowflake로 데이터 복사
        
        Args:
            table_name: 원본 테이블명 (EVENT, UDI)
            stage_path: S3 스테이지 경로 ()
            file_format: 파일 형식 (JSON, CSV, AVRO, ORC, PARQUET, XML)
            
        Returns:
            dict: {
                'rows_loaded': 적재된 행 수,
                'rows_parsed': 파싱된 행 수
            }
        """
        if not table_schema:
            table_schema = self.get_table_schema(stg_table_name)
        
        #컬럼 정의
        column_names = [col_name for col_name, _ in table_schema]
        select_items = []

        #JSON 파싱 문법
        if file_format == 'JSON':
            # $1:col_name 구문은 올바른 Snowflake JSON 파싱 문법
            for col_name, col_type in table_schema:
                # JSON 키는 대소문자를 구분하므로 정확한 키명 사용
                select_items.append(f"$1:{col_name}::{col_type} AS {col_name}")
        else:
            # CSV 등 위치 기반 형식
            for i, (col_name, col_type) in enumerate(table_schema):
                select_items.append(f"$1:{i+1}::{col_type} AS {col_name}")
        
        # 쿼리 생성
        copy_query = f"""
        COPY INTO {stg_table_name} (
            {', '.join(column_names)}
        )
        FROM (
            SELECT {',\n'.join(select_items)}
            FROM {s3_stg_table_name}
        )
        FILE_FORMAT = (TYPE = '{file_format}')
        ON_ERROR = 'CONTINUE'
        """
        
        try:
            cursor = self.conn.cursor()
            
            cursor.execute(f"USE DATABASE {self.database}")
            cursor.execute(f"USE SCHEMA {self.schema}")
            
            cursor.execute(copy_query)
            result = cursor.fetchone()

            if result:
                rows_loaded = result[3] if len(result) > 3 else 0
                
                return {
                    'rows_loaded': rows_loaded,
                    'rows_parsed': result[2] if len(result) > 2 else 0
                }
            else:
                return {'rows_loaded': 0, 'rows_parsed': 0}
                
        except Exception as e:
            msg = f"""
            데이터 복사 실패: {e}
            쿼리: {copy_query}
            """
            raise Exception(msg)

        finally:
            cursor.close()

    def load_merge(self, table_name: str, stg_table_name: str, primary_key: Union[str, List[str]], column_names: List[str]):
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
        
        try:
            cursor = self.conn.cursor()
            
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
        FROM {self.database}.INFORMATION_SCHEMA.column_names
        WHERE table_schema = '{self.schema}'
            AND table_name = '{table_name}'
        """
        cursor = self.conn.cursor()
        
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
        cursor = self.conn.cursor()
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
        table_name = 'EVENT'
        s3_stg_table_name = 'BRONZE_S3_STAGE'
        primary_key = []
        snowflake_loader.load_from_s3()
        
    except Exception as e:
        raise Exception(e)

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

# class SnowflakeLoader:
#     def __init__(self, database: str, schema: str, client: None):
#         self.database = database
#         self.schema = schema
#         self.client = client

#     def load_from_s3(self, table_name: str, stg_name: str):
#         #1. 임시테이블생성
#         #2. copy into 문 실행

import re
import sys
from pathlib import Path
import snowflake.connector

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 직접 모듈 import (src.__init__을 거치지 않음)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "snowflake_config",
    project_root / "src" / "snowflake" / "config.py"
)
snowflake_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(snowflake_config_module)
get_snowflake_config = snowflake_config_module.get_snowflake_config


def generate_snowflake_columns_sql(
    conn,
    database: str,
    schema: str,
    table: str,
    *,
    strip_type_numbers: bool = True,
    include_not_null: bool = True,
    save_to_file: bool = False,
    out_dir: str = "./schema_out"
) -> str:
    """
    Snowflake 테이블 스키마를 조회해서
    - 컬럼명 + 타입 SQL 스니펫 생성
    - (옵션) 타입 뒤 숫자 제거
    - (옵션) NOT NULL 포함
    - (옵션) 파일로 저장

    return: SQL 컬럼 정의 문자열
    """

    def _strip_numbers(type_str: str) -> str:
        return re.sub(r"\s*\(.*\)\s*$", "", type_str).strip()

    cur = conn.cursor()
    try:
        cur.execute(f'SHOW COLUMNS IN TABLE "{database}"."{schema}"."{table}"')
        rows = cur.fetchall()
        colnames = [d[0].lower() for d in cur.description]

        i_name = colnames.index("column_name") if "column_name" in colnames else colnames.index("name")
        i_type = colnames.index("type")
        i_null = colnames.index("null?") if "null?" in colnames else None

        lines = []
        for r in rows:
            name = r[i_name]
            typ = r[i_type]
            nullflag = r[i_null] if i_null is not None else None

            if strip_type_numbers:
                typ = _strip_numbers(typ)

            not_null = ""
            if include_not_null and str(nullflag).upper() in ("N", "NO"):
                not_null = " NOT NULL"

            lines.append(f"    {name} {typ}{not_null}")

        result = ",\n".join(lines)

        if save_to_file:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            path = Path(out_dir) / f"{database}.{schema}.{table}.columns.sql"
            path.write_text(result + "\n", encoding="utf-8")

        return result

    finally:
        cur.close()


if __name__ == "__main__":
    # Snowflake 연결 설정
    config = get_snowflake_config()
    conn_params = config.get_connection_params()
    
    # Snowflake 연결 생성
    conn = snowflake.connector.connect(**conn_params)
    
    try:
        # SQL 컬럼 정의 생성
        sql_columns = generate_snowflake_columns_sql(
            conn,
            database="MAUDE",
            schema="BRONZE",
            table="MENU"
        )
        
        print(sql_columns)
    finally:
        conn.close()


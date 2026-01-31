"""Snowflake Flatten SQL 생성"""

from pathlib import Path
from typing import List
import sys


# ============================================================
# SQL 생성 함수 (순수 함수 - Snowflake 연결 불필요)
# ============================================================

def sanitize(name: str) -> str:
    """컬럼명을 SQL 안전 이름으로 변환"""
    return name.replace(":", "_").replace("-", "_").replace(".", "_").replace(" ", "_")


def generate_flatten_sql(
    table_name: str,
    raw_column: str = "raw_data",
    scalar_keys: List[str] = None,
    patient_keys: List[str] = None,
    device_keys: List[str] = None,
    mdr_text_keys: List[str] = None,
    device_outer: bool = True,
) -> str:
    """Snowflake Flatten SQL 생성
    
    Args:
        table_name: 테이블명 (예: MAUDE.BRONZE.EVENT)
        raw_column: JSON 컬럼명
        scalar_keys: 최상위 스칼라 키들
        patient_keys: patient[0]에서 추출할 키들
        device_keys: device 배열에서 추출할 키들 (LATERAL FLATTEN)
        mdr_text_keys: mdr_text 배열에서 추출할 키들 (TRANSFORM)
        device_outer: device FLATTEN 시 OUTER JOIN 사용 여부
    
    Returns:
        SELECT SQL 문자열
    """
    sections = []
    
    # 1) 최상위 스칼라 키
    if scalar_keys:
        scalar_cols = [
            f"    {raw_column}:{key}::STRING AS {sanitize(key)}"
            for key in sorted(scalar_keys)
        ]
        sections.append(
            "    -- ================================================\n"
            "    -- Top-level Scalar Columns\n"
            "    -- ================================================\n"
            + ",\n".join(scalar_cols)
        )
    
    # 2) patient[0] 키들
    if patient_keys:
        patient_cols = [
            f"    {raw_column}:patient[0]:{key}::STRING AS patient_{sanitize(key)}"
            for key in sorted(patient_keys)
        ]
        sections.append(
            "\n    -- ================================================\n"
            "    -- Patient Information (patient[0])\n"
            "    -- ================================================\n"
            + ",\n".join(patient_cols)
        )
    
    # 3) mdr_text 배열 → TRANSFORM
    if mdr_text_keys:
        mdr_cols = []
        for key in sorted(mdr_text_keys):
            alias = "mdr_text_keys" if key == "mdr_text_key" else f"mdr_text_{sanitize(key)}s"
            mdr_cols.append(
                f"    TRANSFORM({raw_column}:mdr_text, x -> x:{key}::STRING) AS {alias}"
            )
        sections.append(
            "\n    -- ================================================\n"
            "    -- MDR Text Array (TRANSFORM)\n"
            "    -- ================================================\n"
            + ",\n".join(mdr_cols)
        )
    
    # 4) device 배열 → LATERAL FLATTEN
    if device_keys:
        device_cols = [
            f"    d.value:{key}::STRING AS device_{sanitize(key)}"
            for key in sorted(device_keys)
        ]
        sections.append(
            "\n    -- ================================================\n"
            "    -- Device Information (LATERAL FLATTEN)\n"
            "    -- ================================================\n"
            + ",\n".join(device_cols)
        )
        
        outer = "TRUE" if device_outer else "FALSE"
        from_clause = (
            f"FROM {table_name}\n"
            f"    , LATERAL FLATTEN(input => {raw_column}:device, OUTER => {outer}) AS d"
        )
    else:
        from_clause = f"FROM {table_name}"
    
    # SQL 조립
    select_body = ",\n".join(sections)
    
    return f"""-- ====================================================================
-- Snowflake Flatten SQL
-- Table: {table_name}
-- ====================================================================

SELECT
{select_body}
{from_clause};
"""


# ============================================================
# Snowflake 스키마 조회 (snowflake_load 활용)
# ============================================================

def fetch_schema_and_generate_sql(table_name: str, raw_column: str = "raw_data") -> str:
    """Snowflake에서 스키마 조회 후 SQL 생성
    
    Args:
        table_name: 테이블명
        raw_column: JSON 컬럼명
        
    Returns:
        생성된 SQL 문자열
    """
    # snowflake_load 임포트
    try:
        from maude_early_alert.loading.snowflake_load import SnowflakeLoader
        import snowflake.connector
    except ImportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 자격증명 로드 (.env 파일에서)
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        print(f"Error: .env 파일이 없습니다: {env_path}", file=sys.stderr)
        sys.exit(1)
    
    # .env 파싱
    secret = {}
    required = ('account', 'user', 'password', 'database', 'schema', 'warehouse')
    
    content = env_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            secret[key.strip().lower()] = value.strip().strip('"').strip("'")
    
    # 필수 키 확인
    missing = [k for k in required if k not in secret]
    if missing:
        print(f"Error: .env 파일에 필수 키가 없습니다: {missing}", file=sys.stderr)
        sys.exit(1)
    
    from maude_early_alert.utils.secrets import get_secret
    # Snowflake 연결
    try:
        secret = get_secret('snowflake/bronze/credentials')
        conn = snowflake.connector.connect(
            user=secret['user'],
            password=secret['password'],
            account=secret['account'],
            warehouse=secret['warehouse'],
            database=secret['database'],
            schema=secret['schema']
        )
        cursor = conn.cursor()
    except Exception as e:
        # MFA 필요한 경우
        if "TOTP" in str(e) or "MFA" in str(e):
            print("\nMFA 인증이 필요합니다.", file=sys.stderr)
            totp = input("TOTP 코드 입력 (6자리): ").strip()
            try:
                conn = snowflake.connector.connect(
                    user=secret['user'],
                    password=secret['password'],
                    account=secret['account'],
                    warehouse=secret['warehouse'],
                    database=secret['database'],
                    schema=secret['schema'],
                    authenticator="USERNAME_PASSWORD_MFA",
                    passcode=totp
                )
                cursor = conn.cursor()
            except Exception:
                try:
                    conn = snowflake.connector.connect(
                        user=secret['user'],
                        password=secret['password'] + totp,
                        account=secret['account'],
                        warehouse=secret['warehouse'],
                        database=secret['database'],
                        schema=secret['schema'],
                        passcode_in_password=True
                    )
                    cursor = conn.cursor()
                except Exception as e2:
                    print(f"Error: MFA 인증 실패 - {e2}", file=sys.stderr)
                    sys.exit(1)
        else:
            print(f"Error: Snowflake 연결 실패 - {e}", file=sys.stderr)
            sys.exit(1)
    
    # 스키마 조회
    top = SnowflakeLoader.top_keys_with_type(cursor, table_name, raw_column)
    device = SnowflakeLoader.device_keys_with_type(cursor, table_name, raw_column)
    patient = SnowflakeLoader.patient_keys_with_type(cursor, table_name, raw_column)
    mdr_text = SnowflakeLoader.mdr_text_keys_with_type(cursor, table_name, raw_column)
    
    cursor.close()
    conn.close()
    
    # 스칼라 키만 추출
    scalar_keys = [k for k, t in top.items() if t.upper() not in ("ARRAY", "OBJECT")]
    
    # SQL 생성
    return generate_flatten_sql(
        table_name=table_name,
        raw_column=raw_column,
        scalar_keys=scalar_keys,
        patient_keys=list(patient.keys()),
        device_keys=list(device.keys()),
        mdr_text_keys=list(mdr_text.keys())
    )


# ============================================================
# 메인 실행
# ============================================================

if __name__ == "__main__":
    TABLE_NAME = "MAUDE.BRONZE.EVENT"
    RAW_COLUMN = "raw_data"
    
    sql = fetch_schema_and_generate_sql(TABLE_NAME, RAW_COLUMN)
    print(sql)
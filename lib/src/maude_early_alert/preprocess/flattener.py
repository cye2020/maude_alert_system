"""Snowflake Flatten SQL 생성"""

from ast import Dict
from pathlib import Path
from typing import List, Dict, Union
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
    patient_keys: Union[List[str], Dict[str, str]] = None,
    device_keys: Union[List[str], Dict[str, str]] = None,
    device_openfda_keys: Dict[str, str] = None,
    mdr_text_keys: List[str] = None,
    device_outer: bool = True,
) -> str:
    """Snowflake Flatten SQL 생성
    
    Args:
        table_name: 테이블명 (예: MAUDE.BRONZE.EVENT)
        raw_column: JSON 컬럼명
        scalar_keys: 최상위 스칼라 키들
        patient_keys: patient[0]에서 추출할 키들 (리스트 또는 key->TYPE 딕셔너리)
        device_keys: device 배열에서 추출할 키들 (리스트 또는 key->TYPE 딕셔너리)
        device_openfda_keys: deivce_openfda_key 배열에서 추출할 키들
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
    
    # 2) patient[0] 키들 (TYPE 구분: ARRAY → TRANSFORM, 그 외 → ::STRING)
    if patient_keys:
        _keys = sorted(patient_keys.keys() if isinstance(patient_keys, dict) else patient_keys)
        patient_cols = []
        for key in _keys:
            dtype = (patient_keys.get(key) or "VARCHAR").upper() if isinstance(patient_keys, dict) else "VARCHAR"
            if dtype == "ARRAY":
                patient_cols.append(
                    f"    TRANSFORM({raw_column}:patient[0]:{key}, x -> x::STRING) AS patient_{sanitize(key)}"
                )
            else:
                patient_cols.append(
                    f"    {raw_column}:patient[0]:{key}::STRING AS patient_{sanitize(key)}"
                )
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
    
    # 4) device 배열 → LATERAL FLATTEN (TYPE 구분: ARRAY → TRANSFORM, 그 외 → ::STRING)
    # openfda는 아래 device_openfda_keys에서 별도 처리하므로 제외
    if device_keys:
        _keys = sorted(device_keys.keys() if isinstance(device_keys, dict) else device_keys)
        _keys = [k for k in _keys if k != "openfda"]
        device_cols = []
        for key in _keys:
            dtype = (device_keys.get(key) or "VARCHAR").upper() if isinstance(device_keys, dict) else "VARCHAR"
            if dtype == "ARRAY":
                device_cols.append(
                    f"    TRANSFORM(d.value:{key}, x -> x::STRING) AS device_{sanitize(key)}"
                )
            else:
                device_cols.append(
                    f"    d.value:{key}::STRING AS device_{sanitize(key)}"
                )
        sections.append(
            "\n    -- ================================================\n"
            "    -- Device Information (LATERAL FLATTEN)\n"
            "    -- ================================================\n"
            + ",\n".join(device_cols)
        )

    # deivce 속 배열 openfda 배열 
    if device_openfda_keys:
        openfda_scalar_cols =[]
        openfda_array_cols = []

        for key in sorted(device_openfda_keys.keys()):
            dtype = device_openfda_keys[key]

            if dtype.upper() == "ARRAY":
                # 배열 필트는 TRANSFORM 사용
                openfda_array_cols.append(
                    f"    TRANSFORM(d.value:openfda:{key} , x-> x::STRING) AS device_openfda_{sanitize(key)}"
                )
            else:
                openfda_scalar_cols.append(
                    f"    d.value:openfda:{key}::STRING AS device_openfda_{sanitize(key)}"
                )
        if openfda_scalar_cols or openfda_array_cols:
            all_openfda_cols = openfda_scalar_cols + openfda_array_cols
            sections.append(
                "\n    -- ================================================\n"
                "    -- Device OpenFDA (Nested Object)\n"
                "    -- ================================================\n"
                + ",\n".join(all_openfda_cols)
            )

    if device_keys or device_openfda_keys:
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
    device_openfda = SnowflakeLoader.device_openfda_keys_with_type(cursor, table_name, raw_column)
    patient = SnowflakeLoader.patient_keys_with_type(cursor, table_name, raw_column)
    mdr_text = SnowflakeLoader.mdr_text_keys_with_type(cursor, table_name, raw_column)
    
    cursor.close()
    conn.close()
    
    # 최상위 키: device/patient/mdr_text만 제외 (나머지는 모두 포함, ARRAY/OBJECT도 ::STRING으로 출력)
    # TYPE으로 제외하면 product_problems, remedial_action 등 top-level ARRAY 컬럼이 아예 누락됨
    EXCLUDE_KEYS = {'device', 'patient', 'mdr_text'}
    scalar_keys = [k for k in top.keys() if k not in EXCLUDE_KEYS]

    # SQL 생성 (device/patient는 key->TYPE 딕셔너리 그대로 전달해 TYPE별 처리)
    return generate_flatten_sql(
        table_name=table_name,
        raw_column=raw_column,
        scalar_keys=scalar_keys,
        patient_keys=patient,
        device_keys=device,
        device_openfda_keys=device_openfda,
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
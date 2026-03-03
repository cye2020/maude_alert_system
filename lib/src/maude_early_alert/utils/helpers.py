import re
import datetime
from typing import Any, List, Union

_IDENTIFIER_RE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$')
_DATE_RE       = re.compile(r'^\d{4}-\d{2}-\d{2}$')


def validate_identifier(name: str) -> str:
    """SQL 식별자 검증 (테이블명, 컬럼명, 스키마명 등)

    Snowflake SQL에 f-string으로 삽입되는 식별자의 안전성을 보장합니다.
    영문, 숫자, 언더스코어, 점(dotted name)만 허용합니다.

    Args:
        name: 검증할 식별자

    Returns:
        검증 통과한 원본 문자열

    Raises:
        ValueError: 허용되지 않는 문자가 포함된 경우
    """
    if not _IDENTIFIER_RE.match(name):
        raise ValueError(f"Invalid SQL identifier: {name!r}")
    return name


def validate_date(value: str) -> str:
    """날짜 형식 검증 (YYYY-MM-DD)

    SQL f-string에 삽입되는 날짜 값의 안전성을 보장합니다.

    Args:
        value: 검증할 날짜 문자열

    Returns:
        검증 통과한 원본 문자열

    Raises:
        ValueError: YYYY-MM-DD 형식이 아니거나 실존하지 않는 날짜인 경우
    """
    if not _DATE_RE.match(value):
        raise ValueError(f"유효하지 않은 날짜 형식 (YYYY-MM-DD 필요): {value!r}")
    datetime.datetime.strptime(value, "%Y-%m-%d")  # 실존 날짜 검증 (예: 2024-02-30 거부)
    return value


def ensure_list(value: Union[str, List[str]]) -> List[str]:
    """문자열 또는 리스트를 리스트로 정규화"""
    return [value] if isinstance(value, str) else value


def format_sql_literal(key: str, value: Any) -> str:
    """Python 값을 SQL SELECT 표현식으로 변환

    Args:
        key: 컬럼명
        value: 메타데이터 값

    Returns:
        SQL SELECT 표현식 (예: "'value' AS column_name")
    """
    validate_identifier(key)
    if value is None:
        return f"NULL AS {key}"
    elif isinstance(value, datetime.datetime):
        formatted = value.strftime('%Y-%m-%d %H:%M:%S')
        return f"'{formatted}'::TIMESTAMP_TZ AS {key}"
    elif isinstance(value, bool):
        return f"{str(value).upper()} AS {key}"
    elif isinstance(value, (int, float)):
        return f"{value} AS {key}"
    else:
        escaped = str(value).replace("'", "''")
        return f"'{escaped}' AS {key}"

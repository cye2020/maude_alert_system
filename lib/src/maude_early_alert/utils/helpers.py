import datetime
from typing import Any, List, Union


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

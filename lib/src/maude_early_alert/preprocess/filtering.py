"""필터링 SQL 생성

WHERE / QUALIFY 절을 조립하여 Snowflake SQL을 생성한다.
- WHERE 조건: 값 필터링, 컬럼 비교, NOT EXISTS 등
- QUALIFY 조건: 중복 제거 (ROW_NUMBER)
"""

from typing import List

from maude_early_alert.utils.helpers import validate_identifier, ensure_list


def generate_filter_sql(
    source: str, alias: str,
    where: List[str] = None, qualify: List[str] = None
) -> str:
    """WHERE / QUALIFY 절 → Snowflake SQL

    Args:
        source: 소스 테이블명
        alias: 테이블 별칭
        where: WHERE 조건 리스트 (AND로 결합)
        qualify: QUALIFY 조건 리스트 (AND로 결합)

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)

    sql = f"SELECT *\nFROM {source} {alias}"

    if where:
        where = ensure_list(where)
        sql += "\nWHERE " + "\n  AND ".join(c.strip() for c in where)

    if qualify:
        qualify = ensure_list(qualify)
        sql += "\nQUALIFY " + "\n  AND ".join(c.strip() for c in qualify)

    return sql


if __name__=='__main__':
    source = 'EVENT'
    alias = 'e'
    where = "DATE(e.ingest_time, 'YYYYMM') = 202602"
    qualify = "ROW_NUMBER() OVER (PARTITION BY s.mdr_report_key, s.record_hash ORDER BY s.mdr_report_key) = 1"
    sql = generate_filter_sql(source, alias, where=where)
    print(sql)
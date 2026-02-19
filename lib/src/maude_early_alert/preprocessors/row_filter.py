"""필터링 SQL 생성

WHERE / QUALIFY 절을 조립하여 Snowflake SQL을 생성한다.
- WHERE 조건: 값 필터링, 컬럼 비교, NOT EXISTS 등
- QUALIFY 조건: 중복 제거 (ROW_NUMBER)
"""

from typing import Dict, List

from maude_early_alert.utils.helpers import validate_identifier, ensure_list
from maude_early_alert.utils.sql_builder import build_cte_sql


def build_filter_sql(
    source: str, alias: str = "",
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


def build_filter_pipeline(source: str, ctes: List[dict], final: str, alias: str = "s") -> str:
    """CTE 체인 SQL 생성

    filtering.yaml의 chain 스텝 설정을 받아 WITH ... SELECT 형태의 SQL을 반환한다.
    각 CTE의 from 필드가 'source'이면 source 인자로 대체한다.

    Args:
        source: 최초 소스 테이블명 (cte의 from: source 대체값)
        ctes: filtering.yaml QUALITY_FILTER.ctes 리스트
        final: 마지막 SELECT 대상 CTE명
        alias: 테이블 별칭

    Returns:
        완성된 CTE 체인 SQL 문자열
    """
    built_ctes = []
    for cte in ctes:
        from_source = source if cte['from'] == 'source' else cte['from']
        query = build_filter_sql(
            from_source, alias=alias,
            where=cte.get('where'),
            qualify=cte.get('qualify'),
        )
        built_ctes.append({'name': cte['name'], 'query': query})
    return build_cte_sql(ctes=built_ctes, final_query=f"SELECT * FROM {final}")


if __name__ == '__main__':
    print("=== build_filter_sql (WHERE) ===")
    print(build_filter_sql(
        source='my_table',
        where=["col_a IS NOT NULL", "col_b > 0"],
    ))

    print("\n=== build_filter_sql (QUALIFY 중복 제거) ===")
    print(build_filter_sql(
        source='my_table',
        qualify=["ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) = 1"],
    ))

    print("\n=== build_filter_pipeline (CTE 체인) ===")
    ctes = [
        {'name': 'step1', 'from': 'source', 'qualify': ["ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) = 1"]},
        {'name': 'step2', 'from': 'step1', 'where': ["status IS NOT NULL"]},
    ]
    print(build_filter_pipeline(source='my_table', ctes=ctes, final='step2'))

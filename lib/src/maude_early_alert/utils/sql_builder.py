"""
SQL 빌더 유틸리티
CTE 구조의 SQL을 생성하는 범용 함수
"""
from textwrap import dedent, indent
from typing import List, Optional


def _normalize(sql: str) -> str:
    """dedent + strip 으로 들여쓰기 정규화"""
    return dedent(sql).strip()


def _build_with_clause(ctes: List[dict]) -> str:
    """WITH 절 생성 (각 CTE query에 dedent 자동 적용)"""
    cte_definitions = []
    for cte in ctes:
        query = _normalize(cte['query'])
        indented = indent(query, '\t')
        cte_definitions.append(f"{cte['name']} AS (\n{indented}\n)")
    return "WITH\n" + ",\n".join(cte_definitions)


def build_cte_sql(
    ctes: List[dict],
    from_clause: Optional[str] = None,
    table_alias: Optional[str] = None,
    select_cols: Optional[List[str]] = None,
    replace_cols: Optional[List[str]] = None,
    joins: Optional[List[str]] = None,
    where: Optional[str] = None,
    distinct: bool = False,
    final_query: Optional[str] = None,
) -> str:
    """
    CTE 구조의 SQL 생성

    두 가지 모드:
      1) final_query 지정 → WITH ... <final_query> (자유 형식)
      2) from_clause 지정 → WITH ... SELECT ... FROM ... (기존 방식)

    각 CTE의 query 문자열에는 dedent가 자동 적용되므로
    호출부에서 들여쓰기를 신경쓰지 않아도 됩니다.

    Args:
        ctes: CTE 정의 리스트. 각 항목은 {'name': str, 'query': str} 형태
        final_query: 최종 쿼리 (UNION ALL 등 자유 형식). 지정 시 아래 인자 무시
        from_clause: FROM 절 (예: "table_name t")
        table_alias: 테이블 alias (예: "t"). replace_cols 사용 시 필요
        select_cols: SELECT 절에 들어갈 컬럼 리스트
        replace_cols: SELECT * REPLACE 절 컬럼 리스트 (Snowflake 전용)
        joins: JOIN 절 리스트
        where: WHERE 절
        distinct: SELECT DISTINCT 여부

    Returns:
        생성된 SQL 문자열
    """
    # 모드 1: final_query (자유 형식)
    if final_query is not None:
        parts = []
        if ctes:
            parts.append(_build_with_clause(ctes))
        parts.append(_normalize(final_query))
        return "\n".join(parts)

    # 모드 2: from_clause (기존 방식)
    if from_clause is None:
        raise ValueError("from_clause 또는 final_query 중 하나는 필수입니다")
    if select_cols and replace_cols:
        raise ValueError("select_cols와 replace_cols는 동시에 사용할 수 없습니다")

    parts = []

    # WITH 절
    if ctes:
        parts.append(_build_with_clause(ctes))

    # SELECT 절
    keyword = "SELECT DISTINCT" if distinct else "SELECT"
    if replace_cols:
        replace_clause = indent(",\n".join(replace_cols), '\t')
        star = f"{table_alias}.*" if table_alias else "*"
        parts.append(f"{keyword} {star} REPLACE (\n{replace_clause}\n)")
    elif select_cols:
        select_clause = indent(",\n".join(select_cols), '\t')
        parts.append(f"{keyword}\n{select_clause}")
    else:
        parts.append(f"{keyword}\n\t*")

    # FROM 절
    parts.append("FROM\n" + indent(from_clause, '\t'))

    # JOIN 절
    if joins:
        parts.append(indent("\n".join(joins), '\t'))

    # WHERE 절
    if where:
        parts.append("WHERE\n" + indent(where, '\t'))

    return "\n".join(parts)


if __name__ == '__main__':
    # 테스트 1: 기존 방식 (from_clause) — dedent 자동 적용 확인
    ctes = [
        {
            'name': 'mode_1',
            'query': """\
                SELECT
                    product_code,
                    MODE(device_name) AS mode_device_name
                FROM
                    MAUDE.SILVER.EVENT
                WHERE
                    product_code IS NOT NULL
                    AND device_name IS NOT NULL
                GROUP BY
                    product_code"""
        }
    ]

    sql = build_cte_sql(
        ctes=ctes,
        from_clause='MAUDE.SILVER.EVENT t',
        table_alias='t',
        replace_cols=[
            'COALESCE(t.device_name, mode_1.mode_device_name) AS device_name'
        ],
        joins=['LEFT JOIN mode_1 ON t.product_code = mode_1.product_code']
    )
    print("=== 모드 1: from_clause ===")
    print(sql)

    # 테스트 2: 새로운 방식 (final_query)
    sql2 = build_cte_sql(
        ctes=[
            {'name': 'a', 'query': 'SELECT 1 AS x'},
            {'name': 'b', 'query': 'SELECT 2 AS x'},
        ],
        final_query="""\
            SELECT * FROM a
            UNION ALL
            SELECT * FROM b"""
    )
    print("\n=== 모드 2: final_query ===")
    print(sql2)

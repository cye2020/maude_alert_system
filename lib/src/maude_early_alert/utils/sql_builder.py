"""
SQL 빌더 유틸리티
CTE 구조의 SQL을 생성하는 범용 함수
"""
from textwrap import dedent
from typing import List, Optional


def build_cte_sql(
    ctes: List[dict],
    from_clause: str,
    table_alias: Optional[str] = None,
    select_cols: Optional[List[str]] = None,
    replace_cols: Optional[List[str]] = None,
    joins: Optional[List[str]] = None,
    where: Optional[str] = None,
) -> str:
    """
    CTE 구조의 SQL 생성

    Args:
        ctes: CTE 정의 리스트. 각 항목은 {'name': str, 'query': str} 형태
        from_clause: FROM 절 (예: "table_name t")
        table_alias: 테이블 alias (예: "t"). replace_cols 사용 시 필요
        select_cols: SELECT 절에 들어갈 컬럼 리스트 (replace_cols와 함께 사용 불가)
        replace_cols: SELECT t.* REPLACE 절에 들어갈 컬럼 리스트 (Snowflake 전용)
        joins: JOIN 절 리스트 (예: ["LEFT JOIN cte1 ON ..."])
        where: WHERE 절 (예: "t.col IS NOT NULL")

    Returns:
        생성된 SQL 문자열

    Examples:
        # 일반 SELECT
        >>> sql = build_cte_sql(
        ...     ctes=[{'name': 'cte1', 'query': '...'}],
        ...     from_clause='my_table t',
        ...     select_cols=['t.id', 't.name'],
        ... )

        # SELECT t.* REPLACE (Snowflake)
        >>> sql = build_cte_sql(
        ...     ctes=[{'name': 'mode_1', 'query': '...'}],
        ...     from_clause='my_table t',
        ...     table_alias='t',
        ...     replace_cols=['COALESCE(t.val, mode_1.mode_val) AS val'],
        ...     joins=['LEFT JOIN mode_1 ON t.col = mode_1.col']
        ... )
    """
    if select_cols and replace_cols:
        raise ValueError("select_cols와 replace_cols는 동시에 사용할 수 없습니다")

    parts = []

    # WITH 절
    if ctes:
        cte_definitions = []
        for cte in ctes:
            indented_query = "\n    ".join(cte['query'].split("\n"))
            cte_definitions.append(f"{cte['name']} AS (\n    {indented_query}\n)")
        parts.append("WITH\n" + ",\n".join(cte_definitions))

    # SELECT 절
    if replace_cols:
        replace_clause = ",\n        ".join(replace_cols)
        star = f"{table_alias}.*" if table_alias else "*"
        parts.append(f"SELECT {star} REPLACE (\n        {replace_clause}\n    )")
    elif select_cols:
        select_clause = ",\n    ".join(select_cols)
        parts.append(f"SELECT\n    {select_clause}")
    else:
        parts.append("SELECT\n    *")

    # FROM 절
    parts.append(f"FROM\n    {from_clause}")

    # JOIN 절
    if joins:
        parts.append("    " + "\n    ".join(joins))

    # WHERE 절
    if where:
        parts.append(f"WHERE\n    {where}")

    return "\n".join(parts)


if __name__ == '__main__':
    # 테스트: SELECT * REPLACE 패턴
    ctes = [
        {
            'name': 'mode_1',
            'query': dedent("""\
                SELECT
                    product_code,
                    MODE(device_name) AS mode_device_name
                FROM
                    MAUDE.SILVER.EVENT
                WHERE
                    product_code IS NOT NULL
                    AND device_name IS NOT NULL
                GROUP BY
                    product_code""")
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

    print(sql)

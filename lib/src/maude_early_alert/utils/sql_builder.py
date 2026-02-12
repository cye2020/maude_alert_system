"""
SQL 빌더 유틸리티
CTE 구조의 SQL을 생성하는 범용 함수
JOIN 절 생성 및 MDR 추출·적재 CTE 연결
"""
from textwrap import dedent, indent
from typing import List, Optional, Union


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


def build_insert_sql(
    table_name: str,
    columns: List[str],
    num_rows: int = 1,
) -> str:
    """
    INSERT INTO SQL 문자열 생성.

    Snowflake connector의 executemany()와 함께 쓸 때는 num_rows=1로 고정하고
    cursor.executemany(sql, list_of_tuples) 패턴으로 사용합니다.

    Args:
        table_name: INSERT 대상 테이블명
        columns: 컬럼 이름 리스트 (순서가 데이터 튜플 순서와 일치해야 함)
        num_rows: VALUES 절에 생성할 행 수. executemany 사용 시 1 고정

    Returns:
        "INSERT INTO table (col1, col2) VALUES (%s, %s)" 형태 문자열

    Example:
        sql = build_insert_sql("MY_TABLE", ["COL_A", "COL_B"])
        cursor.executemany(sql, [("val1", "val2"), ("val3", "val4")])
    """
    col_clause = ", ".join(columns)
    row_placeholder = "(" + ", ".join(["%s"] * len(columns)) + ")"

    if num_rows == 1:
        return f"INSERT INTO {table_name} ({col_clause}) VALUES {row_placeholder}"

    # num_rows > 1: 단건 execute로 여러 행 한번에 INSERT할 때
    values_clause = ", ".join([row_placeholder] * num_rows)
    return f"INSERT INTO {table_name} ({col_clause}) VALUES {values_clause}"


def build_join_clause(
    left_table: str,
    right_table: str,
    on_columns: Union[str, List[str]],
    join_type: str = "LEFT",
    left_alias: Optional[str] = None,
    right_alias: Optional[str] = None,
) -> str:
    """범용 JOIN 절 한 줄 생성 (예: EVENT_STAGE_12 에 추출 결과 LEFT JOIN).

    Args:
        left_table: 왼쪽 테이블 (또는 CTE 이름)
        right_table: 오른쪽 테이블 (또는 CTE 이름)
        on_columns: ON 조건에 쓸 컬럼명(들). 하나면 단일 컬럼 동등 조인, 리스트면 AND 로 연결
        join_type: JOIN 종류 (LEFT, INNER, RIGHT 등)
        left_alias: 왼쪽 alias (없으면 테이블명 그대로)
        right_alias: 오른쪽 alias (없으면 테이블명 그대로)

    Returns:
        "LEFT JOIN right_table r ON l.MDR_TEXT = r.MDR_TEXT" 형태 문자열
    """
    join_type = join_type.strip().upper()
    cols = [on_columns] if isinstance(on_columns, str) else list(on_columns)
    l_a = left_alias or left_table
    r_a = right_alias or right_table
    on_parts = [f"{l_a}.{c} = {r_a}.{c}" for c in cols]
    on_clause = " AND ".join(on_parts)
    return f"{join_type} JOIN {right_table} {r_a} ON {on_clause}"


def build_extract_join_sql(
    extract_cte_name: str,
    extract_cte_query: str,
    base_table: str,
    join_on_column: str = "MDR_TEXT",
    base_alias: str = "e",
    extract_alias: str = "ex",
    select_columns: Optional[List[str]] = None,
) -> str:
    """MDR_TEXT 추출 CTE + EVENT_STAGE_12(또는 base_table) LEFT JOIN 한 SQL 생성.

    - 1번 CTE: 추출용 (extract_cte_query → extract_cte_name)
    - 최종: base_table base_alias LEFT JOIN extract_cte_name extract_alias ON e.MDR_TEXT = ex.MDR_TEXT

    Args:
        extract_cte_name: 추출 결과 CTE 이름
        extract_cte_query: 추출용 SELECT 쿼리 (예: build_mdr_text_extract_sql 결과)
        base_table: 기준 테이블 (예: EVENT_STAGE_12)
        join_on_column: JOIN ON 에 쓸 컬럼명
        base_alias: 기준 테이블 alias
        extract_alias: 추출 CTE alias
        select_columns: 최종 SELECT 컬럼 (None 이면 base.* + ex.* 등은 호출자가 final_query 로 지정)

    Returns:
        WITH ... SELECT ... FROM base_table e LEFT JOIN ... 형태의 전체 SQL
    """
    ctes = [{"name": extract_cte_name, "query": extract_cte_query}]
    join_clause = build_join_clause(
        left_table=base_table,
        right_table=extract_cte_name,
        on_columns=join_on_column,
        join_type="LEFT",
        left_alias=base_alias,
        right_alias=extract_alias,
    )
    from_clause = f"{base_table} {base_alias}\n{join_clause}"
    if select_columns is not None:
        return build_cte_sql(
            ctes=ctes,
            from_clause=from_clause,
            select_cols=select_columns,
        )
    final_query = (
        f"SELECT {base_alias}.*, {extract_alias}.*\n"
        f"FROM\n\t{base_table} {base_alias}\n\t{join_clause}"
    )
    return build_cte_sql(ctes=ctes, final_query=final_query)


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

    # 테스트 3: 범용 JOIN 절
    join_line = build_join_clause(
        left_table="EVENT_STAGE_12",
        right_table="EXTRACTED",
        on_columns="MDR_TEXT",
        left_alias="e",
        right_alias="ex",
    )
    print("\n=== build_join_clause (LEFT JOIN ON MDR_TEXT) ===")
    print(join_line)

    # 테스트 4: MDR 추출 CTE + EVENT_STAGE_12 LEFT JOIN
    extract_sql = "SELECT MDR_TEXT, patient_harm, defect_type FROM some_extract_table"
    full_sql = build_extract_join_sql(
        extract_cte_name="extracted",
        extract_cte_query=extract_sql,
        base_table="EVENT_STAGE_12",
        join_on_column="MDR_TEXT",
    )
    print("\n=== build_extract_join_sql ===")
    print(full_sql)

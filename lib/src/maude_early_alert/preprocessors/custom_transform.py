"""파이프라인 특화 SQL 생성 (일회용 변환 함수)"""

from textwrap import dedent, indent

from maude_early_alert.utils.helpers import validate_identifier
from maude_early_alert.utils.sql_builder import build_cte_sql

_INDENT = '    '  # 4-space indent


def build_combine_mdr_text_sql(source: str, text_col: str, type_col: str, combine_col: str) -> str:
    """mdr_text_text_type_codes와 mdr_text_texts 배열을 결합하여 mdr_text 컬럼 생성

    결과 형식:
        [code1]
        text1

        [code2]
        text2

    Args:
        source: 소스 테이블명

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)

    mdr_text_expr = dedent(f"""\
        ARRAY_TO_STRING(
            TRANSFORM(
                ARRAY_GENERATE_RANGE(0, ARRAY_SIZE({text_col})),
                i INT -> CONCAT('[', GET({type_col}, i)::STRING, ']',
                    CHAR(10), GET({text_col}, i)::STRING)
            ),
            CONCAT(CHAR(10), CHAR(10))
        ) AS {combine_col}""").strip()

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        select_cols=["*", mdr_text_expr],
    )


def build_primary_udi_di_sql(
        source: str, 
        partition_key: str,
        id_col: str, type_col: str, primary_col: str
    ) -> str:
    """identifiers에서 Primary type의 id를 전체 행에 전파

    같은 레코드(partition_key 기준) 내 identifiers 중
    type='Primary'인 행의 id를 모든 행에 primary_udi_di로 할당한다.

    Args:
        source: 소스 테이블명
        partition_key: PARTITION BY에 사용할 컬럼명 (원본 레코드 식별 키)

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)
    partition_key = validate_identifier(partition_key)

    window_expr = dedent(f"""\
        MAX(CASE WHEN {type_col} = 'Primary'
                 THEN {id_col} END)
        OVER (PARTITION BY {partition_key}) AS {primary_col}""").strip()

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        select_cols=["*", window_expr],
    )


def build_apply_company_alias_sql(source: str, company_col: str, aliases: dict) -> str:
    """M&A 기반 회사명 통합 SQL 생성 (CASE WHEN + LIKE prefix 매칭)

    원본 컬럼을 REPLACE하여 모회사명으로 치환한다.
    aliases는 config/preprocess/transform.yaml의 M&A.aliases에서 로드한다.

    Args:
        source: 소스 테이블명
        company_col: 제조사명 컬럼명
        aliases: {피인수사 prefix: 모회사명} 매핑 dict

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)
    company_col = validate_identifier(company_col)

    # 긴 키부터 매칭 (IRVINE BIOMEDICAL > IRVINE, CR BARD > CR 방지)
    sorted_aliases = sorted(aliases.items(), key=lambda x: -len(x[0]))
    when_lines = "\n".join(
        f"WHEN UPPER(TRIM({company_col})) LIKE '{alias}%' THEN '{parent}'"
        for alias, parent in sorted_aliases
    )
    case_expr = (
        "CASE\n"
        + indent(when_lines + f"\nELSE {company_col}", _INDENT)
        + f"\nEND AS {company_col}"
    )

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        replace_cols=[case_expr],
    )


def build_extract_udi_di_sql(source: str, public_col: str, target_col: str) -> str:
    """UDI_PUBLIC에서 UDI_DI 추출 SQL 생성 (CASE WHEN + REGEXP)

    3가지 패턴에 따라 DI를 추출한다:
    1. +로 시작 → + 이후 다음 특수문자 전까지의 영숫자
    2. (01)로 시작 → (01) 이후 14자리 숫자
    3. 영문자로 시작 → 원본 값 그대로

    Args:
        source: 소스 테이블명
        public_col: UDI PUBLIC 컬럼명 (config: extract_udi_di.columns.public_col)
        target_col: 추출 결과 컬럼명 (config: extract_udi_di.columns.target_col)

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)
    public_col = validate_identifier(public_col)
    target_col = validate_identifier(target_col)

    when_lines = "\n".join([
        f"WHEN REGEXP_LIKE({public_col}, '^\\+[A-Za-z0-9]{{7,}}.*') THEN REGEXP_SUBSTR({public_col}, '^\\+([A-Za-z0-9]+)', 1, 1, 'e')",
        f"WHEN REGEXP_LIKE({public_col}, '^\\(01\\)[0-9]{{14,}}.*') THEN REGEXP_SUBSTR({public_col}, '^\\(01\\)([0-9]{{14}})', 1, 1, 'e')",
        f"WHEN REGEXP_LIKE({public_col}, '^[a-zA-Z].*') THEN {public_col}",
        "ELSE NULL",
    ])
    case_expr = f"CASE\n{indent(when_lines, _INDENT)}\nEND"
    coalesce_expr = f"COALESCE(\n{indent(f'{target_col},', _INDENT)}\n{indent(case_expr, _INDENT)}\n) AS {target_col}"

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        replace_cols=[coalesce_expr],
    )


def build_manufacturer_fuzzy_match_sql(
    target: str,
    source: str,
    mfr_col: str,
    udf_schema: str,
    threshold: float,
) -> str:
    """제조사명 퍼지 매칭 후 MAUDE 테이블의 제조사명을 UDI 매칭 결과로 치환

    DISTINCT 제조사명끼리 CROSS JOIN → fuzzy_match_score UDF로 유사도 계산 →
    threshold 이상인 best match를 LEFT JOIN으로 원본에 적용.

    Args:
        target: 치환 대상
        source: 참조 기순
        mfr_col: 제조사명 컬럼 (config: fuzzy_match.mfr_col)
        udf_schema: UDF 스키마명
        threshold: 매칭 임계값 (config: fuzzy_match.threshold)

    Returns:
        완성된 SQL 문자열
    """
    target = validate_identifier(target)
    source = validate_identifier(source)
    mfr_col = validate_identifier(mfr_col)

    ctes = [
        {
            'name': 'target_mfrs',
            'query': f"""\
                SELECT DISTINCT {mfr_col} AS manufacturer
                FROM {target}
                WHERE {mfr_col} IS NOT NULL""",
        },
        {
            'name': 'source_mfrs',
            'query': f"""\
                SELECT DISTINCT {mfr_col} AS manufacturer
                FROM {source}
                WHERE {mfr_col} IS NOT NULL""",
        },
        {
            'name': 'cross_matched',
            'query': f"""\
                SELECT
                    m.manufacturer AS target_mfr,
                    u.manufacturer AS source_mfr,
                    {udf_schema}.fuzzy_match_score(m.manufacturer, u.manufacturer) AS score
                FROM target_mfrs m
                CROSS JOIN source_mfrs u""",
        },
        {
            'name': 'best_match',
            'query': f"""\
                SELECT target_mfr, source_mfr, score
                FROM cross_matched
                WHERE score >= {threshold}
                QUALIFY ROW_NUMBER() OVER (PARTITION BY target_mfr ORDER BY score DESC) = 1""",
        },
    ]

    return build_cte_sql(
        ctes=ctes,
        from_clause=f"{target} t",
        table_alias='t',
        replace_cols=[f"COALESCE(b.source_mfr, t.{mfr_col}) AS {mfr_col}"],
        joins=[f"LEFT JOIN best_match b ON t.{mfr_col} = b.target_mfr"],
    )


if __name__ == '__main__':
    pass

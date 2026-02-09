"""파이프라인 특화 SQL 생성 (일회용 변환 함수)"""

from maude_early_alert.utils.helpers import validate_identifier
from maude_early_alert.utils.sql_builder import build_cte_sql


def build_combine_mdr_text_sql(source: str) -> str:
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

    mdr_text_expr = (
        "ARRAY_TO_STRING(\n"
        "        TRANSFORM(\n"
        "            ARRAY_GENERATE_RANGE(0, ARRAY_SIZE(MDR_TEXT_TEXTS)),\n"
        "            i INT -> CONCAT('[', GET(MDR_TEXT_TEXT_TYPE_CODES, i)::STRING, ']',\n"
        "                CHAR(10), GET(MDR_TEXT_TEXTS, i)::STRING)\n"
        "        ),\n"
        "        CONCAT(CHAR(10), CHAR(10))\n"
        "    ) AS mdr_text"
    )

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        select_cols=["*", mdr_text_expr],
    ) + "\n;"


def build_primary_udi_di_sql(source: str, partition_key: str) -> str:
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

    window_expr = (
        f"MAX(CASE WHEN identifiers_type = 'Primary'\n"
        f"                 THEN identifiers_id END)\n"
        f"        OVER (PARTITION BY {partition_key}) AS primary_udi_di"
    )

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        select_cols=["*", window_expr],
    ) + "\n;"


# 인수합병(M&A) 기반 회사명 통합 매핑
# key: 피인수/자회사 prefix, value: 모회사명
COMPANY_ALIASES = {
    # Abbott 인수 계열
    "IRVINE BIOMEDICAL": "ABBOTT",
    "ST JUDE": "ABBOTT",
    "THORATEC": "ABBOTT",
    "PACESETTER": "ABBOTT",
    # Medtronic 인수 계열
    "COVIDIEN": "MEDTRONIC",
    "HEARTWARE": "MEDTRONIC",
    "NELLCOR": "MEDTRONIC",
    "EV3": "MEDTRONIC",
    # Becton Dickinson 인수 계열
    "CR BARD": "BECTON DICKINSON",
    "C R BARD": "BECTON DICKINSON",
    "DAVOL": "BECTON DICKINSON",
    "BARD": "BECTON DICKINSON",
    # Johnson & Johnson 계열
    "ETHICON": "JOHNSON & JOHNSON",
    "DEPUY": "JOHNSON & JOHNSON",
    "CODMAN": "JOHNSON & JOHNSON",
    "SYNTHES": "JOHNSON & JOHNSON",
    # Boston Scientific 인수 계열
    "GUIDANT": "BOSTON SCIENTIFIC",
    # LivaNova (Cyberonics 리브랜딩)
    "CYBERONICS": "LIVANOVA",
}

# 긴 키부터 매칭하도록 정렬 (IRVINE BIOMEDICAL > IRVINE, CR BARD > CR 방지)
_SORTED_ALIASES = sorted(COMPANY_ALIASES.items(), key=lambda x: -len(x[0]))


def build_apply_company_alias_sql(source: str, column: str) -> str:
    """M&A 기반 회사명 통합 SQL 생성 (CASE WHEN + LIKE prefix 매칭)

    원본 컬럼을 REPLACE하여 모회사명으로 치환한다.

    Args:
        source: 소스 테이블명
        column: 제조사명 컬럼명

    Returns:
        완성된 SQL 문자열
    """
    source = validate_identifier(source)
    column = validate_identifier(column)

    when_clauses = "\n            ".join(
        f"WHEN UPPER(TRIM({column})) LIKE '{alias}%' THEN '{parent}'"
        for alias, parent in _SORTED_ALIASES
    )
    case_expr = (
        f"CASE\n"
        f"            {when_clauses}\n"
        f"            ELSE {column}\n"
        f"        END AS {column}"
    )

    return build_cte_sql(
        ctes=[],
        from_clause=source,
        replace_cols=[case_expr],
    ) + "\n;"


def build_manufacturer_fuzzy_match_sql(
    maude_table: str,
    udi_table: str,
    maude_col: str,
    udi_col: str,
    udf_schema: str,
    threshold: float = 0.85,
) -> str:
    """제조사명 퍼지 매칭 후 MAUDE 테이블의 제조사명을 UDI 매칭 결과로 치환

    DISTINCT 제조사명끼리 CROSS JOIN → fuzzy_match_score UDF로 유사도 계산 →
    threshold 이상인 best match를 LEFT JOIN으로 원본에 적용.

    Args:
        maude_table: MAUDE 테이블명
        udi_table: UDI 테이블명
        maude_col: MAUDE 제조사명 컬럼
        udi_col: UDI 제조사명 컬럼
        udf_schema: UDF 스키마명
        threshold: 매칭 임계값 (0.0 ~ 1.0)

    Returns:
        완성된 SQL 문자열
    """
    maude_table = validate_identifier(maude_table)
    udi_table = validate_identifier(udi_table)
    maude_col = validate_identifier(maude_col)
    udi_col = validate_identifier(udi_col)

    ctes = [
        {
            'name': 'maude_mfrs',
            'query': (
                f"SELECT DISTINCT {maude_col} AS manufacturer\n"
                f"FROM {maude_table}\n"
                f"WHERE {maude_col} IS NOT NULL"
            ),
        },
        {
            'name': 'udi_mfrs',
            'query': (
                f"SELECT DISTINCT {udi_col} AS manufacturer\n"
                f"FROM {udi_table}\n"
                f"WHERE {udi_col} IS NOT NULL"
            ),
        },
        {
            'name': 'cross_matched',
            'query': (
                f"SELECT\n"
                f"    m.manufacturer AS maude_mfr,\n"
                f"    u.manufacturer AS udi_mfr,\n"
                f"    {udf_schema}.fuzzy_match_score(m.manufacturer, u.manufacturer) AS score\n"
                f"FROM maude_mfrs m\n"
                f"CROSS JOIN udi_mfrs u"
            ),
        },
        {
            'name': 'best_match',
            'query': (
                f"SELECT maude_mfr, udi_mfr, score\n"
                f"FROM cross_matched\n"
                f"WHERE score >= {threshold}\n"
                f"QUALIFY ROW_NUMBER() OVER (PARTITION BY maude_mfr ORDER BY score DESC) = 1"
            ),
        },
    ]

    return build_cte_sql(
        ctes=ctes,
        from_clause=f"{maude_table} t",
        table_alias='t',
        replace_cols=[f"COALESCE(b.udi_mfr, t.{maude_col}) AS {maude_col}"],
        joins=[f"LEFT JOIN best_match b ON t.{maude_col} = b.maude_mfr"],
    ) + "\n;"


if __name__ == '__main__':
    print("-- mdr_text combine")
    print(build_combine_mdr_text_sql("EVENT"))
    print()
    print("-- primary_udi_di")
    print(build_primary_udi_di_sql("UDI_STAGE_02", "public_device_record_key"))
    print()
    print("-- company alias")
    print(build_apply_company_alias_sql("EVENT_STAGE_06", "MANUFACTURER_NAME"))
    print()
    print("-- manufacturer fuzzy match")
    print(build_manufacturer_fuzzy_match_sql(
        "EVENT_STAGE_06", "UDI_STAGE_05",
        "MANUFACTURER_NAME", "MANUFACTURER_NAME",
        "UDF", threshold=0.85
    ))

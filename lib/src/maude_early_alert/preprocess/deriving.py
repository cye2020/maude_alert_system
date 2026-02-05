"""파생 컬럼 SQL 생성"""

import textwrap
from maude_early_alert.utils.helpers import validate_identifier


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

    sql = textwrap.dedent(f"""\
        SELECT
            *,
            ARRAY_TO_STRING(
                TRANSFORM(
                    ARRAY_GENERATE_RANGE(0, ARRAY_SIZE(MDR_TEXT_TEXTS)),
                    i INT -> CONCAT('[', GET(MDR_TEXT_TEXT_TYPE_CODES, i)::STRING, ']', CHAR(10), GET(MDR_TEXT_TEXTS, i)::STRING)
                ),
                CONCAT(CHAR(10), CHAR(10))
            ) AS mdr_text
        FROM {source}
        ;
    """)
    return sql


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

    sql = textwrap.dedent(f"""\
        SELECT
            *,
            MAX(CASE WHEN identifiers_type = 'Primary'
                     THEN identifiers_id END)
                OVER (PARTITION BY {partition_key}) AS primary_udi_di
        FROM {source}
        ;
    """)
    return sql


if __name__ == '__main__':
    print("-- mdr_text combine")
    print(build_combine_mdr_text_sql("EVENT"))
    print()
    print("-- primary_udi_di")
    print(build_primary_udi_di_sql("UDI_STAGE_02", "public_device_record_key"))

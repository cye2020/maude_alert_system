"""mdr_text 결합 SQL 생성"""

import textwrap
from maude_early_alert.utils.helpers import validate_identifier

def generate_combine_mdr_text_sql(source: str) -> str:
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


if __name__ == '__main__':
    sql = generate_combine_mdr_text_sql("EVENT")
    print(sql)
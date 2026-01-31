"""Snowflake Python UDF 등록 SQL 생성

cleaning_sql.py 에서 사용하는 Python UDF 들의 CREATE FUNCTION SQL 을 생성한다.

- clean_text_udf          : cleantext 라이브러리 기반 텍스트 정제
- remove_country_names_udf : NER 기반 국가명 제거

Usage:
    from maude_early_alert.preprocess.udf_registry import register_udfs

    cursor = conn.cursor()
    register_udfs(cursor)
"""

from typing import List


def generate_clean_text_udf_sql() -> str:
    """cleantext 라이브러리 기반 UDF 등록 SQL

    Snowflake Anaconda 채널에 'clean-text' 패키지가 있어야 한다.
    없으면 generate_clean_text_udf_with_stage_sql() 로 대체.
    """
    sql =  """\
    CREATE OR REPLACE FUNCTION clean_text_udf(input_text VARCHAR)
    RETURNS VARCHAR
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = ('clean-text')
    HANDLER = 'clean_text'
    AS
    $$
    import cleantext as cl

    def clean_text(input_text):
        if input_text is None or str(input_text).strip() == '':
            return None
        result = cl.clean(input_text)
        return result if result else None
    $$;
    """
    return sql


def generate_country_removal_udf_sql() -> str:
    """NER 기반 국가명 제거 UDF 등록 SQL

    Snowflake Anaconda 채널에 'country-named-entity-recognition'
    패키지가 있어야 한다. 없으면 아래 대안 참고:

    대안 1) Stage import 방식 (generate_country_removal_udf_with_stage_sql)
    대안 2) 국가명 룩업 테이블 + REGEXP_REPLACE 로 근사 처리
    """
    sql =  """
    CREATE OR REPLACE FUNCTION remove_country_names_udf(input_text VARCHAR)
    RETURNS VARCHAR
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.10'
    PACKAGES = ('country-named-entity-recognition')
    HANDLER = 'remove_countries'
    AS
    $$
    import re
    from country_named_entity_recognition import find_countries

    def remove_countries(input_text):
        if not input_text or str(input_text).strip() == '':
            return None
        try:
            countries_found = find_countries(input_text)
            if not countries_found:
                return input_text
            match_country = countries_found[0][1][0]
            regex = re.compile(match_country, re.IGNORECASE)
            result = regex.sub('', input_text).strip()
            return result if result else None
        except Exception:
            return input_text
    $$;
    """
    return sql


def generate_country_removal_udf_with_stage_sql(
    stage: str = "@udf_stage",
    import_path: str = "country_named_entity_recognition.zip",
) -> str:
    """Stage import 방식 국가명 제거 UDF (패키지 미제공 시 대안)

    사전 준비:
        PUT file://country_named_entity_recognition.zip @udf_stage/
    """
    return f"""\
CREATE OR REPLACE FUNCTION remove_country_names_udf(input_text VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
IMPORTS = ('{stage}/{import_path}')
HANDLER = 'remove_countries'
AS
$$
import re
import sys
sys.path.insert(0, '/tmp')

from country_named_entity_recognition import find_countries

def remove_countries(input_text):
    if not input_text or str(input_text).strip() == '':
        return None
    try:
        countries_found = find_countries(input_text)
        if not countries_found:
            return input_text
        match_country = countries_found[0][1][0]
        regex = re.compile(match_country, re.IGNORECASE)
        result = regex.sub('', input_text).strip()
        return result if result else None
    except Exception:
        return input_text
$$;"""


def generate_all_udf_sql() -> List[str]:
    """모든 UDF 등록 SQL 리스트 반환"""
    return [
        generate_clean_text_udf_sql(),
        generate_country_removal_udf_sql(),
    ]


def register_udfs(cursor) -> None:
    """Snowflake 커서를 통해 모든 UDF 등록 실행

    Args:
        cursor: snowflake.connector.cursor 객체
    """
    for sql in generate_all_udf_sql():
        cursor.execute(sql)

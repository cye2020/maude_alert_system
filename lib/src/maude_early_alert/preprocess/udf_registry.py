"""Snowflake Python UDF 등록 SQL 생성

cleaning_sql.py 에서 사용하는 Python UDF 들의 CREATE FUNCTION SQL 을 생성한다.
Anaconda 채널 미지원 패키지는 Stage import 방식으로 처리한다.

- clean_text_udf          : cleantext 라이브러리 기반 텍스트 정제
- remove_country_names_udf : NER 기반 국가명 제거

Usage:
    from maude_early_alert.preprocess.udf_registry import (
        generate_setup_stage_sql,
        generate_all_udf_sql,
        register_udfs,
    )

    cursor = conn.cursor()

    # 1) 스테이지 생성 + 패키지 업로드
    for sql in generate_setup_stage_sql(['/path/to/cleantext.zip', ...]):
        cursor.execute(sql)

    # 2) UDF 등록
    register_udfs(cursor)
"""

from typing import List

from maude_early_alert.utils.helpers import validate_identifier

DEFAULT_STAGE = "udf_packages"


## ── Stage 유틸리티 ──────────────────────────────────────────────


def generate_create_stage_sql(stage_name: str = DEFAULT_STAGE) -> str:
    """UDF 패키지 업로드용 내부 스테이지 생성 SQL

    Args:
        stage_name: 스테이지 이름 (기본값: udf_packages)
    """
    validate_identifier(stage_name)
    return f"CREATE STAGE IF NOT EXISTS {stage_name};"


def generate_put_sql(
    local_path: str,
    stage_name: str = DEFAULT_STAGE,
) -> str:
    """로컬 패키지(.zip)를 스테이지에 업로드하는 PUT SQL

    Args:
        local_path: 로컬 zip 파일 경로 (예: '/tmp/cleantext.zip')
        stage_name: 대상 스테이지명
    """
    validate_identifier(stage_name)
    return f"PUT file://{local_path} @{stage_name}/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;"


def generate_setup_stage_sql(
    local_paths: List[str],
    stage_name: str = DEFAULT_STAGE,
) -> List[str]:
    """스테이지 생성 + 패키지 업로드 SQL 일괄 생성

    Args:
        local_paths: 업로드할 zip 파일 경로 리스트
        stage_name: 대상 스테이지명

    Returns:
        [CREATE STAGE ..., PUT ..., PUT ..., ...]
    """
    sqls = [generate_create_stage_sql(stage_name)]
    for path in local_paths:
        sqls.append(generate_put_sql(path, stage_name))
    return sqls


## ── UDF SQL 생성 ────────────────────────────────────────────────


def generate_clean_text_udf_sql(
    stage_name: str = DEFAULT_STAGE,
    import_path: str = "cleantext.zip",
) -> str:
    """cleantext 라이브러리 기반 UDF (Stage import 방식)

    사전 준비:
        generate_setup_stage_sql(['/path/to/cleantext.zip']) 실행
    """
    validate_identifier(stage_name)
    return f"""\
CREATE OR REPLACE FUNCTION clean_text_udf(input_text VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
IMPORTS = ('@{stage_name}/{import_path}')
HANDLER = 'clean_text'
AS
$$
import sys
sys.path.insert(0, '/tmp')

import cleantext as cl

def clean_text(input_text):
    if input_text is None or str(input_text).strip() == '':
        return None
    result = cl.clean(input_text)
    return result if result else None
$$;"""


def generate_country_removal_udf_sql(
    stage_name: str = DEFAULT_STAGE,
    import_path: str = "country_named_entity_recognition.zip",
) -> str:
    """NER 기반 국가명 제거 UDF (Stage import 방식)

    사전 준비:
        generate_setup_stage_sql(['/path/to/country_named_entity_recognition.zip']) 실행
    """
    validate_identifier(stage_name)
    return f"""\
CREATE OR REPLACE FUNCTION remove_country_names_udf(input_text VARCHAR)
RETURNS VARCHAR
LANGUAGE PYTHON
RUNTIME_VERSION = '3.10'
IMPORTS = ('@{stage_name}/{import_path}')
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


def generate_all_udf_sql(stage_name: str = DEFAULT_STAGE) -> List[str]:
    """모든 UDF 등록 SQL 리스트 반환"""
    return [
        generate_clean_text_udf_sql(stage_name),
        generate_country_removal_udf_sql(stage_name),
    ]


def register_udfs(cursor, stage_name: str = DEFAULT_STAGE) -> None:
    """Snowflake 커서를 통해 모든 UDF 등록 실행

    스테이지 생성 및 패키지 업로드(generate_setup_stage_sql)는
    이 함수 호출 전에 별도로 실행해야 한다.

    Args:
        cursor: snowflake.connector.cursor 객체
        stage_name: 패키지가 업로드된 스테이지명
    """
    for sql in generate_all_udf_sql(stage_name):
        cursor.execute(sql)


if __name__ == '__main__':
    print("-- Stage setup (example paths)")
    for sql in generate_setup_stage_sql([
        '/tmp/cleantext.zip',
        '/tmp/country_named_entity_recognition.zip',
    ]):
        print(sql)
    print()
    print("-- UDF definitions")
    for sql in generate_all_udf_sql():
        print(sql)
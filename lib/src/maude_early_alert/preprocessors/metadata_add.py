"""
Silver SCD2 메타데이터 추가 SQL 생성 모듈

- add_incremental_metadata:     증분 적재용 SCD2 메타데이터 추가 SELECT SQL 생성
                                (is_current=TRUE, effective_from=지정 컬럼, batch_id→source_batch_id)
- build_expire_old_records_sql: 동일 business_key 이전 레코드 만료 UPDATE SQL 생성

effective_from_col은 storage.yaml transform.tables.{CATEGORY}_CURRENT.effective_from_col 에서 읽어
SilverConfig.get_silver_effective_from_col(category)로 주입합니다.
"""
from textwrap import dedent, indent
from typing import List, Union

import pendulum
import structlog

from maude_early_alert.utils.helpers import ensure_list, format_sql_literal, validate_identifier
from maude_early_alert.utils.sql_builder import build_cte_sql

_INDENT = '    '


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Bronze 테이블에서 Silver로 계보 추적을 위해 추출할 메타데이터 컬럼 목록
BRONZE_META_COLS = ['batch_id', 'source_system']


def build_extract_bronze_metadata_sql(bronze_table: str) -> str:
    """Bronze 테이블에서 BRONZE_META_COLS 값을 추출하는 SQL 생성

    가장 최신 ingest_time 기준 1건을 반환합니다.

    Args:
        bronze_table: Bronze 테이블 전체 경로 (e.g. 'MAUDE.BRONZE.EVENT')

    Returns:
        BRONZE_META_COLS 순서대로 값을 반환하는 SELECT SQL
    """
    bronze_table = validate_identifier(bronze_table)
    cols = ', '.join(BRONZE_META_COLS)
    return f"SELECT {cols} FROM {bronze_table} ORDER BY ingest_time DESC LIMIT 1"


def add_incremental_metadata(
    table_name: str,
    ingest_time: pendulum.DateTime,
    effective_from_col: str,
    source_batch_id: str,
    source_system: str,
) -> str:
    """증분 적재용 Silver SCD2 메타데이터 컬럼을 추가한 SELECT SQL 생성

    신규 배치 레코드에 Silver SCD2 메타데이터를 추가합니다.
    - ingest_time    = logical_date 리터럴 (DAG 실행 시각, TIMESTAMP_TZ)
    - is_current     = TRUE (신규 적재는 항상 현재 유효)
    - effective_from = effective_from_col::DATE (storage.yaml에서 주입)
    - effective_to   = NULL (적재 후 build_expire_old_records_sql로 이전 레코드를 갱신)
    - source_batch_id = Bronze batch_id 값 리터럴 (계보 추적용)
    - source_system   = Bronze source_system 값 리터럴

    Args:
        table_name:       현재 stage 테이블명
        ingest_time:      DAG logical_date (TIMESTAMP_TZ로 캐스팅)
        effective_from_col: effective_from 소스 컬럼명
                            (event: 'DATE_CHANGED', udi: 'PUBLIC_VERSION_DATE')
        source_batch_id:  Bronze 테이블에서 추출한 batch_id 값
        source_system:    Bronze 테이블에서 추출한 source_system 값

    Returns:
        Silver 메타데이터 컬럼이 추가된 SELECT SQL 문자열

    Examples:
        >>> sql = add_incremental_metadata(
        ...     table_name='EVENT_STAGE_14',
        ...     ingest_time=pendulum.datetime(2024, 11, 10, 3, 21, 10),
        ...     effective_from_col='DATE_CHANGED',
        ...     source_batch_id='bronze_20241110_032110',
        ...     source_system='s3',
        ... )
    """
    table_name = validate_identifier(table_name)
    validate_identifier(effective_from_col)

    silver_meta_exprs = [
        format_sql_literal('ingest_time', ingest_time),
        'TRUE AS is_current',
        f'{effective_from_col}::DATE AS effective_from',
        'NULL AS effective_to',
        format_sql_literal('source_batch_id', source_batch_id),
        format_sql_literal('source_system', source_system),
    ]

    select_block = indent(',\n'.join(silver_meta_exprs), _INDENT)
    final_query = (
        f"SELECT\n"
        f"{select_block},\n"
        f"{_INDENT}* EXCLUDE ({effective_from_col})\n"
        f"FROM\n"
        f"{_INDENT}{table_name}"
    )

    logger.debug(
        '증분 SCD2 메타데이터 추가 SQL 생성',
        table=table_name, effective_from_col=effective_from_col,
        source_batch_id=source_batch_id, source_system=source_system,
    )

    return build_cte_sql(ctes=[], final_query=final_query)


def build_expire_old_records_sql(
    target: str,
    business_key: Union[str, List[str]],
) -> str:
    """동일 business_key의 이전 레코드를 만료 처리하는 UPDATE SQL 생성

    MERGE 적재 이후 실행합니다.
    target 테이블 내에서 business_key 기준으로 그룹을 나누고,
    각 그룹의 is_current=TRUE 레코드 중 가장 최신(MAX effective_from) 이외의 것을 만료합니다:
        - is_current  → FALSE
        - effective_to → 그룹 내 최신 effective_from - 1일

    Args:
        target:       대상 테이블명 (e.g. 'MAUDE.SILVER.EVENT_CURRENT')
        business_key: 비즈니스 기준 키 (단일 str 또는 List[str])

    Returns:
        이전 레코드를 만료 처리하는 UPDATE SQL 문자열

    Examples:
        >>> sql = build_expire_old_records_sql(
        ...     target='MAUDE.SILVER.EVENT_CURRENT',
        ...     business_key=['MDR_REPORT_KEY', 'UDI_DI'],
        ... )
    """
    target = validate_identifier(target)
    keys = ensure_list(business_key)
    for k in keys:
        validate_identifier(k)

    group_by_cols = ', '.join(keys)
    where_items = (
        [f't.{k} = latest.{k}' for k in keys]
        + ['t.effective_from < latest.effective_from', 't.is_current = TRUE']
    )

    sql = '\n'.join([
        dedent(f"""\
            UPDATE {target} AS t
            SET
                t.is_current = FALSE,
                t.effective_to = DATEADD('day', -1, latest.effective_from)
            FROM (
                SELECT {group_by_cols}, MAX(effective_from) AS effective_from
                FROM {target}
                WHERE is_current = TRUE
                GROUP BY {group_by_cols}
            ) AS latest"""),
        'WHERE\n' + indent('\nAND '.join(where_items), _INDENT),
    ])

    logger.debug('이전 레코드 만료 처리 SQL 생성', target=target, business_key=keys)

    return sql


if __name__ == '__main__':
    import pendulum as _pendulum

    print("=== add_incremental_metadata (event) ===")
    print(add_incremental_metadata(
        table_name='FOO_STAGE',
        ingest_time=_pendulum.datetime(2024, 1, 1),
        effective_from_col='DATE_CHANGED',
        source_batch_id='batch_xxx',
        source_system='example',
    ))

    print("\n=== add_incremental_metadata (udi) ===")
    print(add_incremental_metadata(
        table_name='BAR_STAGE',
        ingest_time=_pendulum.datetime(2024, 1, 1),
        effective_from_col='PUBLIC_VERSION_DATE',
        source_batch_id='batch_xxx',
        source_system='example',
    ))

    print("\n=== build_expire_old_records_sql ===")
    print(build_expire_old_records_sql(
        target='MY_DB.MY_SCHEMA.FOO_CURRENT',
        business_key=['KEY_A', 'KEY_B'],
    ))

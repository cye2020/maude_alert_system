"""
메타데이터 추출 및 Silver SCD2 메타데이터 추가 SQL 생성 모듈

- extract_metadata: Bronze 테이블에서 메타데이터 컬럼을 SELECT하는 SQL 생성
- add_metadata: 현재 stage 테이블에 Silver SCD2 메타데이터 컬럼을 추가한 SELECT SQL 생성
"""
from typing import List, Union

import pendulum
import structlog

from maude_early_alert.utils.helpers import ensure_list, format_sql_literal, validate_identifier
from maude_early_alert.utils.sql_builder import build_cte_sql


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Bronze 테이블의 메타데이터 컬럼 목록
BRONZE_META_COLS = ['ingest_time', 'batch_id', 'source_system', 'source_file', 'record_hash']


def extract_metadata(table_name: str) -> str:
    """Bronze 테이블에서 메타데이터 컬럼을 SELECT하는 SQL 생성

    Bronze 메타데이터 컬럼(ingest_time, batch_id, source_system, source_file,
    record_hash)을 선택합니다. Silver 파이프라인에서 Bronze 적재 정보를 참조할 때 사용합니다.

    Args:
        table_name: Bronze 테이블명 (예: 'MAUDE.BRONZE.EVENT')

    Returns:
        Bronze 메타데이터 컬럼만 SELECT하는 SQL 문자열

    Examples:
        >>> sql = extract_metadata('MAUDE.BRONZE.EVENT')
    """
    table_name = validate_identifier(table_name)
    logger.debug('Bronze 메타데이터 추출 SQL 생성', table=table_name)
    return build_cte_sql(
        ctes=[],
        from_clause=table_name,
        select_cols=BRONZE_META_COLS,
    )


def add_metadata(
    table_name: str,
    ingest_time: pendulum.DateTime,
    is_current: bool,
    effective_from: pendulum.Date,
    source_batch_id: str,
    source_system: str,
) -> str:
    """Silver SCD2 메타데이터 컬럼을 추가한 SELECT SQL 생성 (리터럴 주입, 단일 배치용)

    현재 stage 테이블에서 SELECT하면서 Silver SCD2 메타데이터 컬럼을 앞에 추가합니다.
    Bronze 메타데이터 컬럼은 Silver 메타데이터로 대체됩니다.
    business_key는 원본 데이터 컬럼으로 유지됩니다 (별도 컬럼을 생성하지 않음).

    추가되는 Silver 메타데이터 컬럼:
        - ingest_time:     Bronze 버전 선택 시각 (TIMESTAMP_TZ)
        - is_current:      현재 유효 여부 (SCD2)
        - effective_from:  유효 시작 시점 (DATE)
        - effective_to:    유효 종료 시점 (NULL — 신규 적재 시 항상 NULL)
        - source_batch_id: Bronze batch_id (계보 추적용)
        - source_system:   원천 시스템명

    Args:
        table_name:      현재 stage 테이블명
        ingest_time:     Bronze 버전 선택 시각 (TIMESTAMP_TZ로 캐스팅)
        is_current:      현재 유효 여부
        effective_from:  유효 시작 시점 (DATE로 캐스팅)
        source_batch_id: Bronze batch_id (lineage 추적용)
        source_system:   원천 시스템명 (예: 'openFDA')

    Returns:
        Silver 메타데이터 컬럼이 추가된 SELECT SQL 문자열

    Examples:
        >>> import pendulum
        >>> sql = add_metadata(
        ...     table_name='EVENT_STAGE_15',
        ...     ingest_time=pendulum.datetime(2024, 11, 10),
        ...     is_current=True,
        ...     effective_from=pendulum.date(2024, 11, 10),
        ...     source_batch_id='B202411',
        ...     source_system='openFDA',
        ... )
    """
    table_name = validate_identifier(table_name)

    # effective_from: pendulum.Date → DATE 리터럴
    date_str = effective_from.strftime('%Y-%m-%d')
    effective_from_expr = f"'{date_str}'::DATE AS effective_from"

    # Bronze 메타데이터 컬럼만 SELECT *에서 제외 (business_key 원본 컬럼은 유지)
    exclude_cols = sorted(set(BRONZE_META_COLS))
    exclude_clause = ', '.join(exclude_cols)

    silver_meta_exprs = [
        format_sql_literal('ingest_time', ingest_time),
        format_sql_literal('is_current', is_current),
        effective_from_expr,
        'NULL AS effective_to',
        format_sql_literal('source_batch_id', source_batch_id),
        format_sql_literal('source_system', source_system),
    ]

    select_lines = ',\n    '.join(silver_meta_exprs)
    final_query = (
        f"SELECT\n"
        f"    {select_lines},\n"
        f"    * EXCLUDE ({exclude_clause})\n"
        f"FROM\n"
        f"    {table_name}"
    )

    logger.debug(
        'Silver 메타데이터 추가 SQL 생성',
        table=table_name, is_current=is_current,
        effective_from=date_str, source_system=source_system,
    )

    return build_cte_sql(ctes=[], final_query=final_query)


def build_scd2_sql(
    table_name: str,
    partition_by: Union[str, List[str]],
    ingest_time_col: str = 'ingest_time',
    source_batch_id_col: str = 'batch_id',
    source_system_col: str = 'source_system',
) -> str:
    """Bronze 전체 이력에서 SCD2 메타데이터를 윈도우 함수로 계산하는 SELECT SQL 생성

    DEDUP 이전 단계(전체 이력 보존 상태)에 적용합니다.
    business_key 컬럼을 새로 생성하지 않고 원본 컬럼을 유지합니다.
    partition_by 컬럼은 윈도우 함수의 PARTITION BY에만 사용됩니다.

    동일 partition_by의 모든 버전에 대해 ingest_time 순서 기준으로:
        - is_current:     가장 최신 버전이면 TRUE
        - effective_from: 해당 버전의 ingest_time::DATE
        - effective_to:   다음 버전의 ingest_time::DATE - 1일 (최신 버전은 NULL)

    add_metadata와의 차이:
        - add_metadata:   Python 값을 리터럴로 주입 (단일 배치 증분 적재용)
        - build_scd2_sql: 전체 이력에서 윈도우 함수로 계산 (Full Refresh용)

    Args:
        table_name:          전체 이력 테이블명 (DEDUP 이전 stage 또는 Bronze)
        partition_by:        SCD2 파티션 기준 컬럼명. 단일 str 또는 List[str]
        ingest_time_col:     ingest 시각 컬럼명
        source_batch_id_col: batch_id 컬럼명
        source_system_col:   source_system 컬럼명

    Returns:
        SCD2 메타데이터가 계산된 SELECT SQL 문자열

    Examples:
        >>> sql = build_scd2_sql('EVENT_STAGE_01', partition_by=['MDR_REPORT_KEY', 'UDI_DI'])
    """
    table_name = validate_identifier(table_name)
    keys = ensure_list(partition_by)
    for k in keys:
        validate_identifier(k)

    partition_cols = ', '.join(keys)

    # SCD2 윈도우 표현식 (business_key 컬럼은 생성하지 않음 — 원본 컬럼 유지)
    over_asc  = f"OVER (PARTITION BY {partition_cols} ORDER BY {ingest_time_col})"
    over_desc = f"OVER (PARTITION BY {partition_cols} ORDER BY {ingest_time_col} DESC)"

    is_current_expr     = f"(ROW_NUMBER() {over_desc} = 1) AS is_current"
    effective_from_expr = f"{ingest_time_col}::DATE AS effective_from"
    effective_to_expr   = (
        f"DATEADD('day', -1, LEAD({ingest_time_col}::DATE) {over_asc}) AS effective_to"
    )

    silver_meta_exprs = [
        f"{ingest_time_col} AS ingest_time",
        is_current_expr,
        effective_from_expr,
        effective_to_expr,
        f"{source_batch_id_col} AS source_batch_id",
        f"{source_system_col} AS source_system",
    ]

    # SELECT *에서 제외: Bronze 메타 컬럼만 (business_key 원본 컬럼은 유지)
    exclude_cols = sorted(
        set(BRONZE_META_COLS)
        | {ingest_time_col, source_batch_id_col, source_system_col}
    )
    exclude_clause = ', '.join(exclude_cols)

    select_lines = ',\n    '.join(silver_meta_exprs)
    final_query = (
        f"SELECT\n"
        f"    {select_lines},\n"
        f"    * EXCLUDE ({exclude_clause})\n"
        f"FROM\n"
        f"    {table_name}"
    )

    logger.debug(
        'SCD2 메타데이터 계산 SQL 생성',
        table=table_name, partition_by=keys, partition_cols=partition_cols,
    )

    return build_cte_sql(ctes=[], final_query=final_query)


if __name__ == '__main__':
    print("=== extract_metadata ===")
    print(extract_metadata('MAUDE.BRONZE.EVENT'))

    print("\n=== add_metadata ===")
    print(add_metadata(
        table_name='EVENT_STAGE_15',
        ingest_time=pendulum.datetime(2024, 11, 10, 3, 21, 10),
        is_current=True,
        effective_from=pendulum.date(2024, 11, 10),
        source_batch_id='B202411',
        source_system='openFDA',
    ))

    print("\n=== build_scd2_sql (복합 partition_by) ===")
    print(build_scd2_sql(
        table_name='EVENT_STAGE_01',
        partition_by=['MDR_REPORT_KEY', 'UDI_DI'],
    ))

"""
컬럼 선택 SQL 생성 모듈
컬럼 리스트 → SELECT SQL 생성
"""
from typing import List

import structlog

from maude_early_alert.utils.sql_builder import build_cte_sql


logger = structlog.get_logger()


def build_select_columns_sql(cols: List[str], table_name: str) -> str:
    """
    SELECT SQL 생성

    Args:
        cols: 선택할 컬럼 리스트
        table_name: 테이블명

    Returns:
        생성된 SELECT SQL 문자열

    Examples:
        >>> sql = build_select_columns_sql(['col1', 'col2'], "MAUDE.SILVER.EVENT")
    """
    if not cols:
        logger.debug("선택할 컬럼이 없습니다. SELECT * 반환")
        return build_cte_sql(ctes=[], from_clause=table_name)

    select_cols = [
        f"{col['name']} AS {col['alias']}"
        for col in cols
    ]

    sql = build_cte_sql(
        ctes=[],
        from_clause=table_name,
        select_cols=select_cols,
    )

    logger.debug("SELECT SQL 생성 완료", table_name=table_name, column_count=len(cols))

    return sql


if __name__ == '__main__':
    from maude_early_alert.logging_config import configure_logging
    configure_logging()

    print("=" * 80)
    print("컬럼 필터링 SQL 테스트")
    print("=" * 80)

    maude_columns = [
        {'name': 'ADVERSE_EVENT_FLAG', 'alias': 'ADVERSE_EVENT_FLAG'},
        {'name': 'DATE_OF_EVENT', 'alias': 'DATE_OCCURRED'},
        {'name': 'DATE_RECEIVED', 'alias': 'DATE_RECEIVED'},
        {'name': 'DEVICE_DATE_OF_MANUFACTURER', 'alias': 'DATE_MANUFACTURED'},
        {'name': 'EVENT_TYPE', 'alias': 'EVENT_TYPE'},
        {'name': 'MDR_REPORT_KEY', 'alias': 'MDR_REPORT_KEY'},
        {'name': 'PREVIOUS_USE_CODE', 'alias': 'PREVIOUS_USE_FLAG'},
        {'name': 'PRODUCT_PROBLEM_FLAG', 'alias': 'PRODUCT_PROBLEM_FLAG'},
        {'name': 'PRODUCT_PROBLEMS', 'alias': 'PRODUCT_PROBLEMS'},
        {'name': 'REPORT_NUMBER', 'alias': 'REPORT_NUMBER'},
        {'name': 'REPROCESSED_AND_REUSED_FLAG', 'alias': 'REPROCESSED_AND_REUSED_FLAG'},
        {'name': 'SINGLE_USE_FLAG', 'alias': 'SINGLE_USE_FLAG'},
        {'name': 'MDR_TEXT', 'alias': 'MDR_TEXT'},
        {'name': 'DEVICE_BRAND_NAME', 'alias': 'BRAND_NAME'},
        {'name': 'DEVICE_CATALOG_NUMBER', 'alias': 'CATALOG_NUMBER'},
        {'name': 'DEVICE_DEVICE_REPORT_PRODUCT_CODE', 'alias': 'PRODUCT_CODE'},
        {'name': 'DEVICE_LOT_NUMBER', 'alias': 'LOT_NUMBER'},
        {'name': 'DEVICE_MANUFACTURER_D_NAME', 'alias': 'MANUFACTURER_NAME'},
        {'name': 'DEVICE_MANUFACTURER_D_POSTAL_CODE', 'alias': 'MANUFACTURER_POSTAL_CODE'},
        {'name': 'DEVICE_MODEL_NUMBER', 'alias': 'MODEL_NUMBER'},
        {'name': 'DEVICE_OPENFDA_DEVICE_NAME', 'alias': 'PRODUCT_NAME'},
        {'name': 'DEVICE_UDI_DI', 'alias': 'UDI_DI'},
        {'name': 'DEVICE_UDI_PUBLIC', 'alias': 'UDI_PUBLIC'},
    ]
    
    sql_maude = build_select_columns_sql(maude_columns, "MAUDE.SILVER.EVENT_STAGE_02")
    print("\n[MAUDE SQL]")
    print(sql_maude)

    udi_columns = [
        {'name': 'BRAND_NAME', 'alias': 'BRAND_NAME'},
        {'name': 'CATALOG_NUMBER', 'alias': 'CATALOG_NUMBER'},
        {'name': 'COMPANY_NAME', 'alias': 'MANUFACTURER_NAME'},
        {'name': 'IDENTIFIERS_ID', 'alias': 'UDI_DI'},
        {'name': 'IDENTIFIERS_TYPE', 'alias': 'ID_TYPE'},
        {'name': 'PUBLIC_VERSION_DATE', 'alias': 'PUBLIC_VERSION_DATE'},
        {'name': 'PUBLISH_DATE', 'alias': 'PUBLISH_DATE'},
        {'name': 'VERSION_OR_MODEL_NUMBER', 'alias': 'MODEL_NUMBER'}
    ]
    
    sql_udi = build_select_columns_sql(udi_columns, "MAUDE.SILVER.UDI_STAGE_02")
    print("\n[UDI SQL]")
    print(sql_udi)

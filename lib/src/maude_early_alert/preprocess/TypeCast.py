"""
타입 변환 SQL 생성 모듈
"""
from typing import List, Dict, Optional

import structlog


class TypeCastSQLBuilder:
    """타입 변환 SQL 생성"""
    
    # Boolean 매핑 패턴 (클래스 내부 상수)
    BOOLEAN_PATTERNS = {
        "PREVIOUS_USE": ("I", "R"),
        "DEVICE_OPERATOR": ("HEALTH PROFESSIONAL", "LAY USER/PATIENT"),
    }
    
    TYPE_NORMALIZATION = {
        "INT": "INTEGER",
        "BOOL": "BOOLEAN",
    }
    
    def __init__(self, columns: List[Dict]):
        """
        Args:
            columns: 컬럼 정보 리스트
                [{'name': 'col1', 'alias': 'col1', 'type': 'DATE'}, ...]
        """
        self.columns = columns
    
    def build_sql(
        self,
        input_table: str,
        output_table: Optional[str] = None,
        table_alias: str = "t"
    ) -> str:
        """타입 변환 SQL 생성
        
        Args:
            input_table: 입력 테이블명
            output_table: 출력 테이블명 (지정 시 CREATE TABLE 생성)
            table_alias: 테이블 alias
            
        Returns:
            생성된 SQL 문자열
        """
        logger = structlog.get_logger()
        
        if not self.columns:
            logger.warning("No columns to cast")
            return f"SELECT * FROM {input_table}"
        
        select_exprs = [
            self._build_cast_expr(col, table_alias)
            for col in self.columns
        ]
        # 한 줄에 컬럼 하나, SELECT 본문 4칸 들여쓰기로 통일
        select_clause = ",\n    ".join(select_exprs)
        sql = f"SELECT\n    {select_clause}\nFROM\n    {input_table} {table_alias}"
        if output_table:
            sql = f"CREATE OR REPLACE TABLE {output_table} AS \n{sql} \n ;"

        logger.debug("Type cast SQL generated", column_count=len(self.columns))
        return sql
    
    def _build_cast_expr(self, col: Dict, table_alias: str) -> str:
        """단일 컬럼 CAST 표현식 생성"""
        col_name = col.get("alias") or col["name"]
        raw_type = col.get("type", "VARCHAR").upper()
        col_type = self.TYPE_NORMALIZATION.get(raw_type, raw_type)
        q = f'"{col_name}"'
        
        if col_type == "DATE":
            return f"TRY_TO_DATE({table_alias}.{q}) AS {q}"
        
        if col_type == "INTEGER":
            return f"TRY_TO_NUMBER({table_alias}.{q})::INTEGER AS {q}"
        
        if col_type == "BOOLEAN":
            true_val, false_val = self._get_boolean_mapping(col_name, col)
            return (
                f"CASE WHEN {table_alias}.{q} = '{true_val}' THEN TRUE "
                f"WHEN {table_alias}.{q} = '{false_val}' THEN FALSE "
                f"ELSE NULL END AS {q}"
            )
        
        if col_type == "CATEGORICAL":
            return f"TRY_CAST({table_alias}.{q} AS VARCHAR(50)) AS {q}"
        
        if col_type == "ARRAY":
            return f"{table_alias}.{q} AS {q}"
        
        # VARCHAR 기본
        return f"{table_alias}.{q} AS {q}"
    
    def _get_boolean_mapping(self, col_name: str, col: Dict) -> tuple:
        """Boolean 매핑 결정"""
        # 명시적 매핑이 있으면 사용
        if col.get("true_value") and col.get("false_value"):
            return col["true_value"], col["false_value"]
        
        # 패턴 매칭
        col_upper = col_name.upper()
        for pattern, mapping in self.BOOLEAN_PATTERNS.items():
            if pattern in col_upper:
                logger = structlog.get_logger()
                logger.debug("Boolean pattern matched", column=col_name, pattern=pattern)
                return mapping
        
        # 기본 Y/N
        return "Y", "N"


if __name__ == "__main__":
    from maude_early_alert.logging_config import configure_logging
    
    configure_logging()
    
    print("=" * 80)
    print("타입 변환 SQL 테스트")
    print("=" * 80)
    
    # columns.yaml의 maude: cols 내용
    maude_columns = [
        {'name': 'ADVERSE_EVENT_FLAG', 'type': 'BOOLEAN', 'alias': 'ADVERSE_EVENT_FLAG', 'final': True},
        {'name': 'DATE_OF_EVENT', 'type': 'DATE', 'alias': 'DATE_OCCURRED', 'final': False},
        {'name': 'DATE_RECEIVED', 'type': 'DATE', 'alias': 'DATE_RECEIVED', 'final': True},
        {'name': 'DEVICE_DATE_OF_MANUFACTURER', 'type': 'DATE', 'alias': 'DATE_MANUFACTURED', 'final': False},
        {'name': 'EVENT_TYPE', 'type': 'VARCHAR', 'alias': 'EVENT_TYPE', 'final': True},
        {'name': 'MDR_REPORT_KEY', 'type': 'INT', 'alias': 'MDR_REPORT_KEY', 'final': True},
        {'name': 'PREVIOUS_USE_CODE', 'type': 'BOOLEAN', 'alias': 'PREVIOUS_USE_FLAG', 'final': True},
        {'name': 'PRODUCT_PROBLEM_FLAG', 'type': 'BOOLEAN', 'alias': 'PRODUCT_PROBLEM_FLAG', 'final': True},
        {'name': 'PRODUCT_PROBLEMS', 'type': 'ARRAY', 'alias': 'PRODUCT_PROBLEMS', 'final': True},
        {'name': 'REPORT_NUMBER', 'type': 'VARCHAR', 'alias': 'REPORT_NUMBER', 'final': False},
        {'name': 'REPROCESSED_AND_REUSED_FLAG', 'type': 'BOOLEAN', 'alias': 'REPROCESSED_AND_REUSED_FLAG', 'final': True},
        {'name': 'SINGLE_USE_FLAG', 'type': 'BOOLEAN', 'alias': 'SINGLE_USE_FLAG', 'final': True},
        {'name': 'MDR_TEXT', 'type': 'VARCHAR', 'alias': 'MDR_TEXT', 'final': True},
        {'name': 'DEVICE_BRAND_NAME', 'type': 'VARCHAR', 'alias': 'BRAND_NAME', 'final': True},
        {'name': 'DEVICE_CATALOG_NUMBER', 'type': 'VARCHAR', 'alias': 'CATALOG_NUMBER', 'final': False},
        {'name': 'DEVICE_DEVICE_REPORT_PRODUCT_CODE', 'type': 'VARCHAR', 'alias': 'PRODUCT_CODE', 'final': True},
        {'name': 'DEVICE_LOT_NUMBER', 'type': 'VARCHAR', 'alias': 'LOT_NUMBER', 'final': False},
        {'name': 'DEVICE_MANUFACTURER_D_NAME', 'type': 'VARCHAR', 'alias': 'MANUFACTURER_NAME', 'final': True},
        {'name': 'DEVICE_MANUFACTURER_D_POSTAL_CODE', 'type': 'VARCHAR', 'alias': 'MANUFACTURER_POSTAL_CODE', 'final': False},
        {'name': 'DEVICE_MODEL_NUMBER', 'type': 'VARCHAR', 'alias': 'MODEL_NUMBER', 'final': True},
        {'name': 'DEVICE_OPENFDA_DEVICE_NAME', 'type': 'VARCHAR', 'alias': 'PRODUCT_NAME', 'final': True},
        {'name': 'DEVICE_UDI_DI', 'type': 'VARCHAR', 'alias': 'UDI_DI', 'final': True},
        {'name': 'DEVICE_UDI_PUBLIC', 'type': 'VARCHAR', 'alias': 'UDI_PUBLIC', 'final': False},
    ]
    
    # TypeCastSQLBuilder 생성
    builder = TypeCastSQLBuilder(maude_columns)
    
    # SQL 생성
    sql = builder.build_sql(
        input_table="MAUDE.SILVER.EVENT_STAGE_05",
        output_table="MAUDE.SILVER.EVENT_STAGE_06"
    )
    
    print(sql)
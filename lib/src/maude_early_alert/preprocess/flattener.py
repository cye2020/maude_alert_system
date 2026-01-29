"""Snowflake MAUDE EVENT 평탄화 SQL 생성기

실제 사용 중인 SQL 패턴 기반:
- mdr_text: 필드별 ORDER BY 전략
- Device: LATERAL FLATTEN으로 행 분리
- Array fields: 타입 캐스트 없이 그대로
"""

from typing import Dict, List, Optional
import logging
from pathlib import Path

from maude_early_alert.utils.flattener_helper import MAUDE_EVENT_SCHEMA

from maude_early_alert.utils.helpers import validate_identifier

logger = logging.getLogger(__name__)


def maude_flatten_sql(
    table: str,
    schema: Dict[str, any],
    limit: Optional[int] = None,
) -> str:
    """
    MAUDE EVENT 테이블 완전 평탄화 SQL 생성
    
    Args:
        table: 원본 테이블명 (예: 'MAUDE.BRONZE.EVENT')
        schema: 스키마 정의 (flattener_helper.MAUDE_EVENT_SCHEMA)
        limit: LIMIT 절 (None이면 제거)
    
    Returns:
        완성된 SQL 쿼리 문자열
    
    Example:
        from utils.flattener_helper import MAUDE_EVENT_SCHEMA
        
        sql = generate_maude_flatten_sql(
            table='MAUDE.BRONZE.EVENT',
            schema=MAUDE_EVENT_SCHEMA,
            limit=150
        )
    """
    try:
        
        # 입력 검증
        validate_identifier(table)
        
        if not schema:
            raise ValueError("스키마가 비어있습니다.")
        
        # 스키마 추출
        scalar_fields = schema.get('scalar_fields', [])
        array_fields = schema.get('array_fields', [])
        first_only = schema.get('first_only', {})
        aggregated_array = schema.get('aggregated_array', {})
        row_split_array = schema.get('row_split_array', {})
        
        logger.info(f"SQL 생성 시작: {table}")
        logger.info(f"   Scalar: {len(scalar_fields)}개")
        logger.info(f"   Array: {len(array_fields)}개")
        logger.info(f"   Patient: {sum(len(v) for v in first_only.values())}개")
        logger.info(f"   mdr_text: {sum(len(v) for v in aggregated_array.values())}개")
        logger.info(f"   Device: {sum(len(v) for v in row_split_array.values())}개")
        
        lines = []
        
        # ============================================
        # 헤더 주석
        # ============================================
        lines.append("-- ============================================")
        lines.append("-- MAUDE EVENT 완전 평탄화 (Device 행 분리)")
        lines.append("-- - 최상위 스칼라: 전체")
        lines.append("-- - 최상위 배열: 그대로 유지")
        lines.append("-- - Patient: 첫 번째만")
        lines.append("-- - mdr_text: 배열로 집계 (중복 제거)")
        lines.append("-- - Device: 행으로 펼치기 (LATERAL FLATTEN)")
        lines.append("-- ============================================")
        lines.append("")
        
        # ============================================
        # CTE: mdr_text 집계
        # ============================================
        if aggregated_array:
            for parent, fields in aggregated_array.items():
                validate_identifier(parent)
                cte_name = f"{parent}_aggregated"
                
                lines.append(f"WITH {cte_name} AS (")
                lines.append("    SELECT")
                lines.append("        raw_data:report_number::STRING AS report_number,")
                
                for i, field in enumerate(fields):
                    # 컬럼명 생성
                    col_name_base = field.replace(':', '_')
                    if field == 'text':
                        col_name = f"{parent}_{col_name_base}s"
                    elif field == 'mdr_text_key':
                        col_name = f"{parent}_keys"
                    elif field == 'patient_sequence_number':
                        col_name = f"{parent}_{col_name_base}s"
                    elif field == 'text_type_code':
                        col_name = f"{parent}_type_codes"
                    else:
                        col_name = f"{parent}_{col_name_base}s"
                    
                    # ORDER BY 전략
                    # text는 index 순서, 나머지는 값 자체로 정렬
                    if field == 'text':
                        order_by = "ORDER BY m.index"
                    else:
                        order_by = f"ORDER BY m.value:{field}"
                    
                    comma = "" if i == len(fields) - 1 else ","
                    lines.append(f"        ARRAY_AGG( m.value:{field}) ")
                    lines.append(f"            WITHIN GROUP ({order_by}) AS {col_name}{comma}")
                
                lines.append(f"    FROM {table},")
                lines.append(f"        LATERAL FLATTEN(input => raw_data:{parent}) AS m")
                lines.append("    GROUP BY raw_data:report_number")
                lines.append(")")
                lines.append("")
        
        # ============================================
        # SELECT 절
        # ============================================
        lines.append("SELECT")
        columns = []
        
        # 1. Scalar fields
        if scalar_fields:
            columns.append("    -- ============================================")
            columns.append("    -- 최상위 스칼라 필드 (각 device 행에 복제)")
            columns.append("    -- ============================================")
            for field in scalar_fields:
                col_name = field.replace(':', '_')
                columns.append(f"    e.raw_data:{field}::STRING AS {col_name},")
            columns.append("")
        
        # 2. Array fields (타입 캐스트 없음!)
        if array_fields:
            columns.append("    -- ============================================")
            columns.append("    -- 최상위 배열 필드 (각 device 행에 복제)")
            columns.append("    -- ============================================")
            for field in array_fields:
                columns.append(f"    e.raw_data:{field} AS {field},")
            columns.append("")
        
        # 3. Patient (첫 번째만)
        if first_only:
            for parent, fields in first_only.items():
                columns.append("    -- ============================================")
                columns.append(f"    -- {parent.upper()} (첫 번째만, 각 device 행에 복제)")
                columns.append("    -- ============================================")
                for field in fields:
                    col_name = f"{parent}_{field.replace(':', '_')}"
                    
                    # patient_problems 등 배열은 타입 캐스트 없음
                    if field in ['patient_problems', 'sequence_number_outcome', 'sequence_number_treatment']:
                        columns.append(f"    e.raw_data:{parent}[0]:{field} AS {col_name},")
                    else:
                        columns.append(f"    e.raw_data:{parent}[0]:{field}::STRING AS {col_name},")
                columns.append("")
        
        # 4. mdr_text (집계)
        if aggregated_array:
            for parent, fields in aggregated_array.items():
                columns.append("    -- ============================================")
                columns.append(f"    -- {parent} (배열로 집계, 각 device 행에 복제)")
                columns.append("    -- ============================================")
                
                # 컬럼명 매핑
                field_map = {
                    'mdr_text_key': 'mdr_text_keys',
                    'patient_sequence_number': 'mdr_text_patient_sequence_numbers',
                    'text': 'mdr_text_texts',
                    'text_type_code': 'mdr_text_type_codes',
                }
                
                for field in fields:
                    col_name = field_map.get(field, f"{parent}_{field.replace(':', '_')}s")
                    columns.append(f"    m.{col_name},")
                columns.append("")
        
        # 5. Device (행으로 펼치기)
        if row_split_array:
            for parent, fields in row_split_array.items():
                columns.append("    -- ============================================")
                columns.append(f"    -- {parent.upper()} (행으로 펼침)")
                columns.append("    -- ============================================")
                for field in fields:
                    col_name = f"{parent}_{field.replace(':', '_')}"
                    columns.append(f"    d.value:{field}::STRING AS {col_name},")
                columns.append("")
        
        # 마지막 쉼표 제거
        if columns and columns[-1] == "":
            columns.pop()
        if columns and columns[-1].endswith(","):
            columns[-1] = columns[-1][:-1]
        
        lines.extend(columns)
        lines.append("")
        
        # ============================================
        # FROM 절
        # ============================================
        lines.append(f"FROM {table} e")
        
        # JOIN mdr_text_aggregated
        if aggregated_array:
            for parent in aggregated_array.keys():
                lines.append(f"LEFT JOIN {parent}_aggregated m")
                lines.append("    ON e.raw_data:report_number::STRING = m.report_number")
        
        # LATERAL FLATTEN for device
        if row_split_array:
            for parent in row_split_array.keys():
                lines.append(f", LATERAL FLATTEN(input => e.raw_data:{parent}, OUTER => TRUE) AS d")
        
        # LIMIT 절
        if limit is not None:
            lines.append("")
            lines.append(f"LIMIT {limit};")
        else:
            lines.append(";")
        
        sql = '\n'.join(lines)
        logger.info(f"SQL 생성 완료: {len(sql):,} bytes")
        
        return sql
    
    except Exception as e:
        logger.error(f"SQL 생성 실패: {e}", exc_info=True)
        raise


# ============================================
# 실행 스크립트
# ============================================

if __name__ == '__main__':
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    
    try:
        # ============================================
        # 스키마 로드
        # ============================================

        print("스키마 로드 완료")
        print(f"   Scalar: {len(MAUDE_EVENT_SCHEMA['scalar_fields'])}개")
        print(f"   Array: {len(MAUDE_EVENT_SCHEMA['array_fields'])}개")
        print(f"   Patient: {len(MAUDE_EVENT_SCHEMA['first_only']['patient'])}개")
        print(f"   mdr_text: {len(MAUDE_EVENT_SCHEMA['aggregated_array']['mdr_text'])}개")
        print(f"   Device: {len(MAUDE_EVENT_SCHEMA['row_split_array']['device'])}개")
        print()
        
        # ============================================
        # SQL 생성 
        # ============================================
        print("SQL 생성 중...")
        
        sql_full = maude_flatten_sql(
            table='MAUDE.BRONZE.EVENT',
            schema=MAUDE_EVENT_SCHEMA,
            limit=None  # LIMIT 없음
        )
        print(sql_full)        
        
    except ImportError as e:
        print(f"\nImport 실패: {e}")
        print("\n해결 방법:")
        print("   utils/flattener_helper.py 파일을 확인하세요.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("완료!")
    print("="*80)
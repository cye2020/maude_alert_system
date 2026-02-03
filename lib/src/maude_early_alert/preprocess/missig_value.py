"""
Missing Value 처리 SQL 생성 모듈
그룹별 최빈값(MODE)으로 NULL 값 대체하는 SELECT SQL 생성
"""
from typing import Dict, List, Optional

# loguru를 선택적으로 import
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class MissingValue:
    """
    Missing Value 처리 SQL 생성 클래스
    
    그룹별 최빈값(MODE)을 사용하여 NULL 값을 대체하는 SQL 생성
    예: product_code별로 가장 많이 나타나는 device_name으로 NULL 채우기
    """
    
    def __init__(self, table_name: Optional[str] = None):
        """
        Args:
            table_name: 기본 테이블명 (선택사항)
        """
        self.default_table_name = table_name
        if table_name:
            logger.info(f"MissingValue 초기화 (기본 테이블: {table_name})")
        else:
            logger.info(f"MissingValue 초기화 (테이블명 없음)")

    def build_mode_fill_sql(
        self,
        group_to_target: Dict[str, str],
        other_cols: List[str] = None,
        table_name: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        그룹별 최빈값으로 NULL 대체 SQL 생성
        
        Args:
            group_to_target: {그룹_컬럼: 대상_컬럼} 딕셔너리
                예: {'product_code': 'device_name', 'postal_code': 'manufacturer_name'}
            other_cols: NU가져올 컬럼들
            table_name: 테이블명 (미지정 시 초LL 처리 안 하고 그냥 기화 시 지정한 테이블 사용)
            verbose: 상세 로깅 여부
            
        Returns:
            생성된 SELECT SQL 문자열
            
        Examples:
            # product_code별로 가장 많이 나타나는 device_name으로 NULL 채우기
            builder = MissingValue("MAUDE.SILVER.EVENT")
            sql = builder.build_mode_fill_sql(
                group_to_target={
                    'product_code': 'device_name',
                    'postal_code': 'manufacturer_name'
                },
                other_cols=['mdr_report_key', 'date_received']
            )
        """
        # 테이블명 결정
        target_table = table_name or self.default_table_name
        
        if not target_table:
            raise ValueError("table_name을 지정해야 합니다 (초기화 시 또는 메서드 호출 시)")
        
        if not group_to_target:
            raise ValueError("group_to_target 딕셔너리가 비어있습니다")
        
        # CTE 생성: 각 그룹별 최빈값 계산
        cte_parts = []
        for i, (group_col, target_col) in enumerate(group_to_target.items()):
            cte_name = f"mode_{i+1}"
            cte_sql = f"""    {cte_name} AS (
        SELECT
            {group_col},
            MODE({target_col}) AS mode_{target_col}
        FROM
            {target_table}
        WHERE
            {group_col} IS NOT NULL
            AND {target_col} IS NOT NULL
        GROUP BY
            {group_col}
    )"""
            cte_parts.append(cte_sql)
        
        # 전체 CTE
        cte_section = "WITH\n" + ",\n".join(cte_parts)
        
        # SELECT 절 생성
        select_items = []
        
        # other_cols 추가
        if other_cols:
            for col in other_cols:
                select_items.append(f"    t.{col}")
        
        # 그룹 컬럼과 대체된 대상 컬럼 추가
        for i, (group_col, target_col) in enumerate(group_to_target.items()):
            cte_name = f"mode_{i+1}"
            # 그룹 컬럼 (원본 유지)
            select_items.append(f"    t.{group_col}")
            # 대상 컬럼 (NULL이면 최빈값으로 대체)
            select_items.append(
                f"    COALESCE(t.{target_col}, {cte_name}.mode_{target_col}) AS {target_col}"
            )
        
        select_clause = ",\n".join(select_items)
        
        # JOIN 절 생성
        join_parts = []
        for i, (group_col, target_col) in enumerate(group_to_target.items()):
            cte_name = f"mode_{i+1}"
            join_parts.append(
                f"    LEFT JOIN {cte_name}\n"
                f"        ON t.{group_col} = {cte_name}.{group_col}"
            )
        
        join_clause = "\n".join(join_parts)
        
        # 전체 SQL 조립
        sql = f"""-- 그룹별 최빈값으로 NULL 대체
-- 대체 매핑: {dict(group_to_target)}
{cte_section}
SELECT
{select_clause}
FROM
    {target_table} t
{join_clause}"""
        
        if verbose:
            logger.info(f"MODE fill SQL 생성 완료: {len(group_to_target)}개 매핑")
            # logger.debug(f"\n{sql}")
        
        return sql


if __name__ == '__main__':
    print("=" * 80)
    print("Missing Value 처리 SQL Builder 테스트")
    print("=" * 80)

    # ========================================
    # 테스트 : 매핑
    # ========================================
    print("\n" + "=" * 80)
    print("테스트 : 다중 그룹-대상 매핑")
    print("=" * 80)

    builder = MissingValue("MAUDE.SILVER.EVENT_STAGE_04")
    
    sql_multi = builder.build_mode_fill_sql(
        group_to_target={
            'device_device_report_product_code': 'device_openfda_device_name',
            'device_manufacturer_d_postal_code': 'device_manufacturer_d_name'
        },
        other_cols=['mdr_report_key', 'date_received', 'event_type'],
        verbose=True
    )
    print(sql_multi)

    # # ========================================
    # # 테스트 2 : MAUDE와 UDI 각각 처리
    # # ========================================
    # print("\n" + "=" * 80)
    # print("테스트 2 : MAUDE와 UDI 각각 처리")
    # print("=" * 80)
    
    # # 테이블명 없이 초기화
    # builder = MissingValue()
    
    # # MAUDE 처리
    # sql_maude = builder.build_mode_fill_sql(
    #     group_to_target={'product_code': 'device_name'},
    #     other_cols=['mdr_report_key'],
    #     table_name="MAUDE.SILVER.EVENT",
    #     verbose=True
    # )
    # print("\n[MAUDE SQL]")
    # print(sql_maude)
    
    # # UDI 처리
    # sql_udi = builder.build_mode_fill_sql(
    #     group_to_target={'company_name': 'brand_name'},
    #     other_cols=['udi_id'],
    #     table_name="UDI.SILVER.DEVICE",
    #     verbose=True
    # )
    # print("\n[UDI SQL]")
    # print(sql_udi)
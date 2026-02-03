""" 
컬럼 선택 SQL 생성 모듈
컬럼 리스트 → SELECT SQL 생성
"""
from typing import List, Optional
from loguru import logger


class ColDrop:
    """
    컬럼 선택 SQL 생성 클래스
    
    컬럼 리스트를 받아서 SELECT SQL 생성
    테이블명을 메서드 호출 시마다 지정 가능
    """
    
    def __init__(self, table_name: Optional[str] = None):
        """
        Args:
            table_name: 기본 테이블명 (선택사항)
        """
        self.default_table_name = table_name
        if table_name:
            logger.info(f"ColDrop 초기화 (기본 테이블: {table_name})")
        else:
            logger.info(f"ColDrop 초기화 (테이블명 없음)")

    def build_select_sql(
        self,
        cols: List[str],
        table_name: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        SELECT SQL 생성
        
        Args:
            cols: 선택할 컬럼 리스트
            table_name: 테이블명 (미지정 시 초기화 시 지정한 테이블 사용)
            verbose: 상세 로깅 여부
            
        Returns:
            생성된 SELECT SQL 문자열
            
        Examples:
            # 방법 1: 초기화 시 테이블 지정
            builder = ColDrop("MAUDE.SILVER.EVENT")
            sql = builder.build_select_sql(cols=['col1', 'col2'])
            
            # 방법 2: 메서드 호출 시마다 테이블 지정
            builder = ColDrop()
            sql_maude = builder.build_select_sql(cols=['col1'], table_name="MAUDE.SILVER.EVENT")
            sql_udi = builder.build_select_sql(cols=['col2'], table_name="UDI.SILVER.DEVICE")
        """
        # 테이블명 결정
        target_table = table_name or self.default_table_name
        
        if not target_table:
            raise ValueError("table_name을 지정해야 합니다 (초기화 시 또는 메서드 호출 시)")
        
        # 컬럼이 없으면 SELECT * 반환
        if not cols:
            logger.warning(f"선택할 컬럼이 없습니다. SELECT * 반환")
            return f"SELECT * FROM {target_table}"
        
        # 컬럼 선택
        select_cols = ',\n    '.join(cols)
        
        # SQL 생성
        sql = f"""-- 선택된 컬럼: {len(cols)}개
SELECT
    {select_cols}
FROM
    {target_table}"""
        
        if verbose:
            logger.info(f"SELECT SQL 생성 완료: {target_table}, {len(cols)}개 컬럼")
            # logger.debug(f"\n{sql}") # 컬럼이 두번 반복되서 나타남 (일단 주석처리리)
        
        return sql

    def build_1st_filter_sql(
        self,
        cols: List[str],
        table_name: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        1차 필터링 SQL 생성 (스코핑 전)
        
        Args:
            cols: 선택할 컬럼 리스트
            table_name: 테이블명 (선택사항)
            verbose: 상세 로깅 여부
            
        Returns:
            생성된 SELECT SQL 문자열
        """
        logger.info("1차 필터링 SQL 생성 중...")
        return self.build_select_sql(cols=cols, table_name=table_name, verbose=verbose)

    def build_2nd_filter_sql(
        self,
        cols: List[str],
        table_name: Optional[str] = None,
        verbose: bool = False
    ) -> str:
        """
        2차 필터링 SQL 생성 (최종 정제 후)
        
        Args:
            cols: 선택할 컬럼 리스트
            table_name: 테이블명 (선택사항)
            verbose: 상세 로깅 여부
            
        Returns:
            생성된 SELECT SQL 문자열
        """
        logger.info("2차 필터링 SQL 생성 중...")
        return self.build_select_sql(cols=cols, table_name=table_name, verbose=verbose)



if __name__ == '__main__':
    print("=" * 80)
    print("컬럼 필터링 SQL Builder 테스트")
    print("=" * 80)

    # ========================================
    # 방법 1: 각 데이터셋마다 builder 생성
    # ========================================
    print("\n" + "=" * 80)
    print("방법 1: 각 데이터셋마다 별도 builder")
    print("=" * 80)
    
    maude_builder = ColDrop(table_name="MAUDE.SILVER.EVENT_STAGE_02")
    udi_builder = ColDrop(table_name="UDI.SILVER.DEVICE_STAGE_02")
    
    maude_columns = ['mdr_report_key', 'date_received', 'event_type']
    udi_columns = ['udi_id', 'device_name', 'catalog_number']
    
    sql_maude = maude_builder.build_1st_filter_sql(cols=maude_columns, verbose=True)
    print("\n[MAUDE SQL]")
    print(sql_maude)

    maude_columns_2nd = ['mdr_report_key', 'date_received', 'event_type', 'product_code']
    udi_columns_2nd = ['udi_id', 'device_name', 'catalog_number']

    sql_maude_2nd = maude_builder.build_2nd_filter_sql(cols=maude_columns_2nd, verbose=True)
    print("\n[MAUDE SQL]")
    print(sql_maude_2nd)

    # # ========================================
    # # 방법 2: 하나의 builder로 여러 테이블 처리 
    # # ========================================
    # print("\n" + "=" * 80)
    # print("방법 2: 하나의 builder로 여러 테이블 처리")
    # print("=" * 80)
    
    # # 테이블명 없이 초기화
    # builder = ColDrop()
    
    # # MAUDE 처리
    # sql_maude = builder.build_1st_filter_sql(
    #     cols=['mdr_report_key', 'date_received'],
    #     table_name="MAUDE.SILVER.EVENT",
    #     verbose=True
    # )
    # print("\n[MAUDE SQL]")
    # print(sql_maude)
    
    # # UDI 처리
    # sql_udi = builder.build_1st_filter_sql(
    #     cols=['udi_id', 'device_name'],
    #     table_name="UDI.SILVER.DEVICE",
    #     verbose=True
    # )
    # print("\n[UDI SQL]")
    # print(sql_udi)

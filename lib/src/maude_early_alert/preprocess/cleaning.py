from collections import defaultdict
from typing import Optional, Dict, List, Tuple, Union
import cleantext as cl
from country_named_entity_recognition import find_countries
from maude_early_alert.preprocess.udf_registry import (
    generate_clean_text_udf_sql,
    generate_country_removal_udf_sql
)
from maude_early_alert.utils.helpers import ensure_list

NULL_FLAVOR = [
    # --- 단어 경계 매칭: 텍스트 어디에든 있으면 null 의도 ---
    (r'\bUNKNOWN\b', 'DELETE', 'UNKNOWN'),
    (r'\bUNKOWN\b', 'DELETE', 'UNKOWN (오타)'),
    (r'\bNULL\b',    'DELETE', 'NULL'),
    (r'\bNONE\b',    'DELETE', 'NONE'),
    (r'\bNIL\b',     'DELETE', 'NIL'),
    (r'\bN\.A\.?\b', 'DELETE', 'N.A'),
    (r'\bNOT\s+AVAILABLE\b',  'DELETE', 'NOT AVAILABLE'),
    (r'\bNOT\s+SPECIFIED\b',  'DELETE', 'NOT SPECIFIED'),   # 추가
    (r'\bNOT\s+PROVIDED\b',   'DELETE', 'NOT PROVIDED'),    # 추가
    (r'\bUNAVAILABLE\b',      'DELETE', 'UNAVAILABLE'),
    (r'\bUNSPECIFIED\b',      'DELETE', 'UNSPECIFIED'),     # 추가
    (r'\bMISSING\b',          'DELETE', 'MISSING'),         # \b 추가

    # --- 정확 매칭: 짧은 코드 / 충돌 위험 ---
    (r'^N/A$',   'DELETE', 'N/A'),
    (r'^NA$',    'DELETE', 'NA'),
    (r'^NI$',    'DELETE', 'NI'),           # .*\b → ^$
    (r'^UNK$',   'DELETE', 'UNK'),          # .*\b → ^$
    (r'^UKN$',   'DELETE', 'UKN'),          # .*\b → ^$
    (r'^NO[\s_]?DATA$', 'DELETE', 'NO DATA'),
    (r'^EMPTY$', 'DELETE', 'EMPTY'),

    # --- HL7 Null Flavor: 전부 정확 매칭 ---
    (r'^NASK$', 'DELETE', 'NASK (not asked)'),
    (r'^ASKU$', 'DELETE', 'ASKU (asked but unknown)'),
    (r'^TRC$',  'DELETE', 'TRC (trace)'),
    (r'^QS$',   'DELETE', 'QS (sufficient quantity)'),
    (r'^MSK$',  'DELETE', 'MSK (masked)'),
    (r'^NAV$',  'DELETE', 'NAV (temporarily unavailable)'),
    (r'^INV$',  'DELETE', 'INV (invalid)'),       # .*\b → ^$
    (r'^OTH$',  'DELETE', 'OTH (other)'),         # .*\b → ^$
    (r'^PINF$', 'DELETE', 'PINF (positive infinity)'),
    (r'^NINF$', 'DELETE', 'NINF (negative infinity)'),
    (r'^UNC$',  'DELETE', 'UNC (uncertain)'),     # .*\b → ^$
    (r'^DER$',  'DELETE', 'DER (derived)'),       # .*\b → ^$
]
    
class ColumnPipeline:
    """단일 컬럼에 대한 작업 파이프라인"""
    def __init__(self, column: str, builder: 'SQLCleanBuilder'):
        self.column = column
        self.operations = []
        self.builder = builder  # 빌더 참조를 유지
    
    def remove_patterns(self, patterns: List[str]):
        self.operations.append(("remove_patterns", {"patterns": patterns}))
        return self
    
    def delete_patterns(self, patterns: List[str]):
        self.operations.append(("delete_patterns", {"patterns": patterns}))
        return self
    
    def clean_default(self):
        self.operations.append(("clean_default", {}))
        return self
    
    def remove_country_names(self):
        self.operations.append(("remove_country", {}))
        return self
    
    def and_column(self, name: str) -> 'ColumnPipeline':
        """다른 컬럼으로 체이닝"""
        return self.builder.column(name)
    
    def build(self) -> str:
        """편의 메서드: 빌더의 build 호출"""
        return self.builder.build()


class SQLCleanBuilder:
    def __init__(self, table_name: str):
        self.table_name = table_name
        self.pipelines: Dict[str, ColumnPipeline] = {}
    
    def column(self, name: str) -> ColumnPipeline:
        """컬럼 파이프라인 시작 또는 가져오기"""
        if name not in self.pipelines:
            self.pipelines[name] = ColumnPipeline(name, self)
        return self.pipelines[name]
    
    def _build_expression(self, column: str, operations: List[tuple]) -> str:
        """작업들을 중첩된 SQL 표현식으로 변환"""
        expr = column
        
        for op_type, params in operations:
            if op_type == "clean_default":
                expr = f"clean_text_udf({expr})"
                
            elif op_type == "remove_patterns":
                patterns = params['patterns']
                escaped = '|'.join(p.replace("'", "''") for p in patterns)
                expr = f"REGEXP_REPLACE({expr}, '{escaped}', '')"
                
            elif op_type == "delete_patterns":
                patterns = params['patterns']
                escaped = '|'.join(p.replace("'", "''") for p in patterns)
                # delete_patterns는 NULL 체크가 필요하므로 전체 표현식을 래핑
                expr = f"IFF(REGEXP_LIKE({expr}, '{escaped}'), NULL, {expr})"
                
            elif op_type == "remove_country":
                expr = f"remove_country_names_udf({expr})"
        
        return expr
    
    def build(self) -> str:
        """SQL 문 생성"""
        if not self.pipelines:
            raise ValueError("No columns defined. Use .column() to add columns.")
        
        select_items = []
        
        # 각 컬럼의 파이프라인을 순회
        for column_name, pipeline in self.pipelines.items():
            # 작업들을 중첩된 표현식으로 변환
            expr = self._build_expression(column_name, pipeline.operations)
            select_items.append(f"{expr} AS {column_name}")
        
        select_clause = ',\n\t'.join(select_items)
        
        sql = f"""
        SELECT
        \t{select_clause}
        FROM
        \t{self.table_name};
        """
        
        return sql


if __name__=='__main__':
    sql = (SQLCleanBuilder("EVENT")
        .column("mdr_text")
            .remove_patterns(["[0-9]+"])              # 숫자 제거
            .remove_patterns(["[^a-zA-Z가-힣\\s]"])    # 특수문자 제거
            .remove_country_names()                   # 국가명 제거
            .clean_default()                          # 최종 정제
        .and_column("email")
            .remove_patterns(["\\s+"])                # 공백 제거
            .delete_patterns(NULL_FLAVOR)      # 스팸 체크
        .and_column("phone")
            .remove_patterns(["[^0-9-]"])             # 숫자와 하이픈만 남김
            .remove_patterns(["--+"])                 # 연속 하이픈 제거
        .and_column("address")
            .clean_default()
        .build())

    print(sql)
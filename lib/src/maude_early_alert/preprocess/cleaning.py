from typing import Dict, List

from maude_early_alert.utils.sql_builder import build_cte_sql
    
class ColumnPipeline:
    """단일 컬럼에 대한 작업 파이프라인"""
    def __init__(self, column: str, builder: 'SQLCleanBuilder'):
        self.column = column
        self.operations = []
        self.builder = builder  # 빌더 참조를 유지
    
    def remove_patterns(self, patterns: List[str]):
        self.operations.append(("remove", {"patterns": patterns}))
        return self
    
    def delete_patterns(self, patterns: List[str]):
        self.operations.append(("delete", {"patterns": patterns}))
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


class SQLCleanBuilder:
    def __init__(self, udf_schema: str = None):
        self.udf_schema = udf_schema
        self.pipelines: Dict[str, ColumnPipeline] = {}
    
    def column(self, name: str) -> ColumnPipeline:
        """컬럼 파이프라인 시작 또는 가져오기"""
        if name not in self.pipelines:
            self.pipelines[name] = ColumnPipeline(name, self)
        return self.pipelines[name]
    
    def _build_single_expression(self, column: str, op_type: str, params: dict) -> str:
        """단일 연산의 SQL 표현식 생성
        
        정규식 패턴을 Snowflake dollar-quoted string($$)으로 전달하여
        백슬래시 이스케이프 문제를 방지합니다.
        """
        if op_type == "clean_default":
            return f"{self.udf_schema}.clean_text_udf({column})"
        elif op_type == "remove":
            pattern = '|'.join(params['patterns'])
            return f"REGEXP_REPLACE({column}, '{pattern}', '')"
        elif op_type == "delete":
            pattern = '|'.join(params['patterns'])
            return f"IFF(REGEXP_LIKE({column}, '{pattern}'), NULL, {column})"
        elif op_type == "remove_country":
            return f"{self.udf_schema}.remove_country_names_udf({column})"
        return column

    def build_steps(self) -> List[List[str]]:
        """단계별 REPLACE 표현식 리스트 반환

        Returns:
            [["expr AS col", ...], ...] 각 단계의 REPLACE 컬럼 리스트
        """
        if not self.pipelines:
            raise ValueError("No columns defined. Use .column() to add columns.")

        max_ops = max(len(p.operations) for p in self.pipelines.values())
        steps = []

        for i in range(max_ops):
            replace_cols = []
            for name, pipeline in self.pipelines.items():
                if i < len(pipeline.operations):
                    op_type, params = pipeline.operations[i]
                    expr = self._build_single_expression(name, op_type, params)
                    replace_cols.append(f"{expr} AS {name}")
            if replace_cols:
                steps.append(replace_cols)

        return steps


def build_clean_sql(table_name: str, config: dict, null_flavor: dict, udf_schema: str = None) -> str:
    """설정 dict로부터 클리닝 SQL 생성 (CTE 체이닝 방식)

    각 연산 단계를 별도 CTE로 분리하여 표현식 중복을 방지한다.
    적용 순서: null_flavor → patterns (순서 유지) → remove_countries → clean_default
    """
    builder = SQLCleanBuilder(udf_schema=udf_schema)

    for column_name, column_config in config.items():
        pipeline = builder.column(column_name)

        # 'clean' 리스트 처리 (리팩토링된 구조)
        for clean_step in column_config.get('clean', []):
            op_type = clean_step['op_type']
            patterns = clean_step['patterns']
            
            if op_type == 'delete':
                pipeline.delete_patterns(patterns)
            elif op_type == 'remove':
                pipeline.remove_patterns(patterns)

        # 기존 'patterns' 지원 (하위 호환성)
        for op_type, patterns in column_config.get('patterns', []):
            if op_type == 'delete':
                pipeline.delete_patterns(patterns)
            elif op_type == 'remove':
                pipeline.remove_patterns(patterns)

        if column_config.get('remove_countries'):
            pipeline.remove_country_names()

        if column_config.get('clean_default'):
            pipeline.clean_default()

    steps = builder.build_steps()

    ctes = []
    prev_source = table_name
    for i, replace_cols in enumerate(steps):
        step_query = build_cte_sql(
            ctes=[],
            from_clause=f"{prev_source} t",
            table_alias="t",
            replace_cols=replace_cols,
        )
        step_name = f"clean_{i + 1}"
        ctes.append({'name': step_name, 'query': step_query})
        prev_source = step_name

    return build_cte_sql(
        ctes=ctes,
        from_clause=prev_source,
    )


if __name__=='__main__':
    udf_schema = 'UDF'
    
    # HL7 표준 Null Flavor 패턴 정의
    null_flavor = {
        'op_type': 'delete',
        'patterns': [
            # 단어 경계 매칭 - \b는 word boundary
            r'\\bUNKNOWN\\b',
            r'\\bUNKOWN\\b',      # 오타 버전도 포함
            r'\\bNULL\\b',
            r'\\bNONE\\b',
            r'\\bNIL\\b',
            r'\\bN\\.A\\.?\\b',     # N.A 또는 N.A.
            r'\\bNOT\\s+AVAILABLE\\b',
            r'\\bNOT\\s+SPECIFIED\\b',
            r'\\bNOT\\s+PROVIDED\\b',
            r'\\bUNAVAILABLE\\b',
            r'\\bUNSPECIFIED\\b',
            r'\\bMISSING\\b',
            
            # 정확 매칭 - ^는 시작, $는 끝
            r'^N/A$',
            r'^NA$',
            r'^NI$',
            r'^UNK$',
            r'^UKN$',
            r'^NO[\\s_]?DATA$',  # 'NO DATA', 'NO_DATA', 'NODATA'
            r'^EMPTY$',
            
            # HL7 Null Flavor - 의료 데이터 표준 null 코드
            r'^NASK$',          # Not Asked (질문 안 함)
            r'^ASKU$',          # Asked but Unknown (질문했지만 모름)
            r'^TRC$',           # Trace (정량 한계 이하)
            r'^QS$',            # Sufficient Quantity
            r'^MSK$',           # Masked (마스킹됨)
            r'^NAV$',           # Temporarily Unavailable (일시적 불가)
            r'^INV$',           # Invalid (유효하지 않음)
            r'^OTH$',           # Other (기타)
            r'^PINF$',          # Positive Infinity (양의 무한대)
            r'^NINF$',          # Negative Infinity (음의 무한대)
            r'^UNC$',           # Un-encoded (인코딩 안 됨)
            r'^DER$',           # Derived (파생됨)
        ]
    }

    maude_clean = {
        'mdr_text': {
            'remove_countries': False,
            'clean_default': True,
            'patterns': [
                ('remove', [
                    r"[0-9]+"  # 모든 숫자 제거
                ])
            ]
        },
        'udi_di': {
            'remove_countries': False,
            'clean_default': False,
            'clean': [
                {
                    'op_type': 'remove',
                    'patterns': [
                        r'\\(\\d{2}\\)',       # (01) 같은 GS1 Application Identifier 제거
                        r'^\\d*\\+',          # 시작 부분의 숫자+플러스 제거 (예: 123+)
                        r'[^a-zA-Z0-9]',    # 영숫자 외 모든 문자 제거
                    ]
                }, 
                null_flavor,
                {
                    'op_type': 'delete',
                    'patterns': [                 
                        r'^$',              # 빈 문자열 삭제
                        r'^0+$',            # 0만 있는 문자열 삭제 (000, 0000 등)
                        r'^N$',             # 단일 'N' 삭제
                        r'^X+$',            # X만 반복되는 문자열 삭제 (XX, XXX 등)
                        r'^.{2,3}$',        # 2-3자 너무 짧은 DI 삭제
                        r'^.{31,}$',        # 31자 이상 너무 긴 DI 삭제
                    ]
                }
            ]
        }
    }
    
    sql = build_clean_sql("EVENT", maude_clean, null_flavor, udf_schema=udf_schema)
    print(sql)
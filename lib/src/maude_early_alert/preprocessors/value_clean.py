from typing import Dict, List

from maude_early_alert.utils.sql_builder import build_cte_sql

class ColumnPipeline:
    """단일 컬럼에 대한 작업 파이프라인"""
    def __init__(self, column: str, builder: 'SQLCleanBuilder'):
        self.column = column
        self.operations = []
        self.builder = builder

    def remove_patterns(self, patterns: List[str]):
        self.operations.append(("remove", {"patterns": patterns}))
        return self

    def delete_patterns(self, patterns: List[str]):
        self.operations.append(("delete", {"patterns": patterns}))
        return self

    def keep_patterns(self, patterns: List[str]):
        """허용 패턴과 일치하는 값만 유지, 나머지는 NULL"""
        self.operations.append(("keep", {"patterns": patterns}))
        return self

    def filter_array_patterns(self, patterns: List[str]):
        """ARRAY 요소 중 패턴과 일치하는 요소 제거 (FILTER)"""
        self.operations.append(("filter_array", {"patterns": patterns}))
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
        """단일 연산의 SQL 표현식 생성"""
        if op_type == "clean_default":
            return f"{self.udf_schema}.clean_text_udf({column})"
        elif op_type == "remove":
            pattern = '|'.join(params['patterns'])
            return f"REGEXP_REPLACE({column}, '{pattern}', '')"
        elif op_type == "delete":
            pattern = '|'.join(params['patterns'])
            return f"IFF(REGEXP_LIKE({column}, '{pattern}'), NULL, {column})"
        elif op_type == "keep":
            pattern = '|'.join(params['patterns'])
            return f"IFF(REGEXP_LIKE({column}, '{pattern}'), {column}, NULL)"
        elif op_type == "filter_array":
            pattern = '|'.join(params['patterns'])
            return f"FILTER({column}, x -> NOT REGEXP_LIKE(x, '{pattern}'))"
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


def build_clean_sql(table_name: str, config: dict, udf_schema: str = None) -> str:
    """설정 dict로부터 클리닝 SQL 생성 (CTE 체이닝 방식)

    각 연산 단계를 별도 CTE로 분리하여 표현식 중복을 방지한다.

    config 형식:
        {
            'COLUMN_NAME': [
                {'op_type': 'remove', 'patterns': [...]},
                {'op_type': 'remove_country'},
                {'op_type': 'delete', 'patterns': [...]},
                {'op_type': 'clean_default'},
            ],
            ...
        }
    """
    builder = SQLCleanBuilder(udf_schema=udf_schema)

    for column_name, operations in config.items():
        pipeline = builder.column(column_name)

        for step in operations:
            op_type = step['op_type']

            if op_type == 'remove_country':
                pipeline.remove_country_names()
            elif op_type == 'clean_default':
                pipeline.clean_default()
            elif op_type == 'delete':
                pipeline.delete_patterns(step['patterns'])
            elif op_type == 'remove':
                pipeline.remove_patterns(step['patterns'])
            elif op_type == 'keep':
                pipeline.keep_patterns(step['patterns'])
            elif op_type == 'filter_array':
                pipeline.filter_array_patterns(step['patterns'])

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
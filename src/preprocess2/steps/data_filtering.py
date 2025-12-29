"""
# regex 패턴에 매칭되는 컬럼에 대해 비교해 필터링
result = udi_lf.filter(
    pl.any_horizontal(cs.matches('*device_class$') == '3')
)

# 한 컬럼 / 여러 컬럼에 대해 비교해 필터링

컬럼이나 비교 op는 config에서 설정 가능
"""

import polars as pl
import polars.selectors as cs
from typing import List, Tuple, Any, Optional, Dict
import logging


class FilterStage:
    """행 필터링 유틸리티
    
    컬럼/패턴 기반으로 조건에 맞는 행만 유지
    """
    
    def __init__(self, verbose: bool = True):
        """
        Args:
            verbose: 로깅 활성화 여부
        """
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)
        
        # 통계 추적
        self.stats = {
            'original_rows': 0,
            'filtered_rows': 0,
            'kept_rows': 0,
            'filter_conditions': [],
        }
        
        # operator 매핑
        self._ops = {
            'eq': lambda col, val: col.eq(val),
            'ne': lambda col, val: col.ne(val),
            'gt': lambda col, val: col.gt(val),
            'ge': lambda col, val: col.ge(val),
            'lt': lambda col, val: col.lt(val),
            'le': lambda col, val: col.le(val),
            'is_null': lambda col, val: col.is_null(),
            'is_not_null': lambda col, val: col.is_not_null(),
            'is_in': lambda col, val: col.is_in(val),
            'not_in': lambda col, val: col.is_in(val).not_(),
        }

    def filter_groups(self, 
        lf: pl.LazyFrame, 
        groups: List[dict], 
        combine_groups: str = 'AND'
    ) -> pl.LazyFrame:
        
        group_conditions = []
        
        for group in groups:
            group_name = group.get('name', 'unnamed')
            group_combine = group.get('combine', 'AND')
            
            cols = group.get('cols', [])
            patterns = group.get('patterns', [])
            
            group_condition = self._build_group_condition(cols, patterns, group_combine)
            group_conditions.append(group_condition)
        
        # 그룹 간 결합
        if not group_conditions:
            return lf
        
        if combine_groups.upper() == 'AND':
            final_condition = pl.all_horizontal(group_conditions)
        else:
            final_condition = pl.any_horizontal(group_conditions)
        
        result_lf = lf.filter(final_condition)
        
        if self.verbose:
            self.logger.info(
                f"Applied {len(group_conditions)} group(s) with {combine_groups}"
            )
        
        return result_lf
    
    def _build_group_condition(
        self,
        cols: Optional[List[Tuple[List[str], str, Any]]] = None,
        patterns: Optional[List[Tuple[List[str], str, Any]]] = None,
        combine: str = 'AND'
    ) -> pl.Expr:
        """조건 기반 행 필터링
        
        Args:
            lf: 입력 LazyFrame
            cols: [(컬럼 리스트, operator, 비교값), ...]
                예: [(['col1', 'col2'], 'eq', 3), (['col4'], 'is_not_null', None)]
            patterns: [(패턴 리스트, operator, 비교값), ...]
                예: [(['*_age$'], 'gt', 0), (['device_*'], 'is_not_null', None)]
            combine: 'AND' (모두 만족) 또는 'OR' (하나라도 만족)
            
        Returns:
            필터링된 LazyFrame
            
        Examples:
            >>> # 나이가 0보다 크고 null이 아닌 행만 유지
            >>> filter(lf, cols=[(['age'], 'gt', 0), (['age'], 'is_not_null', None)])
            
            >>> # device_class가 '3'인 행 유지 (패턴 매칭)
            >>> filter(lf, patterns=[(['*device_class$'], 'eq', '3')])
        """
        # 원본 행 수 저장 (나중에 collect시 계산)
        self.stats['filter_conditions'] = []
        
        cols = cols or []
        patterns = patterns or []
        
        # 필터 조건 생성
        conditions = []
        
        # 1. 컬럼 기반 필터
        for col_list, op, value in cols:
            condition = self._build_column_condition(col_list, op, value)
            conditions.append(condition)
            self.stats['filter_conditions'].append({
                'type': 'column',
                'columns': col_list,
                'operator': op,
                'value': value
            })
        
        # 2. 패턴 기반 필터
        for pattern_list, op, value in patterns:
            condition = self._build_pattern_condition(pattern_list, op, value)
            conditions.append(condition)
            self.stats['filter_conditions'].append({
                'type': 'pattern',
                'patterns': pattern_list,
                'operator': op,
                'value': value
            })
        
        # 조건 결합
        if combine.upper() == 'AND':
            group_condition = pl.all_horizontal(conditions)
        elif combine.upper() == 'OR':
            group_condition = pl.any_horizontal(conditions)
        else:
            raise ValueError(f"Invalid combine: {combine}. Use 'AND' or 'OR'")

        # 로깅
        if self.verbose:
            self.logger.info(f"Applied {len(conditions)} filter condition(s) with {combine}")
            for i, cond_info in enumerate(self.stats['filter_conditions'], 1):
                if cond_info['type'] == 'column':
                    self.logger.info(
                        f"  [{i}] Columns {cond_info['columns']} "
                        f"{cond_info['operator']} {cond_info['value']}"
                    )
                else:
                    self.logger.info(
                        f"  [{i}] Patterns {cond_info['patterns']} "
                        f"{cond_info['operator']} {cond_info['value']}"
                    )
        
        return group_condition
    
    def _build_column_condition(self, col_list: List[str], op: str, value: Any) -> pl.Expr:
        """컬럼 기반 조건 생성"""
        if op not in self._ops:
            raise ValueError(
                f"Unknown operator: {op}. "
                f"Available: {list(self._ops.keys())}"
            )
        
        # 각 컬럼에 대해 조건 생성
        col_conditions = [self._ops[op](pl.col(col), value) for col in col_list]
        
        # 여러 컬럼이면 any_horizontal로 결합 (하나라도 만족하면 OK)
        if len(col_conditions) == 1:
            return col_conditions[0]
        else:
            return pl.any_horizontal(col_conditions)
    
    def _build_pattern_condition(self, pattern_list: List[str], op: str, value: Any) -> pl.Expr:
        """패턴 기반 조건 생성"""
        if op not in self._ops:
            raise ValueError(
                f"Unknown operator: {op}. "
                f"Available: {list(self._ops.keys())}"
            )
        
        # 패턴 결합
        regex = "|".join(pattern_list)
        
        # 매칭된 컬럼들에 대해 조건 적용
        return pl.any_horizontal(self._ops[op](cs.matches(regex), value))
    
    def get_stats(self) -> Dict:
        """필터링 통계 반환"""
        return self.stats.copy()
    
    def print_stats(self, lf_before: pl.LazyFrame, lf_after: pl.LazyFrame):
        """필터링 전후 통계 출력 (디버깅용)
        
        Args:
            lf_before: 필터링 전 LazyFrame
            lf_after: 필터링 후 LazyFrame
        """
        # collect 필요
        rows_before = lf_before.select(pl.len()).collect().item()
        rows_after = lf_after.select(pl.len()).collect().item()
        rows_filtered = rows_before - rows_after
        
        self.logger.info(f"\n필터링 결과:")
        self.logger.info(f"  원본: {rows_before:,} rows")
        self.logger.info(f"  유지: {rows_after:,} rows")
        self.logger.info(f"  제거: {rows_filtered:,} rows ({rows_filtered/rows_before*100:.2f}%)")
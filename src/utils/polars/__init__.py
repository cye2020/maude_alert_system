"""Polars 범용 유틸리티 패키지

재사용 가능한 Polars DataFrame/LazyFrame 조작 함수들
"""

from src.utils.polars.patterns import get_pattern_cols, get_use_cols
from src.utils.polars.memory import (
    estimate_string_size_stats,
    get_unique,
    get_unique_by_cols,
    get_unique_by_cols_safe,
    groupby_nunique_safe,
)
from src.utils.polars.batch import process_in_chunks, collect_unique_safe

__all__ = [
    # 패턴 매칭
    'get_pattern_cols',
    'get_use_cols',
    # 메모리 안전 연산
    'estimate_string_size_stats',
    'get_unique',
    'get_unique_by_cols',
    'get_unique_by_cols_safe',
    'groupby_nunique_safe',
    # 배치 처리
    'process_in_chunks',
    'collect_unique_safe',
]

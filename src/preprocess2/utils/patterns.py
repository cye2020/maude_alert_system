from typing import List
import re

import polars as pl


def get_pattern_cols(
    lf: pl.LazyFrame,
    pattern: List[str],
) -> List[str]:
    """정규표현식 패턴에 매칭되는 컬럼명 추출
    
    Args:
        lf (pl.LazyFrame): 대상 LazyFrame
        pattern (List[str]): 정규표현식 패턴 리스트
    
    Returns:
        List[str]: 패턴에 매칭되는 컬럼명 리스트
    
    Examples:
        >>> get_pattern_cols(lf, [r'^device_\d+', r'.*_date$'])
        ['device_0_name', 'device_1_name', 'report_date', 'event_date']
    """
    # 모든 컬럼명 가져오기
    cols = lf.collect_schema().names()
    
    # 패턴 문자열을 정규표현식 객체로 컴파일
    regexes = [re.compile(p) for p in pattern]
    
    # 각 컬럼명이 패턴 중 하나라도 매칭되면 포함
    return [c for c in cols if any(r.search(c) for r in regexes)]
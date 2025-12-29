"""Polars DataFrame/LazyFrame 컬럼 패턴 매칭 유틸리티"""

import re
from typing import List, Dict, Tuple
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


def get_use_cols(
    lf: pl.LazyFrame,
    patterns: Dict[str, List[str]],
    base_cols: List[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """기본 컬럼과 패턴별 컬럼을 합쳐 분석용 컬럼 리스트 생성

    Args:
        lf (pl.LazyFrame): 대상 LazyFrame
        patterns (Dict[str, List[str]]): 카테고리별 정규표현식 패턴 딕셔너리
            예: {'device': [r'^device_'], 'patient': [r'^patient_']}
        base_cols (List[str]): 기본적으로 포함할 컬럼 리스트

    Returns:
        Tuple[List[str], Dict[str, List[str]]]:
            - 전체 분석 컬럼 리스트 (중복 제거, 역순 정렬)
            - 카테고리별 컬럼 딕셔너리

    Examples:
        >>> patterns = {'device': [r'^device_'], 'event': [r'event_']}
        >>> base_cols = ['report_id', 'date_received']
        >>> all_cols, pattern_cols = get_use_cols(lf, patterns, base_cols)
    """
    # 기본 컬럼으로 시작
    analysis_cols = base_cols

    # 패턴별로 컬럼 추출 및 저장
    pattern_cols = {}
    for k, pattern in patterns.items():
        pattern_cols[k] = get_pattern_cols(lf, pattern)
        analysis_cols += pattern_cols[k]

    # 중복 제거 후 역순 정렬
    analysis_cols = sorted(list(set(analysis_cols)), reverse=True)

    # 요약 정보 출력
    print(f"총 컬럼: {len(analysis_cols)}개")
    for k, pattern in pattern_cols.items():
        print(f"{k} 컬럼: {len(pattern)}개")

    return analysis_cols, pattern_cols

"""데이터 변환 함수들

데이터 타입 변환 및 정제 작업을 수행하는 함수
"""

from typing import List, Union
import polars as pl


def replace_pattern_with_null(lf: pl.LazyFrame, cols: Union[str, List[str]], na_pattern: str) -> pl.LazyFrame:
    """지정된 컬럼들에서 정규식 패턴과 매칭되는 값을 null로 변경

    대소문자 구분 없이 패턴과 매칭되는 모든 값을 null로 치환합니다.
    결측치를 나타내는 다양한 표현('N/A', 'UNKNOWN', 'NONE' 등)을 통일된 null로 변환할 때 유용합니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        처리할 LazyFrame
    cols : str or List[str]
        처리할 컬럼명 (단일 컬럼 문자열 또는 컬럼명 리스트)
    na_pattern : str
        null로 변경할 정규식 패턴
        예: r'^(N/A|UNKNOWN|NONE|NA)$' - 정확히 이 값들만 매칭
            r'UNKNOWN' - UNKNOWN이 포함된 모든 값 매칭

    Returns:
    --------
    pl.LazyFrame
        패턴에 매칭된 값이 null로 변경된 LazyFrame

    Examples:
    ---------
    >>> # 단일 컬럼 처리
    >>> lf = replace_pattern_with_null(lf, 'device_name', r'^(N/A|UNKNOWN)$')

    >>> # 여러 컬럼 동시 처리
    >>> lf = replace_pattern_with_null(
    ...     lf,
    ...     ['device_name', 'manufacturer', 'model'],
    ...     r'^(N/A|UNKNOWN|NONE|NA|-|NULL)$'
    ... )

    Notes:
    ------
    - 대소문자를 구분하지 않습니다 (자동으로 대문자로 변환 후 비교)
    - 원본 컬럼명을 유지합니다 (.name.keep())
    """
    # 단일 컬럼 문자열을 리스트로 변환
    if isinstance(cols, str):
        cols = [cols]

    # 패턴 매칭된 값을 null로 변경
    replace_null_lf = lf.with_columns(
        pl.when(pl.col(cols).str.to_uppercase().str.contains(na_pattern))  # 대문자 변환 후 패턴 검사
        .then(None)  # 매칭되면 null
        .otherwise(pl.col(cols))  # 매칭 안 되면 원본 유지
        .name.keep()  # 원본 컬럼명 유지
    )
    return replace_null_lf


def yn_to_bool(lf: pl.LazyFrame, cols: List[str]) -> pl.LazyFrame:
    """Y/N 문자열 값을 boolean 타입으로 변환

    'Y'는 True로, 'N'은 False로 변환합니다.
    대소문자를 구분하지 않으며, Y/N이 아닌 값은 null이 됩니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        변환할 LazyFrame
    cols : List[str]
        변환할 컬럼명 리스트

    Returns:
    --------
    pl.LazyFrame
        Y/N 값이 boolean으로 변환된 LazyFrame

    Examples:
    ---------
    >>> # 단일 컬럼 변환
    >>> lf = yn_to_bool(lf, ['report_to_fda'])

    >>> # 여러 컬럼 동시 변환
    >>> lf = yn_to_bool(lf, [
    ...     'report_to_fda',
    ...     'report_to_manufacturer',
    ...     'device_operator_known'
    ... ])

    Notes:
    ------
    - 'Y', 'y' → True
    - 'N', 'n' → False
    - 그 외 값 → None (null)
    - 원본이 이미 null인 경우 null 유지
    """
    bool_lf = lf.with_columns([
        pl.col(col)
        .str.to_uppercase()  # 대소문자 통일 (Y/N으로 변환)
        .replace({'Y': True, 'N': False})  # Y→True, N→False, 나머지→null
        .alias(col)  # 동일한 컬럼명으로 덮어쓰기
        for col in cols
    ])
    return bool_lf


def str_to_categorical(lf: pl.LazyFrame, cols: List[str]) -> pl.LazyFrame:
    """String 타입 컬럼을 Categorical 타입으로 변환

    고유값(unique value)이 적은 컬럼을 Categorical로 변환하면:
    - 메모리 사용량 감소 (문자열을 정수 인덱스로 저장)
    - groupby, join 등의 연산 속도 향상
    - 정렬 및 필터링 성능 개선

    Parameters:
    -----------
    lf : pl.LazyFrame
        변환할 LazyFrame
    cols : List[str]
        Categorical로 변환할 컬럼명 리스트

    Returns:
    --------
    pl.LazyFrame
        지정된 컬럼이 Categorical 타입으로 변환된 LazyFrame

    Examples:
    ---------
    >>> # 단일 컬럼 변환
    >>> lf = str_to_categorical(lf, ['device_class'])

    >>> # 여러 컬럼 동시 변환 (고유값이 적은 컬럼들)
    >>> lf = str_to_categorical(lf, [
    ...     'device_class',      # 예: 1, 2, 3
    ...     'event_type',        # 예: Injury, Malfunction, Death
    ...     'report_source',     # 예: Manufacturer, User Facility, Distributor
    ...     'country'            # 예: US, CA, UK, JP, ...
    ... ])

    >>> # 타입 확인
    >>> lf.collect().schema

    Notes:
    ------
    - 고유값이 많은 컬럼(예: ID, 이름)은 변환하지 않는 것이 좋습니다
    - 일반적으로 고유값이 전체 행의 5% 미만일 때 효과적입니다
    - Categorical 타입은 기본적으로 "physical" 순서를 사용합니다
    """
    # 지정된 컬럼들을 Categorical 타입으로 캐스팅
    categorical_lf = lf.with_columns(
        pl.col(cols).cast(pl.Categorical)
    )
    return categorical_lf

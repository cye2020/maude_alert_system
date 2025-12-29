"""MDR (Medical Device Report) 텍스트 처리 함수들

MDR 텍스트 데이터의 중복 제거 및 결합 작업을 수행하는 함수
"""

import re
import polars as pl


def combine_mdr_texts(lf: pl.LazyFrame) -> pl.LazyFrame:
    """여러 MDR 텍스트 컬럼을 중복 제거하여 하나로 결합

    mdr_text_*_text와 mdr_text_*_text_type_code 컬럼 쌍을 찾아
    중복된 텍스트를 제거하고 [타입코드] 형식으로 포맷팅하여 결합합니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        MDR 텍스트 컬럼을 포함한 LazyFrame

    Returns:
    --------
    pl.LazyFrame
        'combined_mdr_text' 컬럼이 추가된 LazyFrame
        중복이 제거되고 포맷팅된 텍스트가 \\n\\n으로 구분되어 저장됨

    Examples:
    ---------
    >>> # mdr_text_1_text, mdr_text_1_text_type_code,
    >>> # mdr_text_2_text, mdr_text_2_text_type_code 컬럼이 있는 경우
    >>> lf = combine_mdr_texts(lf)
    >>> # combined_mdr_text 컬럼에 다음과 같은 형식으로 저장:
    >>> # [Type1]
    >>> # 텍스트 내용 1
    >>> #
    >>> # [Type2]
    >>> # 텍스트 내용 2

    Notes:
    ------
    - mdr_text_*_text와 mdr_text_*_text_type_code 패턴의 컬럼만 처리
    - 같은 텍스트가 여러 컬럼에 중복되어 있으면 한 번만 포함
    - 텍스트가 없거나 빈 문자열인 경우 제외
    - 타입 코드가 없으면 빈 문자열로 표시
    """
    cols = lf.collect_schema().names()
    text_cols = sorted([c for c in cols if c.startswith('mdr_text_') and c.endswith('_text')])

    # 텍스트-타입코드 쌍 찾기
    pairs = []
    for text_col in text_cols:
        type_col = re.sub(r'_text$', '_text_type_code', text_col)
        if type_col in cols:
            pairs.append((text_col, type_col))

    # 쌍이 없으면 None 컬럼 추가
    if not pairs:
        return lf.with_columns(pl.lit(None).alias('combined_mdr_text'))

    # 1. 중복 제거된 리스트를 컬럼에 할당
    lf = lf.with_columns(
        pl.struct([pl.col(tc) for tc, _ in pairs] + [pl.col(ty) for _, ty in pairs])
        .map_elements(
            lambda s: deduplicate_and_format(s, pairs),
            return_dtype=pl.List(pl.String)
        )
        .alias('deduplicated_formatted')
    )

    # 2. 리스트를 문자열로 결합
    lf = lf.with_columns(
        pl.col('deduplicated_formatted')
        .list.join("\n\n")
        .alias('combined_mdr_text')
    )

    return lf.drop('deduplicated_formatted')


def deduplicate_and_format(struct_val, pairs):
    """텍스트 중복 제거하고 [타입코드] 형식으로 포맷팅

    내부 헬퍼 함수로, combine_mdr_texts에서 사용됩니다.

    Parameters:
    -----------
    struct_val : dict
        텍스트와 타입코드 컬럼 값들을 담은 struct
    pairs : list of tuples
        (text_col, type_col) 형태의 컬럼명 쌍 리스트

    Returns:
    --------
    list
        포맷팅된 텍스트 리스트 (중복 제거됨)

    Notes:
    ------
    - 같은 텍스트는 처음 한 번만 포함
    - None이나 빈 문자열은 제외
    - 각 텍스트는 "[타입코드]\\n텍스트" 형식으로 포맷팅
    """
    seen = {}
    result = []

    for text_col, type_col in pairs:
        text = struct_val.get(text_col)
        type_val = struct_val.get(type_col)

        if text is not None and text != "" and text not in seen:
            seen[text] = True
            type_display = type_val if type_val else ""
            result.append(f"[{type_display}]\n{text}")

    return result

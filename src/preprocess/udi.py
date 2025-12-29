"""UDI (Unique Device Identification) 처리 함수들

UDI 데이터 전처리 및 매칭을 위한 도메인 특화 함수
"""

import re
from typing import Optional
from rapidfuzz import fuzz


def extract_di_from_public(udi_public: str) -> Optional[str]:
    """UDI-Public에서 DI (Device Identifier) 추출

    UDI-Public 문자열에서 디바이스 식별자(DI)를 추출합니다.
    두 가지 패턴을 시도합니다:
    1. GS1 형식: (01)로 시작하는 14자리 숫자
    2. + 기호 뒤의 문자열 (슬래시나 달러 기호 전까지)

    Parameters:
    -----------
    udi_public : str
        UDI-Public 문자열

    Returns:
    --------
    Optional[str]
        추출된 DI 문자열, 실패 시 None

    Examples:
    ---------
    >>> extract_di_from_public("(01)12345678901234")
    '12345678901234'

    >>> extract_di_from_public("+ABC123DEF/LOT456")
    'ABC123DEF'

    >>> extract_di_from_public("")
    None
    """
    if not udi_public:
        return None

    # GS1 형식: (01) 뒤 14자리 숫자
    match = re.search(r'\(01\)(\d{14})', str(udi_public))
    if match:
        return match.group(1)

    # + 기호 뒤의 문자열 (/ 또는 $ 전까지)
    match = re.search(r'\+([^/\$]+)', str(udi_public))
    if match:
        return match.group(1).strip()

    return None


def fuzzy_match_dict(source_list: list, target_list: list, threshold: int = 85) -> dict:
    """리스트 간 퍼지 매칭을 통한 매핑 딕셔너리 생성

    source_list의 각 항목에 대해 target_list에서 가장 유사한 항목을 찾아 매핑합니다.
    유사도가 threshold 이상인 경우만 매칭하고, 그렇지 않으면 원본 값을 유지합니다.

    Parameters:
    -----------
    source_list : list
        매칭할 원본 리스트
    target_list : list
        매칭 대상 리스트
    threshold : int, default=85
        매칭으로 인정할 최소 유사도 (0-100)

    Returns:
    --------
    dict
        {source_item: matched_target_item} 형태의 매핑 딕셔너리
        매칭 실패 시 원본 값을 그대로 유지

    Examples:
    ---------
    >>> source = ["ACME Corp", "XYZ Company"]
    >>> target = ["Acme Corporation", "XYZ Co."]
    >>> fuzzy_match_dict(source, target, threshold=80)
    {'ACME Corp': 'Acme Corporation', 'XYZ Company': 'XYZ Co.'}

    >>> # threshold를 충족하지 못하면 원본 유지
    >>> fuzzy_match_dict(["Apple"], ["Microsoft"], threshold=90)
    {'Apple': 'Apple'}

    Notes:
    ------
    - 대소문자를 구분하지 않고 비교합니다
    - 유사도는 Levenshtein distance 기반 비율로 계산됩니다 (0-100)
    - 각 source 항목마다 target 전체를 순회하므로 O(n*m) 시간 복잡도
    - 대용량 리스트의 경우 성능에 주의가 필요합니다
    """
    mapping = {}

    for src in source_list:
        if not src:
            continue

        best_score = 0
        best_match = src

        for tgt in target_list:
            if not tgt:
                continue
            score = fuzz.ratio(str(src).lower(), str(tgt).lower())
            if score > best_score:
                best_score = score
                best_match = tgt

        mapping[src] = best_match if best_score >= threshold else src

    return mapping

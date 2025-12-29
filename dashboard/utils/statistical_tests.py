"""통계적 검정 함수들"""

import numpy as np
from scipy import stats
from typing import Tuple


def fisher_exact_test(death_a: int, total_a: int, death_b: int, total_b: int) -> Tuple[float, float]:
    """Fisher's Exact Test를 사용한 CFR 비교

    Args:
        death_a: A 그룹의 사망 수
        total_a: A 그룹의 총 케이스 수
        death_b: B 그룹의 사망 수
        total_b: B 그룹의 총 케이스 수

    Returns:
        (odds_ratio, p_value)
    """
    # 2x2 contingency table
    # [[death_a, survival_a],
    #  [death_b, survival_b]]
    survival_a = total_a - death_a
    survival_b = total_b - death_b

    table = np.array([
        [death_a, survival_a],
        [death_b, survival_b]
    ])

    odds_ratio, p_value = stats.fisher_exact(table, alternative='two-sided')

    return odds_ratio, p_value


def chi2_test(death_a: int, total_a: int, death_b: int, total_b: int) -> Tuple[float, float]:
    """Chi-square test를 사용한 CFR 비교

    Args:
        death_a: A 그룹의 사망 수
        total_a: A 그룹의 총 케이스 수
        death_b: B 그룹의 사망 수
        total_b: B 그룹의 총 케이스 수

    Returns:
        (chi2_stat, p_value)
    """
    survival_a = total_a - death_a
    survival_b = total_b - death_b

    table = np.array([
        [death_a, survival_a],
        [death_b, survival_b]
    ])

    chi2_stat, p_value, dof, expected = stats.chi2_contingency(table)

    return chi2_stat, p_value


def get_significance_level(p_value: float) -> str:
    """p-value를 유의 수준 문자열로 변환

    Args:
        p_value: p-value

    Returns:
        유의 수준 문자열 (***/**/*/ )
    """
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    else:
        return ""


def interpret_significance(p_value: float, alpha: float = 0.05) -> str:
    """통계적 유의성 해석

    Args:
        p_value: p-value
        alpha: 유의 수준 (기본값: 0.05)

    Returns:
        해석 문자열
    """
    if p_value < alpha:
        if p_value < 0.001:
            return "매우 유의함 (p < 0.001)"
        elif p_value < 0.01:
            return "유의함 (p < 0.01)"
        else:
            return "유의함 (p < 0.05)"
    else:
        return "유의하지 않음 (p ≥ 0.05)"


def calculate_confidence_interval(
    deaths: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """CFR의 신뢰구간 계산 (Wilson score interval)

    Args:
        deaths: 사망 수
        total: 총 케이스 수
        confidence: 신뢰 수준 (기본값: 0.95)

    Returns:
        (lower_bound, upper_bound) - 백분율로 반환
    """
    if total == 0:
        return 0.0, 0.0

    p = deaths / total
    z = stats.norm.ppf((1 + confidence) / 2)

    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    adjustment = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator

    lower = max(0, centre - adjustment)
    upper = min(1, centre + adjustment)

    return lower * 100, upper * 100

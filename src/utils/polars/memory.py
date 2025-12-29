"""Polars 메모리 안전 연산 유틸리티

대용량 데이터 처리 시 메모리 오버플로우를 방지하는 함수들
"""

import sys
import psutil
from typing import List, Dict
import polars as pl
from tqdm import tqdm


def estimate_string_size_stats(lf: pl.LazyFrame, cols: List[str], sample_size: int = 10000) -> Dict[str, float]:
    """컬럼들의 문자열 길이 통계를 샘플 데이터로 추정

    Parameters:
    -----------
    lf : pl.LazyFrame
        샘플링할 LazyFrame
    cols : List[str]
        측정할 컬럼 리스트
    sample_size : int, default=10000
        샘플링할 행 수

    Returns:
    --------
    Dict[str, float]
        문자열 길이 통계
        - 'mean': 평균
        - 'median': 중앙값 (50th percentile)
        - 'p75': 75th percentile
        - 'p90': 90th percentile
        - 'max': 최댓값

    Examples:
    ---------
    >>> stats = estimate_string_size_stats(lf, ['device_0_name', 'device_1_name'])
    >>> print(f"평균: {stats['mean']:.1f}자")
    >>> print(f"중앙값: {stats['median']:.1f}자")
    >>> print(f"75%: {stats['p75']:.1f}자")
    """
    try:
        # 샘플 데이터 추출
        sample = (
            lf.select(cols)
            .head(sample_size)
            .collect(engine='streaming')
        )

        # 모든 값의 길이 수집
        lengths = []
        for col in cols:
            values = sample[col].drop_nulls().to_list()
            lengths.extend([len(str(v)) for v in values])

        if not lengths:
            return {
                'mean': 50.0,
                'median': 50.0,
                'p75': 50.0,
                'p90': 50.0,
                'max': 50.0
            }

        # Polars로 통계 계산
        lengths_series = pl.Series(lengths)

        stats = {
            'mean': lengths_series.mean(),
            'median': lengths_series.median(),
            'p75': lengths_series.quantile(0.75),
            'p90': lengths_series.quantile(0.90),
            'max': lengths_series.max()
        }

        return stats

    except Exception as e:
        print(f"  ⚠️ 문자열 길이 통계 추정 실패: {e}")
        return {
            'mean': 50.0,
            'median': 50.0,
            'p75': 50.0,
            'p90': 50.0,
            'max': 50.0
        }


def get_unique(lf: pl.LazyFrame, cols: List[str]) -> set:
    """여러 컬럼의 모든 고유값을 하나의 set으로 반환

    여러 컬럼에 걸쳐 나타나는 모든 고유한 값들을 추출합니다.
    예를 들어, device_0_name, device_1_name, device_2_name 컬럼들의
    모든 고유한 디바이스 이름을 한 번에 가져올 때 유용합니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        데이터를 추출할 LazyFrame
    cols : List[str]
        고유값을 추출할 컬럼명 리스트

    Returns:
    --------
    set
        모든 컬럼의 고유값을 합친 set (중복 제거됨)

    Examples:
    ---------
    >>> # 여러 device 컬럼의 모든 고유한 디바이스 이름
    >>> device_cols = ['device_0_name', 'device_1_name', 'device_2_name']
    >>> all_devices = get_unique(lf, device_cols)
    >>> print(f"총 고유 디바이스: {len(all_devices)}개")

    >>> # 여러 날짜 컬럼의 모든 고유 날짜
    >>> date_cols = ['event_date', 'report_date', 'received_date']
    >>> all_dates = get_unique(lf, date_cols)

    Notes:
    ------
    - unpivot으로 모든 컬럼을 하나의 컬럼으로 합친 후 unique 추출
    - streaming 엔진 사용으로 메모리 효율성 향상
    - null 값도 set에 포함됨 (None으로 표시)
    - 결과가 메모리에 완전히 로드되므로 고유값이 매우 많으면 주의 필요
    """
    unique_set = set(
        lf.select(cols)  # 지정된 컬럼만 선택
        .unpivot(on=cols)  # 모든 컬럼을 'value' 컬럼 하나로 합치기
        .select('value')  # value 컬럼만 선택
        .unique()  # 중복 제거
        .drop_nulls()  # 결측치 제거
        .collect(engine='streaming')  # streaming 엔진으로 실행 (메모리 효율)
        ['value']  # value 컬럼 추출
    )
    return unique_set


def get_unique_by_cols(lf: pl.LazyFrame, cols_group: Dict[str, List[str]]) -> Dict[str, set]:
    """컬럼 그룹별로 고유값을 추출하여 딕셔너리로 반환

    여러 컬럼 그룹에 대해 각각 고유값을 추출합니다.
    예를 들어, device 관련 컬럼들, patient 관련 컬럼들의 고유값을
    각각 별도로 추출할 때 유용합니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        데이터를 추출할 LazyFrame
    cols_group : Dict[str, List[str]]
        그룹명을 키로, 컬럼 리스트를 값으로 하는 딕셔너리
        예: {
            'devices': ['device_0_name', 'device_1_name'],
            'manufacturers': ['device_0_manufacturer', 'device_1_manufacturer']
        }

    Returns:
    --------
    Dict[str, set]
        각 그룹의 고유값 set을 담은 딕셔너리
        예: {
            'devices': {'Device A', 'Device B', ...},
            'manufacturers': {'Company X', 'Company Y', ...}
        }

    Examples:
    ---------
    >>> # 여러 카테고리별 고유값 추출
    >>> cols_group = {
    ...     'devices': ['device_0_name', 'device_1_name', 'device_2_name'],
    ...     'manufacturers': ['device_0_manufacturer', 'device_1_manufacturer'],
    ...     'models': ['device_0_model', 'device_1_model']
    ... }
    >>> unique_values = get_unique_by_cols(lf, cols_group)
    >>>
    >>> # 결과 확인
    >>> for group, values in unique_values.items():
    ...     print(f"{group}: {len(values)}개의 고유값")

    >>> # 특정 그룹의 값 접근
    >>> all_devices = unique_values['devices']
    >>> all_manufacturers = unique_values['manufacturers']

    WARNINGS:
    ---------
    ⚠️ **메모리 사용 주의사항**:
    - 이 함수는 각 그룹별로 순차적으로 고유값을 추출합니다
    - 각 그룹의 고유값이 메모리에 완전히 로드됩니다
    - 고유값이 매우 많은 경우(수십만~수백만 개) 메모리 부족 발생 가능

    **메모리 절약 대안**:
    - 고유값 개수만 필요한 경우: n_unique() 사용
    - 샘플만 필요한 경우: .unique().limit(n) 사용
    - 매우 큰 데이터: 배치 단위로 처리하거나 디스크 기반 처리 고려

    Example (안전한 사용):
    >>> # 고유값 개수만 확인 (메모리 안전)
    >>> for group, cols in cols_group.items():
    ...     n_unique = lf.select(cols).unpivot(on=cols).select('value').n_unique().collect().item()
    ...     print(f"{group}: {n_unique}개")
    ...     if n_unique > 100000:  # 임계값 체크
    ...         print(f"⚠️ {group}은 고유값이 너무 많아 건너뜁니다")
    """
    unique_by_cols = {}

    # 각 그룹별로 순차적으로 고유값 추출
    for group, cols in cols_group.items():
        unique_by_cols[group] = get_unique(lf, cols)

    return unique_by_cols


def get_unique_by_cols_safe(
    lf: pl.LazyFrame,
    cols_group: Dict[str, List[str]],
    max_unique: int = None,
    memory_safety_ratio: float = 0.1,
    check_first: bool = True,
    estimate_string_size: bool = True,
    sample_size: int = 10000,
    size_metric: str = 'p75',
    calibration_factor: float = 1.0
) -> Dict[str, set]:
    """메모리 안전하게 컬럼 그룹별 고유값 추출 (자동 임계값 계산)

    시스템의 사용 가능한 메모리와 실제 데이터의 문자열 길이 통계를 기반으로
    안전한 최대 고유값 개수를 자동 계산합니다.

    Parameters:
    -----------
    lf : pl.LazyFrame
        데이터를 추출할 LazyFrame
    cols_group : Dict[str, List[str]]
        그룹명: 컬럼 리스트 딕셔너리
        예: {'devices': ['device_0_name', 'device_1_name']}
    max_unique : int, optional
        수동으로 지정할 최대 고유값 개수. None이면 자동 계산. Defaults to None.
    memory_safety_ratio : float, default=0.1
        사용 가능한 메모리의 몇 %까지 사용할지 (0.1 = 10%)
        보수적으로 설정 권장 (0.05 ~ 0.15)
    check_first : bool, default=True
        추출 전에 고유값 개수를 먼저 체크할지 여부
    estimate_string_size : bool, default=True
        실제 데이터로부터 문자열 길이를 추정할지 여부
        True 권장 (더 정확한 예측)
    sample_size : int, default=10000
        문자열 길이 추정 시 샘플링할 행 수
    size_metric : str, default='p75'
        사용할 크기 측정 기준
        - 'mean': 평균 (극단값에 민감, 비추천)
        - 'median': 중앙값 (50th percentile, 안정적)
        - 'p75': 75th percentile (권장, 대부분 커버하면서 안전)
        - 'p90': 90th percentile (매우 보수적)
    calibration_factor : float, default=1.0
        메모리 추정 보정 계수
        - 1.0보다 크면 더 보수적 (메모리를 더 많이 예상)
        - 1.0보다 작으면 덜 보수적
        함수 실행 후 피드백을 보고 조정 가능

    Returns:
    --------
    Dict[str, set or None]
        안전하게 추출된 고유값 딕셔너리
        - 성공: set of unique values
        - 실패 또는 스킵: None

    Examples:
    ---------
    >>> # 기본 설정 (75th percentile 사용)
    >>> cols_group = {
    ...     'devices': ['device_0_name', 'device_1_name'],
    ...     'manufacturers': ['device_0_manufacturer', 'device_1_manufacturer']
    ... }
    >>> unique_values = get_unique_by_cols_safe(lf, cols_group)

    >>> # 중앙값 사용 (덜 보수적)
    >>> unique_values = get_unique_by_cols_safe(
    ...     lf,
    ...     cols_group,
    ...     size_metric='median',
    ...     memory_safety_ratio=0.15
    ... )

    >>> # 90th percentile 사용 (더 보수적)
    >>> unique_values = get_unique_by_cols_safe(
    ...     lf,
    ...     cols_group,
    ...     size_metric='p90',
    ...     memory_safety_ratio=0.05
    ... )

    >>> # 보정 계수 적용 (이전 실행에서 피드백 받은 경우)
    >>> unique_values = get_unique_by_cols_safe(
    ...     lf,
    ...     cols_group,
    ...     calibration_factor=0.8  # 메모리를 과대평가했다면
    ... )

    Notes:
    ------
    - **권장 설정**: size_metric='p75', memory_safety_ratio=0.1 (기본값)
    - 평균(mean)은 극단값에 민감하므로 비추천
    - 중앙값(median)은 안정적이지만 큰 값들을 과소평가할 수 있음
    - calibration_factor는 실행 후 피드백을 보고 조정
    - 실제 메모리 사용량은 데이터 특성에 따라 다를 수 있음
    """
    # max_unique가 지정되지 않았으면 자동 계산
    if max_unique is None:
        # 1. 사용 가능한 메모리 (bytes) 확인
        available_memory = psutil.virtual_memory().available

        # 2. 안전 마진을 고려한 사용 가능 메모리
        safe_memory = available_memory * memory_safety_ratio

        # 3. 문자열 길이 통계 추정
        if estimate_string_size:
            print("문자열 길이 통계를 실제 데이터로부터 추정 중...")
            # 모든 컬럼 수집
            all_cols = [col for cols in cols_group.values() for col in cols]
            # 통계 계산
            stats = estimate_string_size_stats(lf, all_cols, sample_size)

            # 선택된 메트릭 사용
            avg_string_size = stats[size_metric]

            # 통계 출력
            print(f"  문자열 길이 통계:")
            print(f"    - 평균(mean): {stats['mean']:.1f}자")
            print(f"    - 중앙값(median): {stats['median']:.1f}자")
            print(f"    - 75%ile: {stats['p75']:.1f}자")
            print(f"    - 90%ile: {stats['p90']:.1f}자")
            print(f"    - 최댓값: {stats['max']:.0f}자")
            print(f"  → 사용할 크기({size_metric}): {avg_string_size:.1f}자\n")
        else:
            # 추정하지 않으면 기본값 사용
            avg_string_size = 50  # 기본값: 50자

        # 4. Python string의 실제 메모리 사용량 계산
        # Python 3.3+에서:
        # - ASCII는 1byte/char
        # - Unicode는 2-4bytes/char
        # - 평균적으로 2bytes/char로 가정 (영문+숫자 혼합)
        bytes_per_char = 2

        # 5. Python 객체 오버헤드
        str_overhead = 50   # str 객체 헤더: ~50 bytes
        set_overhead = 28   # set entry 오버헤드: ~28 bytes (hash table)

        # 6. 총 예상 바이트 수 계산
        estimated_bytes_per_unique = (
            (avg_string_size * bytes_per_char) +  # 실제 문자열 데이터
            str_overhead +                         # str 객체 오버헤드
            set_overhead                           # set 자료구조 오버헤드
        ) * calibration_factor  # 보정 계수 적용

        # 7. 안전한 최대 고유값 개수 계산
        max_unique = int(safe_memory / estimated_bytes_per_unique)

        # 8. 계산 결과 출력
        print(f"=== 메모리 기반 자동 임계값 계산 ===")
        print(f"사용 가능한 메모리: {available_memory / (1024**3):.2f} GB")
        print(f"안전 사용 메모리 ({memory_safety_ratio*100:.0f}%): {safe_memory / (1024**3):.2f} GB")
        print(f"예상 바이트/고유값: {estimated_bytes_per_unique:.0f} bytes")
        print(f"  - 문자열 데이터: {avg_string_size * bytes_per_char:.0f} bytes")
        print(f"  - str 오버헤드: {str_overhead} bytes")
        print(f"  - set 오버헤드: {set_overhead} bytes")
        if calibration_factor != 1.0:
            print(f"  - 보정 계수: {calibration_factor}x")
        print(f"계산된 최대 고유값: {max_unique:,}개")
        print(f"{'='*40}\n")
    else:
        # 수동 지정된 경우 기본값으로 예상 메모리 계산
        avg_string_size = 50
        bytes_per_char = 2
        str_overhead = 50
        set_overhead = 28
        estimated_bytes_per_unique = (
            (avg_string_size * bytes_per_char) + str_overhead + set_overhead
        ) * calibration_factor

    # 결과를 저장할 딕셔너리
    unique_by_cols = {}

    # 각 그룹별로 처리
    for group, cols in tqdm(cols_group.items(), desc="Extracting unique values"):
        # 사전 체크: 고유값 개수 확인
        if check_first:
            # 고유값 개수 계산 (메모리 효율적)
            n_unique = (
                lf.select(cols)
                .unpivot(on=cols)  # 모든 컬럼을 하나로 합치기
                .select(pl.col('value').n_unique())  # 고유값 개수만 계산
                .collect(engine='streaming')  # streaming 엔진 사용
                .item()  # 단일 값 추출
            )

            # 예상 메모리 계산
            estimated_mem_mb = (n_unique * estimated_bytes_per_unique) / (1024**2)
            print(f"{group}: {n_unique:,}개의 고유값 (예상 메모리: {estimated_mem_mb:.1f} MB)")

            # 임계값 초과 체크
            if n_unique > max_unique:
                print(f"  ⚠️ {group}은 고유값이 {max_unique:,}개를 초과하여 건너뜁니다\n")
                unique_by_cols[group] = None
                continue

        # 안전하면 실제 추출 시도
        try:
            # get_unique 함수로 고유값 추출
            unique_by_cols[group] = get_unique(lf, cols)

            # 실제 메모리 사용량 측정 (더 정확한 방법)
            # 각 문자열의 크기 합계
            actual_bytes = sum(sys.getsizeof(s) for s in unique_by_cols[group])
            # set 객체 자체의 크기 추가
            actual_bytes += sys.getsizeof(unique_by_cols[group])
            actual_mb = actual_bytes / (1024**2)

            # 예측 정확도 계산
            accuracy_pct = (estimated_mem_mb / actual_mb) * 100 if actual_mb > 0 else 0

            # 결과 출력
            print(f"  ✓ {group} 추출 완료 (실제 메모리: {actual_mb:.2f} MB)")
            print(f"    → 예상치 정확도: {accuracy_pct:.1f}% (예상/실제 비율)")

            # 보정 계수 피드백
            # 예측값이 실제값의 150% 이상이면 (과대평가)
            if accuracy_pct > 150:
                suggested_factor = calibration_factor * 0.8
                print(f"    💡 Tip: calibration_factor를 {suggested_factor:.2f}로 낮추면 더 정확할 것 같습니다")
            # 예측값이 실제값의 80% 미만이면 (과소평가)
            elif accuracy_pct < 80:
                suggested_factor = calibration_factor * 1.2
                print(f"    💡 Tip: calibration_factor를 {suggested_factor:.2f}로 높이면 더 정확할 것 같습니다")
            print()

        except MemoryError:
            # 메모리 부족으로 실패
            print(f"  ❌ {group} 메모리 부족으로 실패\n")
            unique_by_cols[group] = None

    # 2. 결과 요약
    print("\n=== 추출 요약 ===")
    success = sum(1 for v in unique_by_cols.values() if v is not None)
    print(f"성공: {success}/{len(cols_group)}")
    print(f"실패/스킵: {len(cols_group) - success}/{len(cols_group)}")
    return unique_by_cols


def groupby_nunique_safe(
    lf: pl.LazyFrame,
    group_cols: List[str],
    agg_cols: List[str] = None,
    top_n: int = 100,
    streaming: bool = True
) -> pl.DataFrame:
    """메모리 효율적으로 group by 후 각 그룹의 행 개수와 unique 개수 계산

    대용량 데이터에서 메모리 오버플로우 없이 그룹별 집계를 수행
    streaming 옵션으로 메모리 사용량 최소화

    Parameters:
    -----------
    lf : pl.LazyFrame
        분석할 LazyFrame
    group_cols : List[str]
        그룹화할 컬럼 리스트
    agg_cols : List[str], optional
        unique 개수를 셀 컬럼 리스트. None이면 count만 계산. Defaults to None.
    top_n : int, optional
        상위 몇 개 그룹만 반환할지. Defaults to 100.
    streaming : bool, optional
        streaming 엔진 사용 여부 (메모리 효율성 향상). Defaults to True.

    Returns:
    --------
    pl.DataFrame: 그룹별 집계 결과 DataFrame
        - group_cols: 그룹화 컬럼들
        - count: 각 그룹의 행 개수
        - {col}_unique: 각 컬럼의 고유값 개수 (agg_cols 지정 시)

    Examples:
    ---------
    >>> # 단순 카운트만
    >>> result = groupby_nunique_safe(
    ...     lf,
    ...     group_cols=['device_model', 'brand_name'],
    ...     top_n=50
    ... )

    >>> # unique 개수도 함께 계산
    >>> result = groupby_nunique_safe(
    ...     lf,
    ...     group_cols=['device_model', 'brand_name'],
    ...     agg_cols=['report_id', 'event_type'],
    ...     top_n=100,
    ...     streaming=True
    ... )
    """
    # 집계 표현식 구성
    if agg_cols is None:
        # count만 계산
        agg_exprs = [pl.len().alias('count')]
    else:
        # count + 각 컬럼의 unique 개수 계산
        agg_exprs = [
            pl.len().alias('count')
        ] + [
            pl.col(col).n_unique().alias(f'{col}_unique')
            for col in agg_cols
        ]

    # streaming 여부에 따라 엔진 선택
    engine = 'streaming' if streaming else 'auto'

    # group by 후 집계, 정렬, 상위 N개만 반환
    return (
        lf.group_by(group_cols)
        .agg(agg_exprs)
        .sort('count', descending=True)  # count 기준 내림차순
        .head(top_n)  # 상위 N개만
        .collect(engine=engine)  # 지정된 엔진으로 실행
        .to_pandas()  # pandas DataFrame으로 변환
    )

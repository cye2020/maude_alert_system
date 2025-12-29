"""도메인 특화 EDA 함수들

MAUDE 데이터 분석을 위한 전문화된 탐색적 데이터 분석 함수들
"""

import polars as pl
from typing import Optional


def remove_na_values(lf: pl.LazyFrame, dedup_cols, na_patterns, verbose=True):
    """
    Na / Unknown 값이 있는 행을 제거하는 함수

    작동방식:
    1. 각 컬럼에 대해 유효한 값인지 체크
    2. 모든 조건을 포함한(모두 만족하는) 행으로 필터링
    3. 필터 적용

    Parameters:
    -----------
    lf : polars.LazyFrame
        원본 LazyFrame
    dedup_cols : list
        모두 유효해야 하는 컬럼 리스트
    na_patterns : str
        NA / Unknown 패턴 정규식
    verbose : bool
        현재 진행상황 출력 여부

    Returns:
    --------
    polars.LazyFrame
        비어진 값이나 모르는 값이 없는 LazyFrame
    """

    # 진행상황 확인
    if verbose:
        print("NA 값 제거")
        print(f"패턴: {na_patterns}")

    # 제거 전 개수 확인
    before_cnt = lf.select(pl.len()).collect().item()
    if verbose:
        print(f"제거 전 행 개수: {before_cnt:,}개")

    # 각 컬럼별로 필터 조건
    conditions = []

    for col in dedup_cols:
        # 컬럼이 존재하는지 확인
        if col in lf.collect_schema().names():
            # 유효한 값의 조건
            # null / na_patterns 패턴에 매칭되지 않는 값
            cond = (
                pl.col(col).is_not_null()
                &
                ~ pl.col(col).cast(pl.Utf8).str.to_uppercase().str.contains(na_patterns)
                # ~ 은 not 연산자
            )
            conditions.append(cond)

            if verbose:
                print(f"컬럼 '{col}'에 대해 NA/Unknown 값 제거 조건 추가")
        else:
            if verbose:
                print(f"컬럼 '{col}'이(가) 존재하지 않음. 건너뜀")

    # 예외처리
    if not conditions:
        if verbose:
            print("제거할 조건이 없음. 원본 LazyFrame 반환")
        return lf

    # 모든 조건을 and 조건으로 결합
    # (모든 조건을 만족해야 함)
    final_condition = conditions[0]
    for cond in conditions[1:]:
        final_condition = final_condition & cond

    # 필터 적용
    lf_cleaned = lf.filter(final_condition)

    # 결과
    after_cnt = lf_cleaned.select(pl.len()).collect().item()
    removed_cnt = before_cnt - after_cnt

    if verbose:
        print(f"제거 후 행 개수: {after_cnt:,}개")
        print(f"제거된 행 개수: {removed_cnt:,}개")

    return lf_cleaned


def analyze_duplicates(lf, group_cols, verbose=True):
    """
    중복 데이터 분석 함수

    작동 방식:
    1. 전체 개수 확인
    2. 고유(unique) 개수 확인
    3. 전체 - 고유 = 중복 개수 확인

    Parameters:
    -----------
    lf : polars.LazyFrame
        원본 LazyFrame
    group_cols : list
        중복 확인 컬럼
    verbose : bool
        진행상황 출력 여부

    Returns:
    --------
    tuple
        (전체 개수, 고유 개수, 중복 개수)
    """

    if verbose:
        print("중복 확인")

    # 전체 개수
    total_cnt = lf.select(pl.len()).collect().item()

    # 고유 개수
    unique_cnt = lf.unique(
        subset=group_cols,
        maintain_order=True
    ).select(pl.len()).collect().item()

    # 중복 개수
    duplicate_cnt = total_cnt - unique_cnt

    if verbose:
        print(f"전체 개수: {total_cnt:,}개")
        print(f"고유 개수: {unique_cnt:,}개")
        print(f"중복 개수: {duplicate_cnt:,}개")
        for i, col in enumerate(group_cols, start=1):
            print(f"{i}. {col}")

    return total_cnt, unique_cnt, duplicate_cnt


def remove_duplicates(lf, dedup_cols, keep='first', verbose=True):
    """
    중복 데이터 제거 함수

    작동 방식:
    1. dedup_cols 기준으로 중복 판단
    2. keep 옵션에 따라 첫번째/마지막 행 유지
    3. 중복 제거된 LazyFrame 반환

    Parameters:
    -----------
    lf : polars.LazyFrame
        원본 LazyFrame
    dedup_cols : list
        중복 판단 컬럼 리스트
    keep : str
        'first' or 'last'
        'first': 첫번째 행 유지
        'last': 마지막 행 유지
    verbose : bool
        진행상황 출력 여부

    Returns:
    --------
    polars.LazyFrame
        중복 제거된 LazyFrame
    """

    if verbose:
        print("중복 제거 시작")
        print(f"중복 판단 컬럼: {dedup_cols}")
        print(f"유지 옵션: {keep}")

    # 중복 제거
    lf_deduped = lf.unique(
        subset=dedup_cols,
        maintain_order=True,
        keep=keep
    )

    if verbose:
        before_cnt = lf.select(pl.len()).collect().item()
        after_cnt = lf_deduped.select(pl.len()).collect().item()
        removed_cnt = before_cnt - after_cnt

        print(f"제거 전 행 개수: {before_cnt:,}개")
        print(f"제거 후 행 개수: {after_cnt:,}개")
        print(f"제거된 행 개수: {removed_cnt:,}개")

    return lf_deduped


def analyze_column_distribution(lf, column_name, top_n=None, show_null=True, verbose=True):
    """
    특정 컬럼의 값 분포와 비율을 분석하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    column_name : str
        분석할 컬럼명
    top_n : int, optional
        상위 N개만 표시 (None이면 전체 표시)
    show_null : bool, default=True
        NULL 값 개수를 별도로 표시할지 여부
    verbose : bool, default=True
        결과를 출력할지 여부

    Returns:
    --------
    polars.DataFrame
        분포 결과 (컬럼명, count, percentage 포함)

    Examples:
    ---------
    >>> # 기본 사용
    >>> result = analyze_column_distribution(lf_class3, 'patient_harm')

    >>> # 상위 10개만
    >>> result = analyze_column_distribution(lf_class3, 'device_0_generic_name', top_n=10)

    >>> # 출력 없이 결과만
    >>> result = analyze_column_distribution(lf_class3, 'patient_harm', verbose=False)
    """
    # 전체 개수
    total = lf.select(pl.len()).collect().item()

    # 분포 계산
    dist = lf.group_by(column_name).agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True)

    # 상위 N개만
    if top_n is not None:
        dist = dist.head(top_n)

    result = dist.collect()

    # NULL 개수 확인
    null_count = lf.filter(pl.col(column_name).is_null()).select(pl.len()).collect().item()

    # 출력
    if verbose:
        print("=" * 80)
        print(f"{column_name} 분포 (전체: {total:,}건)")
        if top_n:
            print(f"(상위 {top_n}개만 표시)")
        print("=" * 80)
        print(result)

        if show_null:
            print(f"\nNULL 값: {null_count:,}개 ({null_count/total*100:.2f}%)")

    return result


def get_top_n_by_column(lf, group_column, top_n=10, column_display_name=None, verbose=True):
    """
    특정 컬럼을 기준으로 상위 N개 항목을 추출하고 비율과 함께 출력하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    group_column : str
        그룹화할 컬럼명 (예: 'device_0_generic_name')
    top_n : int, default=10
        상위 N개 항목 추출
    column_display_name : str, optional
        출력 시 표시할 컬럼 이름 (None이면 group_column 사용)
    verbose : bool, default=True
        결과를 출력할지 여부

    Returns:
    --------
    polars.DataFrame
        상위 N개 항목 (컬럼명, count, percentage 포함)

    Examples:
    ---------
    >>> # 기본 사용 - 상위 20개 기기
    >>> top_devices = get_top_n_by_column(lf_class3, 'device_0_generic_name', top_n=20)

    >>> # 제조사 상위 10개
    >>> top_manufacturers = get_top_n_by_column(
    ...     lf_class3,
    ...     'device_0_manufacturer_d_name',
    ...     top_n=10,
    ...     column_display_name='제조사'
    ... )

    >>> # 출력 없이 결과만
    >>> result = get_top_n_by_column(lf_class3, 'device_0_generic_name', top_n=5, verbose=False)
    """
    # 전체 개수
    total = lf.select(pl.len()).collect().item()

    # 상위 N개 추출
    top_items = lf.group_by(group_column).agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).head(top_n).collect()

    # 출력
    if verbose:
        display_name = column_display_name if column_display_name else group_column

        print("=" * 90)
        print(f"{display_name} 사건 수 Top {top_n} (전체: {total:,}건)")
        print("=" * 90)
        print()

        for i, row in enumerate(top_items.iter_rows(named=True), 1):
            value = row[group_column] if row[group_column] else "(NULL)"
            count = row['count']
            pct = row['percentage']

            # 긴 값은 70자로 자르기
            display_value = value[:70] if len(value) > 70 else value

            print(f"{i:2d}. {display_value:70s} {count:>8,}건 ({pct:>5.1f}%)")

    return top_items


def analyze_top_n_by_category(lf, group_column, category_column, top_n=3,
                               group_display_name=None, category_display_name=None,
                               verbose=True):
    """
    상위 N개 항목에 대해 카테고리별 분포를 분석하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    group_column : str
        그룹화할 주요 컬럼 (예: 'device_0_generic_name')
    category_column : str
        분포를 확인할 카테고리 컬럼 (예: 'patient_harm')
    top_n : int, default=3
        상위 N개 항목 분석
    group_display_name : str, optional
        그룹 컬럼의 표시 이름 (None이면 group_column 사용)
    category_display_name : str, optional
        카테고리 컬럼의 표시 이름 (None이면 category_column 사용)
    verbose : bool, default=True
        결과를 출력할지 여부

    Returns:
    --------
    dict
        각 상위 항목별 카테고리 분포 딕셔너리

    Examples:
    ---------
    >>> # Top 3 기기별 patient harm 분포
    >>> result = analyze_top_n_by_category(
    ...     lf_class3,
    ...     'device_0_generic_name',
    ...     'patient_harm',
    ...     top_n=3
    ... )

    >>> # Top 5 제조사별 patient harm 분포
    >>> result = analyze_top_n_by_category(
    ...     lf_class3,
    ...     'device_0_manufacturer_d_name',
    ...     'patient_harm',
    ...     top_n=5,
    ...     group_display_name='제조사',
    ...     category_display_name='사건 유형'
    ... )
    """
    # 상위 N개 항목 추출
    top_items = lf.group_by(group_column).len()\
        .sort('len', descending=True).head(top_n).collect()

    # 결과 저장용 딕셔너리
    results = {}

    # 표시 이름 설정
    group_name = group_display_name if group_display_name else group_column
    category_name = category_display_name if category_display_name else category_column

    if verbose:
        print("=" * 90)
        print(f"Top {top_n} {group_name}별 {category_name} 분포")
        print("=" * 90)

    # 각 항목에 대해 카테고리 분포 계산
    for rank, row in enumerate(top_items.iter_rows(named=True), 1):
        item_value = row[group_column]
        total_count = row['len']

        # 해당 항목의 카테고리 분포
        dist = lf.filter(
            pl.col(group_column) == item_value
        ).group_by(category_column).agg([
            pl.len().alias('count')
        ]).with_columns([
            (pl.col('count') / total_count * 100).round(2).alias('percentage')
        ]).sort('count', descending=True).collect()

        # 결과 저장
        results[item_value] = dist

        # 출력
        if verbose:
            print(f"\n[{rank}위] {item_value}")
            print(f"총 사건 수: {total_count:,}건")
            print("-" * 90)
            print(f"{category_name:<20} {'건수':>15} {'비율':>15}")
            print("-" * 90)

            for cat_row in dist.iter_rows(named=True):
                category = cat_row[category_column] if cat_row[category_column] else "(NULL)"
                count = cat_row['count']
                pct = cat_row['percentage']
                print(f"{category:<20} {count:>15,} {pct:>14.1f}%")

    if verbose:
        print("=" * 90)

    return results


def analyze_single_column_pretty(lf, column_name, column_display_name=None):
    """
    단일 컬럼의 분포를 예쁘게 출력하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    column_name : str
        분석할 컬럼명
    column_display_name : str, optional
        출력 시 표시할 컬럼 이름 (None이면 column_name 사용)

    Returns:
    --------
    polars.DataFrame
        분포 결과 (컬럼값, count, percentage 포함)

    Examples:
    ---------
    >>> # Reprocessed Flag 분포
    >>> analyze_single_column_pretty(lf_class3, 'reprocessed_and_reused_flag')

    >>> # patient harm 분포
    >>> analyze_single_column_pretty(
    ...     lf_class3,
    ...     'patient_harm',
    ...     column_display_name='사건 유형'
    ... )
    """
    # 전체 개수
    total = lf.select(pl.len()).collect().item()

    # 분포 계산
    dist = lf.group_by(column_name).agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).collect()

    # 표시 이름
    display_name = column_display_name if column_display_name else column_name

    # 출력
    print("=" * 70)
    print(f"{display_name} 분포 (전체: {total:,}건)")
    print("=" * 70)
    print(f"{'값':<20} {'사건 수':>15} {'비율':>15}")
    print("-" * 70)

    for row in dist.iter_rows(named=True):
        value = row[column_name] if row[column_name] else "(NULL)"
        count = row['count']
        pct = row['percentage']

        # 긴 값은 자르기
        display_value = value[:18] if len(str(value)) > 18 else value
        print(f"{display_value:<20} {count:>15,} {pct:>14.2f}%")

    print("=" * 70)

    return dist


def analyze_crosstab(lf, column1, column2,
                     column1_display_name=None,
                     column2_display_name=None,
                     show_detail=True):
    """
    두 컬럼의 크로스탭(교차 분석)을 수행하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    column1 : str
        첫 번째 컬럼 (예: 'reprocessed_and_reused_flag')
    column2 : str
        두 번째 컬럼 (예: 'patient_harm')
    column1_display_name : str, optional
        첫 번째 컬럼의 표시 이름
    column2_display_name : str, optional
        두 번째 컬럼의 표시 이름
    show_detail : bool, default=True
        각 column1 값별로 상세 분포를 출력할지 여부

    Returns:
    --------
    polars.DataFrame
        크로스탭 결과

    Examples:
    ---------
    >>> # Reuse Flag × patient harm
    >>> analyze_crosstab(
    ...     lf_class3,
    ...     'reprocessed_and_reused_flag',
    ...     'patient_harm'
    ... )
    """
    # 표시 이름 설정
    col1_name = column1_display_name if column1_display_name else column1
    col2_name = column2_display_name if column2_display_name else column2

    # 크로스탭 생성
    crosstab = lf.group_by([column1, column2]).agg([
        pl.len().alias('count')
    ]).sort([column1, 'count'], descending=[False, True]).collect()

    print("=" * 90)
    print(f"{col1_name} × {col2_name} 크로스탭")
    print("=" * 90)
    print(crosstab)

    # 상세 분포 출력
    if show_detail:
        print("\n" + "=" * 90)
        print(f"{col1_name}별 {col2_name} 상세 분포")
        print("=" * 90)

        # column1의 고유값 추출 (NULL 포함)
        unique_values = lf.select(column1).unique().collect()[column1].to_list()

        # NULL도 포함
        if None not in unique_values and lf.filter(pl.col(column1).is_null()).select(pl.len()).collect().item() > 0:
            unique_values.append(None)

        for value in unique_values:
            value_display = value if value else "(NULL)"

            # 해당 값의 총 개수
            value_total = lf.filter(
                pl.col(column1) == value if value else pl.col(column1).is_null()
            ).select(pl.len()).collect().item()

            if value_total == 0:
                continue

            print(f"\n[{value_display}] 총 {value_total:,}건")
            print("-" * 90)

            # 해당 값의 column2 분포
            dist = lf.filter(
                pl.col(column1) == value if value else pl.col(column1).is_null()
            ).group_by(column2).agg([
                pl.len().alias('count')
            ]).with_columns([
                (pl.col('count') / value_total * 100).round(2).alias('percentage')
            ]).sort('count', descending=True).collect()

            print(f"{col2_name:<25} {'건수':>15} {'비율':>15}")
            print("-" * 90)

            for row in dist.iter_rows(named=True):
                cat = row[column2] if row[column2] else "(NULL)"
                count = row['count']
                pct = row['percentage']

                cat_display = cat[:23] if len(str(cat)) > 23 else cat
                print(f"{cat_display:<25} {count:>15,} {pct:>14.2f}%")

        print("=" * 90)

    return crosstab


def calculate_cfr_by_device(lf, device_column='device_0_generic_name',
                            harm_column='patient_harm',
                            top_n=None, min_cases=10):
    """
    기기별 치명률(Case Fatality Rate)을 계산하는 함수

    치명률(CFR) = (사망 건수 / 해당 기기 총 보고 건수) × 100

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    device_column : str, default='device_0_generic_name'
        기기 컬럼명
    harm_column : str, default='patient_harm'
        사건 유형 컬럼명
    top_n : int, optional
        상위 N개 기기만 분석 (None이면 전체)
    min_cases : int, default=10
        최소 보고 건수 (이보다 적은 기기는 제외, 통계적 신뢰도 확보)

    Returns:
    --------
    polars.DataFrame
        기기별 치명률 결과 (기기명, 총건수, 사망건수, 부상건수, 오작동건수, CFR)

    Examples:
    ---------
    >>> # 전체 기기 CFR
    >>> cfr_result = calculate_cfr_by_device(lf_class3)

    >>> # 상위 20개 기기만
    >>> cfr_top20 = calculate_cfr_by_device(lf_class3, top_n=20)

    >>> # 최소 100건 이상 보고된 기기만
    >>> cfr_reliable = calculate_cfr_by_device(lf_class3, min_cases=100)
    """

    # 기기별 전체 건수와 사건 유형별 건수
    device_stats = lf.group_by(device_column).agg([
        pl.len().alias('total_cases'),
        (pl.col(harm_column) == 'Death').sum().alias('death_count'),
        (pl.col(harm_column) == 'Serious Injury').sum().alias('serious_injury_count'),
        (pl.col(harm_column) == 'Minor Injury').sum().alias('minor_injury_count'),
        (pl.col(harm_column) == 'No Harm').sum().alias('no_harm_count')
    ]).filter(
        pl.col('total_cases') >= min_cases  # 최소 건수 필터
    ).with_columns([
        # CFR 계산
        (pl.col('death_count').add(pl.col('serious_injury_count')) / pl.col('total_cases') * 100).round(2).alias('cfr'),
    ]).sort('cfr', descending=True)

    # Top N만
    if top_n:
        device_stats = device_stats.head(top_n)

    result = device_stats.collect()

    # 출력
    print("=" * 120)
    print(f"기기별 치명률(CFR) 분석 (최소 {min_cases}건 이상)")
    print("=" * 120)
    print(f"{'순위':>4} {'기기명':<50} {'총건수':>10}{'CFR(%)':>10}")
    print("-" * 120)

    for i, row in enumerate(result.iter_rows(named=True), 1):
        device = row[device_column] if row[device_column] else "(NULL)"
        total = row['total_cases']
        death = row['death_count']
        serious_injury = row['serious_injury_count']
        minor_injury = row['minor_injury_count']
        no_harm = row['no_harm_count']
        cfr = row['cfr']

        device_short = device[:48] if len(device) > 48 else device

        print(f"{i:4d} {device_short:<50} {total:>10,} {death:>8,} {serious_injury:>8,} {minor_injury:>8,} {no_harm:>10,} {cfr:>10.2f}%")

    print("=" * 120)

    # 요약 통계
    print(f"\n◼️ 요약 통계:")
    print(f"  - 분석 기기 수: {len(result):,}개")
    print(f"  - 평균 CFR: {result['cfr'].mean():.2f}%")
    print(f"  - 최대 CFR: {result['cfr'].max():.2f}%")
    print(f"  - 최소 CFR: {result['cfr'].min():.2f}%")
    print(f"  - CFR 중앙값: {result['cfr'].median():.2f}%")

    return result


def analyze_cfr_distribution(lf, device_column='device_0_generic_name', min_cases=10):
    """
    CFR 구간별 기기 분포를 분석하는 함수

    치명률(CFR) = (사망 건수 + 심각한 부상 건수) / 해당 기기 총 보고 건수 × 100

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    device_column : str
        기기 컬럼명
    min_cases : int
        최소 보고 건수

    Examples:
    ---------
    >>> analyze_cfr_distribution(lf_class3)
    """
    # CFR 계산
    cfr_data = lf.group_by(device_column).agg([
        pl.len().alias('total_cases'),
        (pl.col('patient_harm') == 'Death').sum().alias('death_count'),
        (pl.col('patient_harm') == 'Serious Injury').sum().alias('serious_injury_count')
    ]).filter(
        pl.col('total_cases') >= min_cases
    ).with_columns([
        ((pl.col('death_count') + pl.col('serious_injury_count')) / pl.col('total_cases') * 100).alias('cfr')
    ]).collect()

    # CFR 구간별 분류
    cfr_ranges = [
        (0, 1, "매우 낮음 (0-1%)"),
        (1, 3, "낮음 (1-3%)"),
        (3, 5, "보통 (3-5%)"),
        (5, 10, "높음 (5-10%)"),
        (10, 100, "매우 높음 (10%+)")
    ]

    print("=" * 80)
    print("CFR 구간별 기기 분포")
    print("=" * 80)
    print(f"{'CFR 구간':<25} {'기기 수':>15} {'비율':>15}")
    print("-" * 80)

    total_devices = len(cfr_data)

    for min_cfr, max_cfr, label in cfr_ranges:
        count = cfr_data.filter(
            (pl.col('cfr') >= min_cfr) & (pl.col('cfr') < max_cfr)
        ).shape[0]

        pct = (count / total_devices * 100) if total_devices > 0 else 0
        print(f"{label:<25} {count:>15,} {pct:>14.1f}%")

    print("=" * 80)
    print(f"{'총 기기 수':<25} {total_devices:>15,} {100.0:>14.1f}%")
    print("=" * 80)


def analyze_device_detail(lf, device_name, device_column='device_0_generic_name'):
    """
    특정 기기의 상세 통계 분석

    치명률(CFR) = (사망 건수 + 심각한 부상 건수) / 해당 기기 총 보고 건수 × 100

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    device_name : str
        분석할 기기명
    device_column : str
        기기 컬럼명

    Examples:
    ---------
    >>> analyze_device_detail(lf_class3, 'Catheter, Intravascular, Therapeutic')
    """
    device_data = lf.filter(pl.col(device_column) == device_name).collect()

    total = len(device_data)
    death = device_data.filter(pl.col('patient_harm') == 'Death').shape[0]
    serious_injury = device_data.filter(pl.col('patient_harm') == 'Serious Injury').shape[0]
    minor_injury = device_data.filter(pl.col('patient_harm') == 'Minor Injury').shape[0]
    no_harm = device_data.filter(pl.col('patient_harm') == 'No Harm').shape[0]

    cfr = ((death + serious_injury) / total * 100) if total > 0 else 0
    serious_injury_rate = (serious_injury / total * 100) if total > 0 else 0
    minor_injury_rate = (minor_injury / total * 100) if total > 0 else 0
    no_harm_rate = (no_harm / total * 100) if total > 0 else 0

    print("=" * 80)
    print(f"기기 상세 분석: {device_name}")
    print("=" * 80)
    print(f"\n◼️ 기초 통계:")
    print(f"  총 보고 건수: {total:,}건")
    print(f"\n◼️ 사건 유형별 분포:")
    print(f"  • 사망 (Death):            {death:>8,}건 ({death/total*100:>5.2f}%)")
    print(f"  • 심각한 부상 (Serious Injury): {serious_injury:>8,}건 ({serious_injury_rate:>5.2f}%)")
    print(f"  • 경미한 부상 (Minor Injury):   {minor_injury:>8,}건 ({minor_injury_rate:>5.2f}%)")
    print(f"  • 피해 없음 (No Harm):        {no_harm:>8,}건 ({no_harm_rate:>5.2f}%)")
    print(f"\n◼️ 치명률(CFR):")
    print(f"  • CFR (사망+심각한부상): {cfr:>5.2f}% ⚠️")
    print("=" * 80)


def analyze_defect_types(lf, top_n=20, min_cases=5):
    """
    기기 결함 종류 분포를 분석하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    top_n : int, default=20
        상위 N개 결함 종류 표시
    min_cases : int, default=5
        최소 발생 건수 (이보다 적은 결함은 제외)

    Returns:
    --------
    polars.DataFrame
        결함 종류별 분포 결과

    Examples:
    ---------
    >>> # 전체 결함 종류 분포
    >>> defect_dist = analyze_defect_types(lf_class3)

    >>> # 상위 30개만
    >>> defect_top30 = analyze_defect_types(lf_class3, top_n=30)
    """
    # 전체 개수
    total = lf.select(pl.len()).collect().item()

    # 결함 종류 분포
    defect_dist = lf.group_by('defect_type').agg([
        pl.len().alias('count')
    ]).filter(
        pl.col('count') >= min_cases
    ).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).head(top_n).collect()

    # 결함 확인 여부
    defect_confirmed = lf.group_by('defect_confirmed').len().collect()

    print("=" * 100)
    print(f"기기 결함 종류 분석 (전체: {total:,}건)")
    print("=" * 100)

    # 결함 확인 여부 먼저 출력
    print("\n◼️ 결함 확인 여부:")
    print("-" * 100)
    for row in defect_confirmed.iter_rows(named=True):
        confirmed = row['defect_confirmed']
        count = row['len']
        pct = (count / total * 100)
        confirmed_display = confirmed if confirmed else "(NULL)"
        print(f"  {confirmed_display:<20} {count:>10,}건 ({pct:>5.2f}%)")

    # 결함 종류 분포
    print(f"\n◼️ 결함 종류 Top {top_n} (최소 {min_cases}건 이상):")
    print("-" * 100)
    print(f"{'순위':>4} {'결함 종류':<60} {'건수':>12} {'비율':>12}")
    print("-" * 100)

    for i, row in enumerate(defect_dist.iter_rows(named=True), 1):
        defect_type = row['defect_type']
        count = row['count']
        pct = row['percentage']

        defect_display = defect_type[:58] if defect_type and len(defect_type) > 58 else (defect_type or "(NULL)")
        print(f"{i:4d} {defect_display:<60} {count:>12,} {pct:>11.2f}%")

    print("=" * 100)

    return defect_dist


def analyze_defect_impact(lf, top_defects=10):
    """
    주요 결함 종류별 환자 피해 및 사건 유형 분석

    치명률(CFR) = (사망 건수 + 심각한 부상 건수) / 해당 기기 총 보고 건수 × 100

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    top_defects : int, default=10
        분석할 상위 결함 종류 개수

    Returns:
    --------
    dict
        결함 종류별 분석 결과

    Examples:
    ---------
    >>> # 상위 10개 결함의 영향 분석
    >>> impact_result = analyze_defect_impact(lf_class3, top_defects=10)

    >>> # 상위 5개만
    >>> impact_top5 = analyze_defect_impact(lf_class3, top_defects=5)
    """
    # 상위 결함 종류 추출
    top_defects_list = lf.group_by('defect_type').len()\
        .sort('len', descending=True).head(top_defects).collect()

    results = {}

    print("=" * 110)
    print(f"상위 {top_defects}개 결함 종류별 영향 분석")
    print("=" * 110)

    for rank, row in enumerate(top_defects_list.iter_rows(named=True), 1):
        defect_type = row['defect_type']
        total_count = row['len']

        if not defect_type:
            defect_type = "(NULL)"

        # 해당 결함의 데이터 필터링
        defect_data = lf.filter(
            pl.col('defect_type') == defect_type if defect_type != "(NULL)"
            else pl.col('defect_type').is_null()
        )

        # patient harm 분포
        harm_dist = defect_data.group_by('patient_harm').agg([
            pl.len().alias('count')
        ]).with_columns([
            (pl.col('count') / total_count * 100).round(2).alias('percentage')
        ]).sort('count', descending=True).collect()

        # 환자 피해 여부 (NULL이 아닌 경우)
        harm_count = defect_data.filter(
            pl.col('patient_harm').is_not_null()
        ).select(pl.len()).collect().item()

        # CFR 계산
        death_count = defect_data.filter(pl.col('patient_harm') == 'Death').select(pl.len()).collect().item()
        serious_injury_count = defect_data.filter(pl.col('patient_harm') == 'Serious Injury').select(pl.len()).collect().item()
        cfr = ((death_count + serious_injury_count) / total_count * 100) if total_count > 0 else 0

        results[defect_type] = {
            'total': total_count,
            'harm_dist': harm_dist,
            'harm_count': harm_count,
            'cfr': cfr
        }

        # 출력
        print(f"\n[{rank}위] {defect_type}")
        print(f"총 {total_count:,}건 | 환자 피해 기록: {harm_count:,}건 | CFR: {cfr:.2f}%")
        print("-" * 110)
        print(f"{'사건 유형':<20} {'건수':>15} {'비율':>15}")
        print("-" * 110)

        for harm_row in harm_dist.iter_rows(named=True):
            harm = harm_row['patient_harm'] if harm_row['patient_harm'] else "(NULL)"
            count = harm_row['count']
            pct = harm_row['percentage']
            print(f"{harm:<20} {count:>15,} {pct:>14.2f}%")

    print("=" * 110)

    return results


def analyze_defect_components(lf, defect_type, top_n=10):
    """
    특정 결함 종류의 문제 기기 부품 분석

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임
    defect_type : str
        분석할 결함 종류
    top_n : int, default=10
        상위 N개 문제 부품 표시

    Returns:
    --------
    polars.DataFrame
        문제 부품 분포

    Examples:
    ---------
    >>> # 특정 결함의 문제 부품
    >>> analyze_defect_components(lf_class3, 'Software Failure', top_n=15)
    """
    # 해당 결함 필터링
    defect_data = lf.filter(
        pl.col('defect_type') == defect_type
    )

    total = defect_data.select(pl.len()).collect().item()

    # 문제 부품 분포 - list[str]을 explode하여 각 부품을 개별적으로 카운트
    component_dist = defect_data.filter(
        pl.col('problem_components').is_not_null()
    ).select(
        pl.col('problem_components').explode()
    ).group_by('problem_components').agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).head(top_n).collect()

    print("=" * 100)
    print(f"결함 종류: {defect_type}")
    print(f"문제 기기 부품 분석 (전체: {total:,}건)")
    print("=" * 100)
    print(f"{'순위':>4} {'문제 부품':<60} {'건수':>12} {'비율':>12}")
    print("-" * 100)

    for i, row in enumerate(component_dist.iter_rows(named=True), 1):
        component = row['problem_components']
        count = row['count']
        pct = row['percentage']

        component_display = component[:58] if component and len(component) > 58 else (component or "(NULL)")
        print(f"{i:4d} {component_display:<60} {count:>12,} {pct:>11.2f}%")

    print("=" * 100)

    return component_dist


def comprehensive_defect_analysis(lf):
    """
    결함에 대한 종합적인 분석을 수행하는 함수

    Parameters:
    -----------
    lf : polars.LazyFrame or polars.DataFrame
        분석할 데이터프레임

    Examples:
    ---------
    >>> comprehensive_defect_analysis(lf_class3)
    """
    total = lf.select(pl.len()).collect().item()

    print("=" * 120)
    print("◼️ Class 3 기기 결함 종합 분석 대시보드")
    print("=" * 120)

    # 1. 결함 확인 여부
    print("\n◼️ 1. 제조사 검사 - 결함 확인 여부")
    print("-" * 120)
    defect_confirmed = lf.group_by('defect_confirmed').agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).collect()

    for row in defect_confirmed.iter_rows(named=True):
        confirmed = row['defect_confirmed'] or "(NULL)"
        count = row['count']
        pct = row['percentage']
        print(f"  {confirmed:<30} {count:>15,}건 ({pct:>5.2f}%)")

    # 2. 결함 종류 Top 10
    print("\n◼️ 2. 결함 종류 Top 10")
    print("-" * 120)
    defect_types = lf.filter(
        pl.col('defect_type').is_not_null()
    ).group_by('defect_type').agg([
        pl.len().alias('count')
    ]).with_columns([
        (pl.col('count') / total * 100).round(2).alias('percentage')
    ]).sort('count', descending=True).head(10).collect()

    for i, row in enumerate(defect_types.iter_rows(named=True), 1):
        defect = row['defect_type'][:50]
        count = row['count']
        pct = row['percentage']
        print(f"  {i:2d}. {defect:<50} {count:>10,}건 ({pct:>5.2f}%)")

    # 3. 문제 부품 Top 10
    print("\n◼️ 3. 문제 기기 부품 Top 10")
    print("-" * 120)
    # list[str]을 explode하여 각 부품을 개별적으로 카운트
    components = lf.filter(
        pl.col('problem_components').is_not_null()
    ).select(
        pl.col('problem_components').explode()
    ).group_by('problem_components').agg([
        pl.len().alias('count')
    ]).sort('count', descending=True).head(10).collect()

    for i, row in enumerate(components.iter_rows(named=True), 1):
        component = row['problem_components']
        component_display = component[:50] if component and len(component) > 50 else component
        count = row['count']
        print(f"  {i:2d}. {component_display:<50} {count:>10,}건")

    # 4. 환자 피해 요약
    print("\n◼️  4. 환자 피해 기록")
    print("-" * 120)
    harm_count = lf.filter(
        pl.col('patient_harm').is_not_null()
    ).select(pl.len()).collect().item()
    harm_pct = (harm_count / total * 100)

    print(f"  환자 피해 기록 있음: {harm_count:>10,}건 ({harm_pct:>5.2f}%)")
    print(f"  환자 피해 기록 없음: {total - harm_count:>10,}건 ({100 - harm_pct:>5.2f}%)")

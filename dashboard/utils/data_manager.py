"""
데이터 로딩 및 캐싱 관리 모듈

이 모듈은 다음 기능을 제공합니다:
1. 데이터 파일 로딩 (Parquet, CSV)
2. 자주 사용하는 집계 결과 캐싱
3. 범용 데이터 처리 함수
4. 세션 상태 관리

Note:
- UI 필터링 관련 함수는 filter_helpers.py 참고
- 차트/분석 관련 함수는 analysis.py 참고
"""

import streamlit as st
import polars as pl
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import ast
from collections import Counter


# ==================== 세션 상태 관리 ====================

def get_shared_data() -> Dict[str, Any]:
    """세션 상태에 저장된 공유 데이터 반환"""
    if 'shared_data' not in st.session_state:
        st.session_state.shared_data = {}
    return st.session_state.shared_data


def set_shared_data(key: str, value: Any) -> None:
    """세션 상태에 데이터 저장"""
    shared_data = get_shared_data()
    shared_data[key] = value


# ==================== 데이터 로딩 ====================

@st.cache_data
def load_parquet(
    file_path: Union[str, Path],
    lazy: bool = True,
    _cache_key: Optional[str] = None
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    Parquet 파일 로드 (캐싱)

    Args:
        file_path: 파일 경로
        lazy: LazyFrame으로 로드할지 여부 (기본: True)
        _cache_key: 캐시 키 (월 변경 시 자동 갱신용, 예: "2025-12")

    Returns:
        LazyFrame 또는 DataFrame
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if lazy:
        return pl.scan_parquet(path)
    else:
        return pl.read_parquet(path)


@st.cache_data
def load_csv(
    file_path: Union[str, Path],
    lazy: bool = False,
    **kwargs
) -> Union[pl.LazyFrame, pl.DataFrame]:
    """
    CSV 파일 로드 (캐싱)

    Args:
        file_path: 파일 경로
        lazy: LazyFrame으로 로드할지 여부 (기본: False, CSV는 즉시 로드 권장)
        **kwargs: pl.read_csv 또는 pl.scan_csv에 전달할 추가 인자

    Returns:
        LazyFrame 또는 DataFrame
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    if lazy:
        return pl.scan_csv(path, **kwargs)
    else:
        return pl.read_csv(path, **kwargs)


# ==================== 공통 집계 함수 (캐싱) ====================

@st.cache_data
def get_monthly_aggregation(
    _lf: pl.LazyFrame,
    date_col: str = 'date_received',
    group_by_cols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pl.DataFrame:
    """
    월별 집계 (캐싱)

    Args:
        _lf: LazyFrame (언더스코어로 시작하여 캐싱에서 제외)
        date_col: 날짜 컬럼명
        group_by_cols: 그룹화할 컬럼 리스트 (None이면 날짜만)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)

    Returns:
        월별 집계 결과 DataFrame
    """
    # 날짜 필터링
    filtered_lf = _lf
    if start_date:
        filtered_lf = filtered_lf.filter(
            pl.col(date_col) >= pl.lit(start_date).str.strptime(pl.Date, "%Y-%m-%d")
        )
    if end_date:
        filtered_lf = filtered_lf.filter(
            pl.col(date_col) <= pl.lit(end_date).str.strptime(pl.Date, "%Y-%m-%d")
        )

    # 월 단위로 truncate
    group_cols = [pl.col(date_col).dt.truncate("1mo").alias("month")]

    if group_by_cols:
        group_cols.extend([pl.col(c) for c in group_by_cols])

    result = (
        filtered_lf
        .group_by(group_cols)
        .agg(pl.len().alias("count"))
        .sort("month")
        .collect()
    )

    return result


@st.cache_data
def get_top_n_by_column(
    _lf: pl.LazyFrame,
    column: str,
    top_n: int = 10,
    filters: Optional[Dict[str, Any]] = None
) -> pl.DataFrame:
    """
    특정 컬럼의 상위 N개 값 집계 (캐싱)

    Args:
        _lf: LazyFrame
        column: 집계할 컬럼명
        top_n: 상위 N개
        filters: 필터 딕셔너리 (예: {"manufacturer_name": ["A", "B"]})

    Returns:
        상위 N개 집계 결과 DataFrame
    """
    filtered_lf = _lf

    # 필터 적용
    if filters:
        for col, values in filters.items():
            if values and len(values) > 0:
                filtered_lf = filtered_lf.filter(pl.col(col).is_in(values))

    result = (
        filtered_lf
        .filter(pl.col(column).is_not_null())
        .group_by(column)
        .agg(pl.len().alias("count"))
        .sort("count", descending=True)
        .head(top_n)
        .collect()
    )

    return result


@st.cache_data
def calculate_statistics(
    _lf: pl.LazyFrame,
    numeric_cols: Optional[List[str]] = None,
    filters: Optional[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """
    숫자형 컬럼의 통계량 계산 (캐싱)

    Args:
        _lf: LazyFrame
        numeric_cols: 계산할 숫자형 컬럼 리스트 (None이면 모든 숫자형 컬럼)
        filters: 필터 딕셔너리

    Returns:
        {컬럼명: {mean, median, std, min, max}} 형태의 딕셔너리
    """
    filtered_lf = _lf

    # 필터 적용
    if filters:
        for col, values in filters.items():
            if values and len(values) > 0:
                filtered_lf = filtered_lf.filter(pl.col(col).is_in(values))

    # 숫자형 컬럼 자동 감지 (numeric_cols가 None인 경우)
    if numeric_cols is None:
        schema = filtered_lf.collect_schema()
        numeric_cols = [
            col for col, dtype in schema.items()
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]
        ]

    stats = {}
    for col in numeric_cols:
        result = filtered_lf.select([
            pl.col(col).mean().alias("mean"),
            pl.col(col).median().alias("median"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max")
        ]).collect()

        stats[col] = {
            "mean": result["mean"][0],
            "median": result["median"][0],
            "std": result["std"][0],
            "min": result["min"][0],
            "max": result["max"][0]
        }

    return stats


# ==================== 데이터 변환 헬퍼 함수 ====================

def apply_date_filter(
    lf: pl.LazyFrame,
    date_col: str,
    start_date: Optional[Any] = None,
    end_date: Optional[Any] = None
) -> pl.LazyFrame:
    """
    날짜 범위 필터 적용

    Args:
        lf: LazyFrame
        date_col: 날짜 컬럼명
        start_date: 시작 날짜
        end_date: 종료 날짜

    Returns:
        필터링된 LazyFrame
    """
    filtered_lf = lf

    if start_date:
        filtered_lf = filtered_lf.filter(pl.col(date_col) >= start_date)
    if end_date:
        filtered_lf = filtered_lf.filter(pl.col(date_col) <= end_date)

    return filtered_lf


def apply_column_filter(
    lf: pl.LazyFrame,
    column: str,
    values: Union[List[Any], Any]
) -> pl.LazyFrame:
    """
    컬럼 값 필터 적용

    Args:
        lf: LazyFrame
        column: 컬럼명
        values: 필터 값 (리스트 또는 단일 값)

    Returns:
        필터링된 LazyFrame
    """
    if isinstance(values, list):
        if len(values) > 0:
            return lf.filter(pl.col(column).is_in(values))
        return lf
    else:
        return lf.filter(pl.col(column) == values)


# ==================== 날짜 관련 헬퍼 함수 ====================

def get_date_range_from_data(
    lf: pl.LazyFrame,
    date_col: str = 'date_received'
) -> tuple:
    """
    데이터의 최소/최대 날짜 반환

    Args:
        lf: LazyFrame
        date_col: 날짜 컬럼명

    Returns:
        (min_date, max_date) 튜플
    """
    result = lf.select([
        pl.col(date_col).min().alias("min_date"),
        pl.col(date_col).max().alias("max_date")
    ]).collect()

    return result["min_date"][0], result["max_date"][0]


def generate_monthly_cache_key() -> str:
    """
    현재 월 기준 캐시 키 생성 (매월 1일에 자동 갱신)

    Returns:
        "YYYY-MM" 형태의 캐시 키
    """
    return datetime.now().strftime("%Y-%m")


# ==================== 키워드 분석 함수 ====================

def cluster_keyword_unpack(
    df: pl.DataFrame,
    col_name: str,
    cluster_col: str = 'defect_type',
    verbose: bool = True
) -> pl.DataFrame:
    """
    클러스터 별로 col_name마다 있는 리스트를 열어서 키워드 종류를 추출하고 count
    (벡터화 연산으로 대용량 데이터 처리 최적화)

    Parameters:
    -----------
    df : pl.DataFrame
        클러스터 정보가 포함된 데이터프레임
    col_name : str
        리스트가 들어있는 열 이름 (예: 'problem_components')
    cluster_col : str
        클러스터 열 이름 (기본값: 'defect_type')
    verbose : bool
        결과 출력 여부 (기본값: True)

    Returns:
    --------
    pl.DataFrame
        클러스터별 키워드, count, ratio를 포함한 데이터프레임
    """

    # 1. 문자열을 리스트로 변환 (필요한 경우)
    df_temp = df.select([cluster_col, col_name])

    if df_temp[col_name].dtype == pl.Utf8:
        df_temp = df_temp.with_columns(
            pl.col(col_name)
            .map_elements(lambda x: ast.literal_eval(x) if x else [], return_dtype=pl.List(pl.Utf8))
        )

    # 2. 전체 데이터를 한 번에 explode (벡터화)
    exploded_df = (df_temp
                   .explode(col_name)
                   .filter(pl.col(col_name).is_not_null())
                   .filter(pl.col(col_name) != "")  # 빈 문자열 제거
                  )

    # 3. 클러스터별로 그룹화하여 카운트 (벡터화)
    keyword_counts = (exploded_df
                      .with_columns(
                          pl.col(col_name).str.to_lowercase().str.strip_chars()  # 소문자 + 공백 제거
                          )
                      .group_by([cluster_col, col_name])
                      .agg(pl.len().alias('count'))
                     )

    # 4. 클러스터별 전체 키워드 수 계산
    cluster_totals = (keyword_counts
                      .group_by(cluster_col)
                      .agg(pl.col('count').sum().alias('total_count'))
                     )

    # 5. ratio 계산 및 정렬
    result_df = (keyword_counts
                 .join(cluster_totals, on=cluster_col)
                 .with_columns(
                     (pl.col('count') / pl.col('total_count')).alias('ratio')
                 )
                 .select([cluster_col, col_name, 'count', 'ratio'])
                 .sort([cluster_col, 'count'], descending=[False, True])
                )

    # 6. 결과 출력 (요약 정보)
    if verbose:
        for cluster_id in result_df[cluster_col].unique().sort():
            cluster_data = result_df.filter(pl.col(cluster_col) == cluster_id)
            print(f"\n=== Cluster {cluster_id} ===")
            print(f"총 키워드 수: {cluster_data['count'].sum()}")
            print(f"고유 키워드 수: {len(cluster_data)}")
            print(f"Top 10:")
            print(cluster_data.head(10))

    return result_df


@st.cache_data
def preprocess_problem_components(
    _df: pl.DataFrame,
    col_name: str = 'problem_components',
    cluster_col: str = 'defect_type'
) -> pl.DataFrame:
    """
    데이터 로드 직후 problem_components 컬럼을 전처리하여 키워드 분석 결과 반환
    (Streamlit 캐싱으로 중복 실행 방지)

    Parameters:
    -----------
    _df : pl.DataFrame
        원본 데이터프레임 (언더스코어로 시작하여 캐싱)
    col_name : str
        분석할 컬럼명 (기본값: 'problem_components')
    cluster_col : str
        클러스터 컬럼명 (기본값: 'defect_type')

    Returns:
    --------
    pl.DataFrame
        클러스터별 키워드 분석 결과
    """
    return cluster_keyword_unpack(_df, col_name, cluster_col, verbose=False)
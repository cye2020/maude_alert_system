"""전처리용 EDA (Exploratory Data Analysis) 함수들

데이터 탐색 및 분석을 위한 시각화/프로파일링 함수
"""

import polars as pl
from IPython.display import display

from src.utils import is_running_in_notebook
from src.utils.visualization import draw_donut_chart

if not is_running_in_notebook():
    display = print


def eda_proportion(lf: pl.LazyFrame, col: str, n_rows: int = 100, top_n: int = 5) -> None:
    """특정 컬럼의 값 분포를 테이블과 도넛 차트로 시각화

    값의 빈도와 비율을 계산하여 테이블로 표시하고, 도넛 차트로 시각화

    Args:
        lf (pl.LazyFrame): 분석할 LazyFrame
        col (str): 분석할 컬럼명
        n_rows (int, optional): 테이블에 표시할 최대 행 수. Defaults to 100.
        top_n (int, optional): 차트에 개별 표시할 상위 항목 개수. Defaults to 5.

    Returns:
        None: 테이블과 차트를 화면에 표시

    Examples:
        >>> eda_proportion(lf, 'device_class', n_rows=50, top_n=3)
    """
    # 값별 카운트 및 비율 계산
    count_lf = lf.select(
        pl.col(col).value_counts(sort=True)  # 빈도수 계산 및 정렬
    ).unnest(col).with_columns(
        (pl.col('count') / pl.col('count').sum() * 100).round(2).alias('percentage')  # 백분율 계산
    ).sort(by='count', descending=True).head(n_rows)

    # 테이블 표시
    display(count_lf.collect().to_pandas())

    # 도넛 차트 표시
    draw_donut_chart(count_lf.collect(), col, top_n)


def overview_col(lf: pl.LazyFrame, col: str, n_rows: int = 100) -> None:
    """특정 컬럼의 고유값 개수와 샘플 값들을 표시

    컬럼의 고유값(unique) 개수를 출력하고, 상위/하위 샘플 값들을 테이블로 표시

    Args:
        lf (pl.LazyFrame): 분석할 LazyFrame
        col (str): 분석할 컬럼명
        n_rows (int, optional): 표시할 샘플 개수. Defaults to 100.

    Returns:
        None: 고유값 개수와 샘플 테이블을 화면에 표시

    Examples:
        >>> overview_col(lf, 'manufacturer_name', n_rows=50)
        manufacturer_name의 고유 개수: 1234
        [head/tail 샘플 테이블 표시]
    """
    # 고유값 개수 계산
    nunique = lf.select(
        pl.col(col).n_unique().alias(f'unique_{col}')
    ).collect().item()

    print(f'{col}의 고유 개수: {nunique}')

    # 고유값을 정렬하여 상위/하위 샘플 추출
    unique_lf = lf.select(
        pl.col(col).unique().sort().head(n_rows).alias(f'head_{col}'),  # 상위 n개
        pl.col(col).unique().sort().tail(n_rows).alias(f'tail_{col}'),  # 하위 n개
    )

    # 테이블 표시
    display(unique_lf.collect().to_pandas())


def analyze_null_values(lf: pl.LazyFrame, analysis_cols=None, verbose=True) -> pl.DataFrame:
    """전체 컬럼의 결측치(null) 개수와 비율을 분석

    각 컬럼별 결측치 개수와 비율을 계산하여 내림차순으로 정렬된 DataFrame 반환

    Args:
        lf (pl.LazyFrame): 분석할 LazyFrame
        analysis_cols (List[str], optional): 분석할 컬럼 리스트. None이면 전체 컬럼. Defaults to None.
        verbose (bool, optional): 결과를 출력할지 여부. Defaults to True.

    Returns:
        pl.DataFrame: 'column', 'null_count', 'null_pct' 컬럼을 포함한 결측치 분석 결과
            - column: 컬럼명
            - null_count: 결측치 개수
            - null_pct: 결측치 비율(%)

    Examples:
        >>> null_df = analyze_null_values(lf, verbose=True)
        === 결측치 분석 ===
        전체 행 수: 1,000,000

        patient_age                                  :    500,000개 ( 50.00%)
        device_model                                 :    300,000개 ( 30.00%)
        ...
    """
    # 분석할 컬럼 결정 (None이면 전체 컬럼)
    if analysis_cols is None:
        analysis_cols = lf.collect_schema().names()

    # 전체 행 수 계산
    total_rows = lf.select(pl.len()).collect().item()

    # 각 컬럼의 null count를 한 번에 계산
    null_df = (
        lf.select([pl.col(col).null_count().alias(col) for col in analysis_cols])
        .collect()
        .transpose(include_header=True, header_name='column', column_names=['null_count'])  # 전치
        .with_columns(
            (pl.col('null_count') / total_rows * 100).round(2).alias('null_pct')  # 백분율 계산
        )
        .sort('null_pct', descending=True)  # 결측치 비율 내림차순 정렬
    )

    # verbose 모드일 경우 결과 출력
    if verbose:
        print("\n=== 결측치 분석 ===")
        print(f"전체 행 수: {total_rows:,}\n")
        for row in null_df.iter_rows(named=True):
            print(f"{row['column']:45s}: {row['null_count']:>10,}개 ({row['null_pct']:>6.2f}%)")

    return null_df

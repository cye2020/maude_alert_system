from typing import List
import polars as pl

def analyze_null_values(
    lf: pl.LazyFrame, 
    analysis_cols: List[str] = None, 
    verbose: bool = True
) -> pl.DataFrame:
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
            pl.col('null_count').truediv(total_rows).mul(100).round(2).alias('null_pct')  # 백분율 계산
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
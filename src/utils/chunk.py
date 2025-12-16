"""
LazyFrame 처리 유틸리티 함수
"""

import polars as pl
import shutil
from pathlib import Path
from typing import Callable
from tqdm import trange


def process_lazyframe_in_chunks(
    lf: pl.LazyFrame,
    transform_func: Callable[[pl.LazyFrame], pl.LazyFrame],
    output_path: Path,
    chunk_size: int = 10_000,
    temp_dir_name: str = "temp_chunks",
    keep_temp: bool = False,
    desc: str = "Processing"
) -> None:
    """
    LazyFrame을 chunk 단위로 처리하고 병합
    
    Args:
        lf: 입력 LazyFrame
        transform_func: 변환 함수 (LazyFrame → LazyFrame)
        output_path: 최종 출력 경로
        chunk_size: chunk 크기 (행 수)
        temp_dir_name: 임시 디렉토리 이름
        keep_temp: 임시 파일 유지 여부
        desc: 진행 상황 설명
    
    Example:
        >>> def my_transform(lf):
        >>>     return lf.with_columns(pl.col("text").str.to_uppercase())
        >>> 
        >>> process_lazyframe_in_chunks(
        >>>     lf=data_lf,
        >>>     transform_func=my_transform,
        >>>     output_path=Path("output.parquet")
        >>> )
    """
    # 1. 전체 행 수 확인
    total_rows = lf.select(pl.len()).collect().item()
    print(f"Processing {total_rows:,} rows in chunks of {chunk_size:,}...")
    
    # 2. 임시 디렉토리 생성
    temp_dir = output_path.parent / temp_dir_name
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 3. Chunk 단위 처리
        for offset in trange(0, total_rows, chunk_size, desc=desc):
            chunk_lf = lf.slice(offset, chunk_size)
            
            # 변환 함수 적용
            chunk_transformed = transform_func(chunk_lf)
            
            # Chunk 저장
            chunk_path = temp_dir / f'chunk_{offset}.parquet'
            chunk_transformed.sink_parquet(
                chunk_path,
                compression='zstd',
                compression_level=3
            )
        
        # 4. 병합
        print("Merging chunks...")
        pl.scan_parquet(str(temp_dir / 'chunk_*.parquet')).sink_parquet(
            output_path,
            compression='zstd',
            compression_level=3
        )
        
        print(f"✓ Saved to {output_path}")
    
    finally:
        # 5. 임시 파일 정리
        if not keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir)


def apply_mapping_to_columns(
    lf: pl.LazyFrame,
    columns: list[str],
    mapping_dict: dict,
    return_dtype: type = pl.Utf8
) -> pl.LazyFrame:
    """
    여러 컬럼에 동일한 매핑 딕셔너리 적용
    
    Args:
        lf: 입력 LazyFrame
        columns: 처리할 컬럼 리스트
        mapping_dict: 매핑 딕셔너리
        return_dtype: 반환 데이터 타입
    
    Returns:
        변환된 LazyFrame
    """
    def map_func(val):
        return mapping_dict.get(val, val)
    
    return lf.with_columns([
        pl.col(col).map_elements(map_func, return_dtype=return_dtype)
        for col in columns
    ])
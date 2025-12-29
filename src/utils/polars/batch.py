"""Polars 배치 처리 유틸리티

대용량 데이터를 chunk 단위로 나누어 처리하는 함수들
"""

from typing import Callable
from pathlib import Path
import polars as pl
from tqdm import tqdm


def process_in_chunks(
    lf: pl.LazyFrame,
    transform_func: Callable[[pl.LazyFrame], pl.LazyFrame],
    output_path: Path,
    chunk_size: int = 1_000_000,
    desc: str = "Processing"
) -> None:
    """
    LazyFrame을 chunk 단위로 처리하여 저장

    Args:
        lf: 입력 LazyFrame
        transform_func: 변환 함수 (LazyFrame → LazyFrame)
        output_path: 출력 경로
        chunk_size: chunk 크기
        desc: 진행 표시줄 설명
    """
    # 전체 행 수 계산
    total_rows = lf.select(pl.count()).collect().item()
    n_chunks = (total_rows + chunk_size - 1) // chunk_size

    temp_dir = output_path.parent / f"temp_{output_path.stem}"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Chunk 단위 처리
        for i in tqdm(range(n_chunks), desc=desc):
            chunk_lf = lf.slice(i * chunk_size, chunk_size)
            transformed_lf = transform_func(chunk_lf)

            # 임시 파일 저장
            chunk_path = temp_dir / f"chunk_{i:04d}.parquet"
            transformed_lf.collect().write_parquet(chunk_path)

        # 병합
        print(f"Merging {n_chunks} chunks...")
        pl.scan_parquet(temp_dir / "*.parquet").sink_parquet(output_path)

    finally:
        # 임시 파일 삭제
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def collect_unique_safe(lf: pl.LazyFrame, column: str) -> list:
    """
    LazyFrame에서 unique 값만 안전하게 collect

    Args:
        lf: LazyFrame
        column: 컬럼명

    Returns:
        Unique 값 리스트 (null 제외)
    """
    return lf.select(column).unique().drop_nulls().collect()[column].to_list()

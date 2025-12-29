from pathlib import Path
import polars as pl


def read_parquet(path: Path) -> pl.LazyFrame:
    return pl.scan_parquet(path)


def write_parquet(lf: pl.LazyFrame, path: Path):
    lf.sink_parquet(path)
    return path

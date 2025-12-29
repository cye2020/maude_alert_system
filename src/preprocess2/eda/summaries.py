import polars as pl


def summary_nulls(lf: pl.LazyFrame) -> pl.DataFrame:
    cols = lf.collect_schema().names()
    return lf.select([
        (pl.col(c).null_count() / pl.len()).alias(c) for c in cols
    ]).collect()

import polars as pl


def profile_value_counts(lf: pl.LazyFrame, col: str, top_n: int = 10) -> pl.DataFrame:
    return (
        lf.select(pl.col(col).value_counts(sort=True))
        .unnest(col)
        .head(top_n)
        .collect()
    )

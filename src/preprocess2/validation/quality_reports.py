import polars as pl


def drop_low_quality(lf: pl.LazyFrame, min_match_score: int, drop_missing_signature: bool) -> pl.LazyFrame:
    expr = pl.col("match_score") >= min_match_score
    if drop_missing_signature:
        expr = expr & pl.col("device_signature").is_not_null()
    return lf.filter(expr)

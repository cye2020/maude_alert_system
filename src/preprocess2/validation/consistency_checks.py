import polars as pl


def flag_consistency(lf: pl.LazyFrame) -> pl.LazyFrame:
    """간단한 일관성 검사: adverse_event_flag와 product_problem_flag가 둘 다 null인 행을 표시"""
    return lf.with_columns([
        pl.when(
            pl.col("adverse_event_flag").is_null() & pl.col("product_problem_flag").is_null()
        )
        .then(pl.lit("low_quality"))
        .otherwise(pl.lit("ok"))
        .alias("consistency_flag")
    ])

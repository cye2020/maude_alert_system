from datetime import date
from typing import Optional

import structlog
from snowflake.connector.cursor import SnowflakeCursor
import numpy as np
import pandas as pd

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier
from maude_early_alert.utils.config_loader import load_config
from maude_early_alert.analyze.aggregation import Aggregation
from maude_early_alert.analyze.spike_detection import SpikeDetection
from maude_early_alert.analyze.statistical_analysis_python import StatisticalAnalysisPython

_cfg  = load_config('gold')
_spike = _cfg['spike_detection']
_stat  = _cfg['statistical_analysis']

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class GoldPipeline(SnowflakeBase):
    """
    Silver -> Gold 파이프라인
     
    Aggregation / SpikeDetection / StatisticalAnalysis 결과물
    
    생성 테이블
    CLUSTER_{YYYYMM}_DASHBOARD
    DEFECT_{YYYYMM}_SPIKE
    PRODUCT_{YYYYMM}_DASHBOARD
    OVERVIEW_{YYYYMM}_DASHBOARD
    SPIKE_{YYYYMM}_RESULT
    STAT_{YYYYMM}_CHI2 / ODDS / FINAL
    """
    
    CLUSTER_METRICS = [
        "REPORT_COUNT", "DEATH_COUNT", "SERIOUS_INJURY_COUNT",
        "MINOR_INJURY_COUNT", "NO_HARM_COUNT",
        "CFR", "DEFECT_CONFIRMED_COUNT", "DEFECT_CONFIRMED_RATE",
    ]
    PRODUCT_METRICS = ["REPORT_COUNT", "DEATH_COUNT", "CFR"]
    OVERVIEW_METRICS = [
        "TOTAL_REPORT_COUNT", "DEATH_COUNT", "SERIOUS_INJURY_COUNT",
        "SEVERE_HARM_RATE", "DEFECT_CONFIRMED_RATE",
    ]
    
    def __init__(
        self, 
        database: str,
        schema : str = "SILVER", 
        gold_schema : str = "GOLD",
    ):
        super().__init__(database, schema)
        self.gold_schema = validate_identifier(gold_schema)
    
    # ===================================================================
    # 내부 헬퍼
    # ===================================================================
    WINDOWS = _cfg['windows']

    def _table_name(
        self,
        group : str,
        use   : str,
        window: int = None,
    ) -> str:
        w_part = f"_{window}M" if window else ""
        return f"{group.upper()}{w_part}_{use.upper()}"
    
    def _ensure_gold_table(
        self, 
        cursor:SnowflakeCursor,
        table_name: str
    ) -> None:
        full = f"{self.database}.{self.gold_schema}.{table_name}"
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {full} (
                METRIC_DATE DATE,
                GRAIN VARCHAR,
                MANUFACTURER VARCHAR,
                ENTITY_VALUE VARCHAR,
                METRIC_NAME VARCHAR,
                METRIC_VALUE FLOAT,
                SOURCE_TABLE VARCHAR,
                SNAPSHOT_DATE DATE
            )
        """)
        
    @staticmethod
    def _dtype_to_sf(dtype) -> str:
        if pd.api.types.is_integer_dtype(dtype):
            return "NUMBER"
        if pd.api.types.is_float_dtype(dtype):
            return "FLOAT"
        if pd.api.types.is_bool_dtype(dtype):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "TIMESTAMP_NTZ"
        return "VARCHAR"

    def _write_gold(
        self,
        cursor : SnowflakeCursor,
        df_long : pd.DataFrame,
        table_name : str,
    ) -> None:
        full = f"{self.database}.{self.gold_schema}.{table_name}"
        self._ensure_gold_table(cursor, table_name)
        cursor.execute(f"TRUNCATE TABLE {full}")
        rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v
                      for v in r)
                for r in df_long.itertuples(index=False, name=None)]
        cursor.executemany(
            f"INSERT INTO {full} "
            "(METRIC_DATE,GRAIN,MANUFACTURER,ENTITY_VALUE,"
            "METRIC_NAME,METRIC_VALUE,SOURCE_TABLE,SNAPSHOT_DATE) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
            rows,
        )
        logger.info("Gold 적재 완료", table=full, rows=len(df_long))

    def _write_spike(
        self,
        cursor : SnowflakeCursor,
        df : pd.DataFrame,
        table_name : str,
    ) -> None:
        full = f"{self.database}.{self.gold_schema}.{table_name}"
        col_defs = ", ".join(
            f"{c} {self._dtype_to_sf(t)}"
            for c, t in zip(df.columns, df.dtypes)
        )
        cursor.execute(f"DROP TABLE IF EXISTS {full}")
        cursor.execute(f"CREATE TABLE {full} ({col_defs})")
        placeholders = ", ".join(["%s"] * len(df.columns))
        rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v
                      for v in r)
                for r in df.itertuples(index=False, name=None)]
        cursor.executemany(
            f"INSERT INTO {full} VALUES ({placeholders})",
            rows,
        )
        logger.info("Spike 적재 완료", table=full, rows=len(df))
        
    def _melt(
        self,
        df_wide : pd.DataFrame,
        entity_col : str,
        manufacturer_col : str,
        snapshot_date : str,
        source_table : str,
        metric_cols : list,
    ) -> pd.DataFrame:
        df = df_wide.copy()
        df["GRAIN"] = "month"
        df["SOURCE_TABLE"] = source_table
        df["SNAPSHOT_DATE"] = snapshot_date
        df = df.rename(columns={
            entity_col : "ENTITY_VALUE",
            manufacturer_col:"MANUFACTURER",
        })
        long = df.melt(
            id_vars = ["METRIC_DATE", "GRAIN", "MANUFACTURER", "ENTITY_VALUE",
                       "SOURCE_TABLE", "SNAPSHOT_DATE"],
            value_vars = metric_cols,
            var_name = "METRIC_NAME", 
            value_name = "METRIC_VALUE",
        )
        long["METRIC_VALUE"] = pd.to_numeric(long["METRIC_VALUE"], errors="coerce")
        return long[[
            "METRIC_DATE", "GRAIN", "MANUFACTURER", "ENTITY_VALUE",
            "METRIC_NAME", "METRIC_VALUE",
            "SOURCE_TABLE", "SNAPSHOT_DATE",
        ]]
        
    # ===================================================================
    # 1. CLUSTER_{YYYYMM}_DASHBOARD
    # ===================================================================

    @with_context
    def run_cluster_dashboard(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        windows      : list = None,
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        agg = Aggregation(database=self.database, schema=self.schema)
        for w in windows:
            table_name = self._table_name("CLUSTER", "DASHBOARD", w)
            df = agg.cluster_dashboard(cursor, source_table, snapshot_date, window=w)
            df_long = self._melt(df, "CLUSTER_ID", "MANUFACTURER",
                                 snapshot_date, source_table, self.CLUSTER_METRICS)
            self._write_gold(cursor, df_long, table_name)
            logger.info("CLUSTER 적재완료", table=table_name, window=w, rows=len(df_long))
        
    # ===================================================================
    # 2. DEFECT_{YYYYMM}_SPIKE
    # ===================================================================

    @with_context
    def run_defect_spike(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        windows      : list = None,
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        agg = Aggregation(database=self.database, schema=self.schema)
        for w in windows:
            table_name = self._table_name("DEFECT", "SPIKE", w)
            df = agg.defect_spike(cursor, source_table, snapshot_date, window=w)
            df_long = self._melt(df, "DEFECT_TYPE", "MANUFACTURER",
                                 snapshot_date, source_table, ["REPORT_COUNT"])
            self._write_gold(cursor, df_long, table_name)
            logger.info("DEFECT 적재완료", table=table_name, window=w, rows=len(df_long))
        
    # ------------------------------------------------------------------
    # 3. PRODUCT_{YYYYMM}_DASHBOARD
    # ------------------------------------------------------------------

    @with_context
    def run_product_dashboard(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        windows      : list = None,
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        agg = Aggregation(database=self.database, schema=self.schema)
        for w in windows:
            table_name = self._table_name("PRODUCT", "DASHBOARD", w)
            df = agg.product_dashboard(cursor, source_table, snapshot_date, window=w)
            df_long = self._melt(df, "PRODUCT_CODE", "MANUFACTURER",
                                 snapshot_date, source_table, self.PRODUCT_METRICS)
            self._write_gold(cursor, df_long, table_name)
            logger.info("PRODUCT 적재 완료", table=table_name, window=w, rows=len(df_long))

    # ------------------------------------------------------------------
    # 4. OVERVIEW_{YYYYMM}_DASHBOARD
    # ------------------------------------------------------------------

    @with_context
    def run_overview_dashboard(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        windows      : list = None,
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        agg = Aggregation(database=self.database, schema=self.schema)
        for w in windows:
            table_name = self._table_name("OVERVIEW", "DASHBOARD", w)
            df = agg.overview_dashboard(cursor, source_table, snapshot_date, window=w)
            df["ENTITY_VALUE"] = "TOTAL"
            df_long = self._melt(df, "ENTITY_VALUE", "MANUFACTURER",
                                 snapshot_date, source_table, self.OVERVIEW_METRICS)
            self._write_gold(cursor, df_long, table_name)
            logger.info("OVERVIEW 적재 완료", table=table_name, window=w, rows=len(df_long))

    # ------------------------------------------------------------------
    # 5. SPIKE_{YYYYMM}_RESULT
    # ------------------------------------------------------------------

    @with_context
    def run_spike_detection(
        self,
        cursor        : SnowflakeCursor,
        source_table  : str = _cfg['source_table'],
        snapshot_date : Optional[str] = None,
        windows       : list = None,
        keyword_column: str = _spike['keyword_column'],
        date_column   : str = _spike['date_column'],
        count_column  : str = _spike['count_column'],
        filters       : Optional[str] = _spike['filters'],
        z_threshold   : float = _spike['z_threshold'],
        min_c_recent  : int = _spike['min_c_recent'],
        alpha         : float = _spike['alpha'],
        correction    : str = _spike['correction'],
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        src           = f"{self.database}.{self.schema}.{source_table}"

        detector = SpikeDetection(database=self.database, schema=self.schema)

        with self._error_logging("SPIKE fetch", table=source_table):
            keyword_monthly, monthly_total = detector.fetch_monthly_counts(
                cursor,
                source_table   = src,
                keyword_column = keyword_column,
                date_column    = date_column,
                count_column   = count_column,
                filters        = filters,
            )

        as_of_month  = monthly_total["MONTH"].sort_values(ascending=False).iloc[0]
        all_keywords = keyword_monthly["KEYWORD"].unique()

        for w in windows:
            recent_months, base_months = detector.get_window_months(as_of_month, w)
            df = detector.aggregate_window(
                keyword_monthly, monthly_total, all_keywords, recent_months, base_months
            )
            df["AS_OF_MONTH"] = as_of_month
            df["WINDOW"]      = w
            df = detector.calculate_ratio_metrics(df)
            df = detector.calculate_zscore_metrics(df)
            df = detector.calculate_poisson_metrics(df, alpha, correction)
            df = detector.determine_spikes(df, z_threshold, min_c_recent, alpha)
            df = detector.add_ensemble_pattern(df)
            df["SNAPSHOT_DATE"] = snapshot_date

            table_name = self._table_name("SPIKE", "RESULT", w)
            logger.info("SPIKE 탐지 완료", window=w, rows=len(df), snapshot_date=snapshot_date)
            self._write_spike(cursor, df, table_name)

    # ------------------------------------------------------------------
    # 6. STAT_{YYYYMM}_{CHI2|ODDS|FINAL}
    # ------------------------------------------------------------------

    @with_context
    def run_statistical_analysis(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = _cfg['source_table'],
        snapshot_date: Optional[str] = None,
        row_column   : str = _stat['row_column'],
        col_column   : str = _stat['col_column'],
        filters      : Optional[str] = _stat['filters'],
        min_cell     : int = _stat['min_cell'],
        alpha        : float = _stat['alpha'],
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        src           = f"{self.database}.{self.schema}.{source_table}"

        analyzer = StatisticalAnalysisPython(database=self.database, schema=self.schema)

        with self._error_logging("STAT 분석", table=source_table):
            result = analyzer.run(
                cursor,
                source_table   = src,
                row_column     = row_column,
                col_column     = col_column,
                filters        = filters,
                min_cell_count = min_cell,
                alpha          = alpha,
            )

        chi2_table  = "STAT_CHI2_RESULT"
        odds_table  = "STAT_ODDS_RATIOS"
        final_table = "STAT_FINAL"

        self._write_spike(cursor, result["chi2"]["detail"], chi2_table)
        self._write_spike(cursor, result["odds_ratios"],    odds_table)
        self._write_spike(cursor, result["final"],          final_table)

        logger.info("통계 분석 완료",
                    tables=[chi2_table, odds_table, final_table],
                    snapshot_date=snapshot_date)

    # ------------------------------------------------------------------
    # run : 전체 Gold 집계 실행
    # ------------------------------------------------------------------

    def run(
        self,
        cursor       : SnowflakeCursor,
        source_table : str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        windows      : list = None,
    ) -> None:
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        windows       = windows or self.WINDOWS
        logger.info("Gold 파이프라인 시작",
                    source_table=source_table, snapshot_date=snapshot_date, windows=windows)

        self.run_cluster_dashboard   (cursor, source_table, snapshot_date, windows)
        self.run_defect_spike        (cursor, source_table, snapshot_date, windows)
        self.run_product_dashboard   (cursor, source_table, snapshot_date, windows)
        self.run_overview_dashboard  (cursor, source_table, snapshot_date, windows)
        self.run_spike_detection     (cursor, source_table, snapshot_date, windows)
        self.run_statistical_analysis(cursor, source_table, snapshot_date)

        logger.info("Gold 파이프라인 완료", snapshot_date=snapshot_date)


if __name__ == "__main__":
    import snowflake.connector
    from maude_early_alert.logging_config import configure_logging
    from maude_early_alert.utils.secrets import get_secret

    configure_logging(level="INFO")

    SOURCE_TABLE  = "EVENT_CLUSTERED"
    SNAPSHOT_DATE = None

    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret)
    cursor = conn.cursor()

    try:
        pipeline = GoldPipeline(database="MAUDE", schema="SILVER", gold_schema="GOLD")
        pipeline.run_cluster_dashboard (cursor, SOURCE_TABLE, SNAPSHOT_DATE)
        pipeline.run_defect_spike      (cursor, SOURCE_TABLE, SNAPSHOT_DATE)
        pipeline.run_product_dashboard (cursor, SOURCE_TABLE, SNAPSHOT_DATE)
        pipeline.run_overview_dashboard(cursor, SOURCE_TABLE, SNAPSHOT_DATE)
    except Exception:
        logger.error("Gold 파이프라인 실패", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

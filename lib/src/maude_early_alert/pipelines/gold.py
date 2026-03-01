from datetime import date
from typing import Optional

import structlog
from snowflake.connector.cursor import SnowflakeCursor
from snowflake.connector.pandas_tools import write_pandas
import pandas as pd

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier, validate_date
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
        "SEVERE_HARM_RATE", "DEFECT_CONFIRMED_COUNT", "DEFECT_CONFIRMED_RATE",
    ]
    PRODUCT_METRICS = ["REPORT_COUNT", "DEATH_COUNT", "SEVERE_HARM_RATE"]
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
    
    def _write_ctas(
        self,
        cursor    : SnowflakeCursor,
        sql       : str,
        table_name: str,
    ) -> None:
        """CREATE OR REPLACE TABLE ... AS SELECT 로 Gold 테이블 전체 교체"""
        cursor.execute(sql)
        logger.info("Gold CTAS 완료", table=f"{self.database}.{self.gold_schema}.{table_name}")

    def _write_incremental(
        self,
        cursor       : SnowflakeCursor,
        df           : pd.DataFrame,
        table_name   : str,
        snapshot_date: str,
    ) -> None:
        """자동 스키마 추론 + 증분 적재 (Spike용, DELETE + INSERT, 트랜잭션 보장)

        [Phase 1] DDL: 빈 DataFrame으로 스키마만 생성. Snowflake DDL은 auto-commit이므로
                       트랜잭션 밖에서 실행. 테이블이 이미 존재하면 no-op.
        [Phase 2] DML: snapshot_date 기준 DELETE + INSERT를 트랜잭션으로 보호.
                       첫 실행 / 재실행 모두 동일 경로 → 멱등성 보장.
        """
        full = f"{self.database}.{self.gold_schema}.{table_name}"

        # Phase 1: 테이블 스키마 생성 (DDL, auto-commit, 이미 있으면 no-op)
        write_pandas(
            cursor.connection,
            df.head(0),
            table_name,
            database=self.database,
            schema=self.gold_schema,
            overwrite=False,
            auto_create_table=True,
            quote_identifiers=True,
        )

        # Phase 2: 트랜잭션으로 보호된 DELETE + INSERT
        with self.transaction(cursor):
            cursor.execute(f"DELETE FROM {full} WHERE SNAPSHOT_DATE = '{snapshot_date}'")
            success, _, nrows, _ = write_pandas(
                cursor.connection,
                df,
                table_name,
                database=self.database,
                schema=self.gold_schema,
                overwrite=False,
                auto_create_table=False,
                quote_identifiers=True,
            )
            if not success:
                raise RuntimeError(f"write_pandas 실패: {full}")
        logger.info("Spike 적재 완료", table=full, rows=nrows)

    def _write_auto(
        self,
        cursor    : SnowflakeCursor,
        df        : pd.DataFrame,
        table_name: str,
    ) -> None:
        """자동 스키마 추론 + 전체 교체 (Stat용, 테이블명에 YYYYMM 포함)"""
        full = f"{self.database}.{self.gold_schema}.{table_name}"
        success, _, nrows, _ = write_pandas(
            cursor.connection,
            df,
            table_name,
            database=self.database,
            schema=self.gold_schema,
            overwrite=True,
            auto_create_table=True,
            quote_identifiers=True,
        )
        if not success:
            raise RuntimeError(f"write_pandas 실패: {full}")
        logger.info("Stat 적재 완료", table=full, rows=nrows)
        
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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        windows       = windows or self.WINDOWS
        agg     = Aggregation(database=self.database, schema=self.schema)
        metrics = ", ".join(self.CLUSTER_METRICS)
        for w in windows:
            table_name = self._table_name("CLUSTER", "DASHBOARD", w)
            full       = f"{self.database}.{self.gold_schema}.{table_name}"
            agg_sql    = agg.cluster_dashboard_sql(source_table, snapshot_date, w)
            self._write_ctas(cursor, f"""
                CREATE OR REPLACE TABLE {full} AS
                SELECT METRIC_DATE,
                       'month'::VARCHAR        AS GRAIN,
                       MANUFACTURER,
                       CLUSTER_ID              AS ENTITY_VALUE,
                       METRIC_NAME,
                       METRIC_VALUE::FLOAT     AS METRIC_VALUE,
                       '{source_table}'        AS SOURCE_TABLE,
                       '{snapshot_date}'::DATE AS SNAPSHOT_DATE
                FROM ({agg_sql}) base
                UNPIVOT (METRIC_VALUE FOR METRIC_NAME IN ({metrics}))
                ORDER BY 1, 2, 3, 4, 5
            """, table_name)
            logger.info("CLUSTER 적재완료", table=table_name, window=w)

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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        windows       = windows or self.WINDOWS
        agg = Aggregation(database=self.database, schema=self.schema)
        for w in windows:
            table_name = self._table_name("DEFECT", "SPIKE", w)
            full       = f"{self.database}.{self.gold_schema}.{table_name}"
            agg_sql    = agg.defect_spike_sql(source_table, snapshot_date, w)
            self._write_ctas(cursor, f"""
                CREATE OR REPLACE TABLE {full} AS
                SELECT METRIC_DATE,
                       'month'::VARCHAR        AS GRAIN,
                       MANUFACTURER,
                       DEFECT_TYPE             AS ENTITY_VALUE,
                       'REPORT_COUNT'          AS METRIC_NAME,
                       REPORT_COUNT::FLOAT     AS METRIC_VALUE,
                       '{source_table}'        AS SOURCE_TABLE,
                       '{snapshot_date}'::DATE AS SNAPSHOT_DATE
                FROM ({agg_sql}) base
                ORDER BY 1, 2, 3
            """, table_name)
            logger.info("DEFECT 적재완료", table=table_name, window=w)

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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        windows       = windows or self.WINDOWS
        agg     = Aggregation(database=self.database, schema=self.schema)
        metrics = ", ".join(self.PRODUCT_METRICS)
        for w in windows:
            table_name = self._table_name("PRODUCT", "DASHBOARD", w)
            full       = f"{self.database}.{self.gold_schema}.{table_name}"
            agg_sql    = agg.product_dashboard_sql(source_table, snapshot_date, w)
            self._write_ctas(cursor, f"""
                CREATE OR REPLACE TABLE {full} AS
                SELECT METRIC_DATE,
                       'month'::VARCHAR        AS GRAIN,
                       MANUFACTURER,
                       PRODUCT_CODE            AS ENTITY_VALUE,
                       METRIC_NAME,
                       METRIC_VALUE::FLOAT     AS METRIC_VALUE,
                       '{source_table}'        AS SOURCE_TABLE,
                       '{snapshot_date}'::DATE AS SNAPSHOT_DATE
                FROM ({agg_sql}) base
                UNPIVOT (METRIC_VALUE FOR METRIC_NAME IN ({metrics}))
                ORDER BY 1, 2, 3, 4, 5
            """, table_name)
            logger.info("PRODUCT 적재 완료", table=table_name, window=w)

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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        windows       = windows or self.WINDOWS
        agg     = Aggregation(database=self.database, schema=self.schema)
        metrics = ", ".join(self.OVERVIEW_METRICS)
        for w in windows:
            table_name = self._table_name("OVERVIEW", "DASHBOARD", w)
            full       = f"{self.database}.{self.gold_schema}.{table_name}"
            agg_sql    = agg.overview_dashboard_sql(source_table, snapshot_date, w)
            self._write_ctas(cursor, f"""
                CREATE OR REPLACE TABLE {full} AS
                SELECT METRIC_DATE,
                       'month'::VARCHAR        AS GRAIN,
                       MANUFACTURER,
                       'TOTAL'                 AS ENTITY_VALUE,
                       METRIC_NAME,
                       METRIC_VALUE::FLOAT     AS METRIC_VALUE,
                       '{source_table}'        AS SOURCE_TABLE,
                       '{snapshot_date}'::DATE AS SNAPSHOT_DATE
                FROM ({agg_sql}) base
                UNPIVOT (METRIC_VALUE FOR METRIC_NAME IN ({metrics}))
                ORDER BY 1, 2, 3, 4, 5
            """, table_name)
            logger.info("OVERVIEW 적재 완료", table=table_name, window=w)

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
        eps           : float = _spike['eps'],
        z_threshold   : float = _spike['z_threshold'],
        min_c_recent  : int = _spike['min_c_recent'],
        alpha         : float = _spike['alpha'],
        correction    : str = _spike['correction'],
    ) -> None:
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        windows       = windows or self.WINDOWS
        src           = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"

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
            df = detector.calculate_zscore_metrics(df, eps)
            df = detector.calculate_poisson_metrics(df, alpha, correction)
            df = detector.determine_spikes(df, z_threshold, min_c_recent, alpha)
            df = detector.add_ensemble_pattern(df)
            df["SNAPSHOT_DATE"] = snapshot_date

            table_name = self._table_name("SPIKE", "RESULT", w)
            logger.info("SPIKE 탐지 완료", window=w, rows=len(df), snapshot_date=snapshot_date)
            self._write_incremental(cursor, df, table_name, snapshot_date)

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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
        src           = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"

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

        yyyymm      = snapshot_date[:7].replace("-", "")
        chi2_table  = f"STAT_{yyyymm}_CHI2_RESULT"
        odds_table  = f"STAT_{yyyymm}_ODDS_RATIOS"
        final_table = f"STAT_{yyyymm}_FINAL"

        self._write_auto(cursor, result["chi2"]["detail"], chi2_table)
        self._write_auto(cursor, result["odds_ratios"],    odds_table)
        self._write_auto(cursor, result["final"],          final_table)

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
        snapshot_date = validate_date(snapshot_date or date.today().strftime("%Y-%m-%d"))
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
        pipeline.run(cursor, SOURCE_TABLE, SNAPSHOT_DATE)
    except Exception:
        logger.error("Gold 파이프라인 실패", exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

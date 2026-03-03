from datetime import date
from typing import Optional

import pandas as pd
import structlog
from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class Aggregation(SnowflakeBase):
    """
    Parameters
    ----------
    database : str
        Snowflake 데이터베이스 (예: 'MAUDE')
    schema : str
        Silver 스키마 (기본값: 'SILVER')
    """

    def __init__(self, database: str, schema: str = "SILVER"):
        super().__init__(database, schema)

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    @staticmethod
    def _union_all(specific_sql: str, total_sql: str, order_by: str = "1, 2") -> str:
        """제조사별 집계 + 전체('ALL') 집계 UNION ALL"""
        return f"{specific_sql}\nUNION ALL\n{total_sql}\nORDER BY {order_by}"

    @staticmethod
    def _as_of(src: str) -> str:
        """데이터의 실제 최신 월 서브쿼리. snapshot_date 대신 MAX(DATE_RECEIVED) 기준."""
        return f"(SELECT DATE_TRUNC('MONTH', MAX(DATE_RECEIVED)) FROM {src} WHERE DATE_RECEIVED IS NOT NULL)"

    # ------------------------------------------------------------------
    # 1. 클러스터별 월간 지표
    # ------------------------------------------------------------------

    def cluster_dashboard_sql(
        self,
        source_table: str = "EVENT_CLUSTERED",
        window      : int = 12,
    ) -> str:
        """클러스터별 월간 지표 집계 SQL (wide 형식).

        Columns
        -------
        METRIC_DATE, MANUFACTURER, CLUSTER_ID,
        REPORT_COUNT, DEATH_COUNT, SERIOUS_INJURY_COUNT,
        MINOR_INJURY_COUNT, NO_HARM_COUNT,
        SEVERE_HARM_RATE, DEFECT_CONFIRMED_COUNT, DEFECT_CONFIRMED_RATE
        """
        src   = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"
        as_of = self._as_of(src)
        where = f"""WHERE DATE_RECEIVED IS NOT NULL
              AND CLUSTER != -1
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) >= DATEADD('month', -{window - 1}, {as_of})
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) <= {as_of}"""

        metric_expr = """
                COUNT(DISTINCT MDR_REPORT_KEY)::FLOAT                         AS REPORT_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Death'
                         THEN 1 ELSE 0 END)::FLOAT                            AS DEATH_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Serious Injury'
                         THEN 1 ELSE 0 END)::FLOAT                            AS SERIOUS_INJURY_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Minor Injury'
                         THEN 1 ELSE 0 END)::FLOAT                            AS MINOR_INJURY_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'No Apparent Injury'
                         THEN 1 ELSE 0 END)::FLOAT                            AS NO_HARM_COUNT,
                ROUND(
                    SUM(CASE WHEN PATIENT_HARM IN ('Death', 'Serious Injury')
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT MDR_REPORT_KEY), 0) * 100, 2)::FLOAT AS SEVERE_HARM_RATE,
                SUM(CASE WHEN DEFECT_CONFIRMED = TRUE
                         THEN 1 ELSE 0 END)::FLOAT                            AS DEFECT_CONFIRMED_COUNT,
                ROUND(
                    SUM(CASE WHEN DEFECT_CONFIRMED = TRUE
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT MDR_REPORT_KEY), 0) * 100, 2)::FLOAT AS DEFECT_CONFIRMED_RATE
        """

        return self._union_all(
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED) AS METRIC_DATE,
                   MANUFACTURER_NAME                  AS MANUFACTURER,
                   CLUSTER::VARCHAR                   AS CLUSTER_ID,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1, 2, 3
            """,
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED) AS METRIC_DATE,
                   'ALL'                              AS MANUFACTURER,
                   CLUSTER::VARCHAR                   AS CLUSTER_ID,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1, 3
            """,
            order_by="1, 2, 3",
        )

    @with_context
    def cluster_dashboard(
        self,
        cursor      : SnowflakeCursor,
        source_table: str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        window      : int = 12,
    ) -> pd.DataFrame:
        """클러스터별 월간 지표 집계. cluster_dashboard_sql() 결과를 실행."""
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        sql = self.cluster_dashboard_sql(source_table, window)
        with self._error_logging("CLUSTER 집계", table=source_table):
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
        logger.info("CLUSTER 집계 완료", rows=len(df), snapshot_date=snapshot_date)
        return df

    # ------------------------------------------------------------------
    # 2. defect_type별 월간 건수 (스파이크 탐지 입력용)
    # ------------------------------------------------------------------

    def defect_spike_sql(
        self,
        source_table: str = "EVENT_CLUSTERED",
        window      : int = 12,
    ) -> str:
        """defect_type별 월간 건수 집계 SQL (wide 형식).

        Columns
        -------
        METRIC_DATE, MANUFACTURER, DEFECT_TYPE, REPORT_COUNT
        """
        src   = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"
        as_of = self._as_of(src)
        where = f"""WHERE DATE_RECEIVED IS NOT NULL
              AND DEFECT_TYPE IS NOT NULL
              AND DEFECT_TYPE NOT IN ('Unknown', 'Other')
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) >= DATEADD('month', -{window - 1}, {as_of})
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) <= {as_of}"""

        return self._union_all(
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)   AS METRIC_DATE,
                   MANUFACTURER_NAME                    AS MANUFACTURER,
                   DEFECT_TYPE                          AS DEFECT_TYPE,
                   COUNT(DISTINCT MDR_REPORT_KEY)       AS REPORT_COUNT
            FROM {src} {where}
            GROUP BY 1, 2, 3
            """,
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)   AS METRIC_DATE,
                   'ALL'                                AS MANUFACTURER,
                   DEFECT_TYPE                          AS DEFECT_TYPE,
                   COUNT(DISTINCT MDR_REPORT_KEY)       AS REPORT_COUNT
            FROM {src} {where}
            GROUP BY 1, 3
            """,
            order_by="1, 2, 3",
        )

    @with_context
    def defect_spike(
        self,
        cursor      : SnowflakeCursor,
        source_table: str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        window      : int = 12,
    ) -> pd.DataFrame:
        """defect_type별 월간 건수 집계. defect_spike_sql() 결과를 실행."""
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        sql = self.defect_spike_sql(source_table, window)
        with self._error_logging("DEFECT 집계", table=source_table):
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
        logger.info("DEFECT 집계 완료", rows=len(df), snapshot_date=snapshot_date)
        return df

    # ------------------------------------------------------------------
    # 3. 제품별 월간 지표
    # ------------------------------------------------------------------

    def product_dashboard_sql(
        self,
        source_table: str = "EVENT_CLUSTERED",
        window      : int = 12,
    ) -> str:
        """제품별 월간 지표 집계 SQL (wide 형식).

        Columns
        -------
        METRIC_DATE, MANUFACTURER, PRODUCT_CODE,
        REPORT_COUNT, DEATH_COUNT, SEVERE_HARM_RATE
        """
        src   = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"
        as_of = self._as_of(src)
        where = f"""WHERE DATE_RECEIVED IS NOT NULL
              AND PRODUCT_CODE IS NOT NULL
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) >= DATEADD('month', -{window - 1}, {as_of})
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) <= {as_of}"""

        metric_expr = """
                COUNT(DISTINCT MDR_REPORT_KEY)::FLOAT                         AS REPORT_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Death'
                         THEN 1 ELSE 0 END)::FLOAT                            AS DEATH_COUNT,
                ROUND(
                    SUM(CASE WHEN PATIENT_HARM IN ('Death', 'Serious Injury')
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT MDR_REPORT_KEY), 0) * 100, 2)::FLOAT AS SEVERE_HARM_RATE
        """

        return self._union_all(
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)  AS METRIC_DATE,
                   MANUFACTURER_NAME                   AS MANUFACTURER,
                   PRODUCT_CODE                        AS PRODUCT_CODE,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1, 2, 3
            """,
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)  AS METRIC_DATE,
                   'ALL'                               AS MANUFACTURER,
                   PRODUCT_CODE                        AS PRODUCT_CODE,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1, 3
            """,
            order_by="1, 2, 3",
        )

    @with_context
    def product_dashboard(
        self,
        cursor      : SnowflakeCursor,
        source_table: str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        window      : int = 12,
    ) -> pd.DataFrame:
        """제품별 월간 지표 집계. product_dashboard_sql() 결과를 실행."""
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        sql = self.product_dashboard_sql(source_table, window)
        with self._error_logging("PRODUCT 집계", table=source_table):
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
        logger.info("PRODUCT 집계 완료", rows=len(df), snapshot_date=snapshot_date)
        return df

    # ------------------------------------------------------------------
    # 4. 전체 월간 요약 지표
    # ------------------------------------------------------------------

    def overview_dashboard_sql(
        self,
        source_table: str = "EVENT_CLUSTERED",
        window      : int = 12,
    ) -> str:
        """전체 월간 요약 지표 집계 SQL (wide 형식).

        Columns
        -------
        METRIC_DATE, MANUFACTURER,
        TOTAL_REPORT_COUNT, DEATH_COUNT, SERIOUS_INJURY_COUNT,
        SEVERE_HARM_RATE, DEFECT_CONFIRMED_RATE
        """
        src   = f"{self.database}.{self.schema}.{validate_identifier(source_table)}"
        as_of = self._as_of(src)
        where = f"""WHERE DATE_RECEIVED IS NOT NULL
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) >= DATEADD('month', -{window - 1}, {as_of})
              AND DATE_TRUNC('MONTH', DATE_RECEIVED) <= {as_of}"""

        metric_expr = """
                COUNT(DISTINCT MDR_REPORT_KEY)::FLOAT                         AS TOTAL_REPORT_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Death'
                         THEN 1 ELSE 0 END)::FLOAT                            AS DEATH_COUNT,
                SUM(CASE WHEN PATIENT_HARM = 'Serious Injury'
                         THEN 1 ELSE 0 END)::FLOAT                            AS SERIOUS_INJURY_COUNT,
                ROUND(
                    SUM(CASE WHEN PATIENT_HARM IN ('Death', 'Serious Injury')
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT MDR_REPORT_KEY), 0) * 100, 2)::FLOAT AS SEVERE_HARM_RATE,
                ROUND(
                    SUM(CASE WHEN DEFECT_CONFIRMED = TRUE
                             THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(DISTINCT MDR_REPORT_KEY), 0) * 100, 2)::FLOAT AS DEFECT_CONFIRMED_RATE
        """

        return self._union_all(
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)  AS METRIC_DATE,
                   MANUFACTURER_NAME                   AS MANUFACTURER,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1, 2
            """,
            f"""
            SELECT DATE_TRUNC('MONTH', DATE_RECEIVED)  AS METRIC_DATE,
                   'ALL'                               AS MANUFACTURER,
                   {metric_expr}
            FROM {src} {where}
            GROUP BY 1
            """,
        )

    @with_context
    def overview_dashboard(
        self,
        cursor      : SnowflakeCursor,
        source_table: str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        window      : int = 12,
    ) -> pd.DataFrame:
        """전체 월간 요약 지표 집계. overview_dashboard_sql() 결과를 실행."""
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        sql = self.overview_dashboard_sql(source_table, window)
        with self._error_logging("OVERVIEW 집계", table=source_table):
            cursor.execute(sql)
            df = cursor.fetch_pandas_all()
        logger.info("OVERVIEW 집계 완료", rows=len(df), snapshot_date=snapshot_date)
        return df

    # ------------------------------------------------------------------
    # run : 전체 집계 실행 → dict 반환
    # ------------------------------------------------------------------

    def run(
        self,
        cursor      : SnowflakeCursor,
        source_table: str = "EVENT_CLUSTERED",
        snapshot_date: Optional[str] = None,
        window      : int = 12,
    ) -> dict[str, pd.DataFrame]:
        """
        전체 집계 실행.

        Returns
        -------
        dict
            {
                'cluster' : cluster_dashboard DataFrame,
                'defect'  : defect_spike DataFrame,
                'product' : product_dashboard DataFrame,
                'overview': overview_dashboard DataFrame,
            }
        """
        snapshot_date = snapshot_date or date.today().strftime("%Y-%m-%d")
        logger.info("집계 시작", source_table=source_table, snapshot_date=snapshot_date, window=window)

        results = {
            "cluster" : self.cluster_dashboard (cursor, source_table, snapshot_date, window),
            "defect"  : self.defect_spike      (cursor, source_table, snapshot_date, window),
            "product" : self.product_dashboard (cursor, source_table, snapshot_date, window),
            "overview": self.overview_dashboard(cursor, source_table, snapshot_date, window),
        }

        logger.info("집계 완료", snapshot_date=snapshot_date,
                    **{k: len(v) for k, v in results.items()})
        return results


# ============================================================================
# 예시 실행 (__main__)
# ============================================================================

if __name__ == "__main__":
    import snowflake.connector
    from maude_early_alert.logging_config import configure_logging
    from maude_early_alert.utils.secrets import get_secret

    configure_logging(level="INFO")

    SOURCE_TABLE  = "EVENT_CLUSTERED"
    SNAPSHOT_DATE = None  # None → 오늘 날짜 자동 사용

    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret)
    cursor = conn.cursor()

    try:
        agg = Aggregation(database="MAUDE", schema="SILVER")
        dfs = agg.run(cursor, source_table=SOURCE_TABLE, snapshot_date=SNAPSHOT_DATE)

        for name, df in dfs.items():
            print(f"\n{'='*55}")
            print(f"[{name.upper()}]  {len(df):,}행")
            print(f"{'='*55}")
            print(df.head(10).to_string(index=False))

    finally:
        cursor.close()
        conn.close()

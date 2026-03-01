"""
이상징후 시그널 탐지 모듈
SpikeDetection: 계산 로직만 담당
실제 파이프라인 흐름은 __main__ 참고
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, List

from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier


class SpikeDetection(SnowflakeBase):
    """
    Python 기반 이상징후 시그널 탐지 (계산 전담)

    탐지 방법:
        1. Ratio   기반: score_ratio >= mean + z * std
        2. Z-score 기반: z_log >= threshold
        3. Poisson 기반: p_adjusted <= alpha

    앙상블 판정:
        severe(3) / alert(2) / attention(1) / general(0)
    """

    PATTERN_MAP = {3: "severe", 2: "alert", 1: "attention", 0: "general"}

    SPIKE_CONFIG = {
        "ratio":   ("IS_SPIKE",   "SCORE_RATIO"),
        "z":       ("IS_SPIKE_Z", "Z_LOG"),
        "poisson": ("IS_SPIKE_P", "SCORE_POIS"),
    }

    def __init__(self, database: str, schema: str):
        super().__init__(database, schema)

    # ──────────────────────────────────────────────
    # 1. Snowflake에서 월별 집계 읽기
    # ──────────────────────────────────────────────
    @with_context
    def fetch_monthly_counts(
        self,
        cursor: SnowflakeCursor,
        source_table: str,
        keyword_column: str,
        date_column: str,
        count_column: str,
        filters: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        validate_identifier(keyword_column)
        validate_identifier(date_column)
        validate_identifier(count_column)
        conditions = [
            f"{keyword_column} IS NOT NULL",
            f"{date_column} IS NOT NULL",
        ]
        if filters:
            conditions.insert(0, filters)
        where_clause = "WHERE " + "\n            AND ".join(conditions)

        keyword_sql = f"""
            SELECT
                {keyword_column}                      AS KEYWORD,
                TO_CHAR({date_column}, 'YYYY-MM')     AS MONTH,
                COUNT(DISTINCT {count_column})         AS N_REPORTS
            FROM {source_table}
            {where_clause}
            GROUP BY {keyword_column}, TO_CHAR({date_column}, 'YYYY-MM')
            ORDER BY KEYWORD, MONTH
        """
        cursor.execute(keyword_sql)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        keyword_monthly = pd.DataFrame(rows, columns=cols)

        total_conditions = [f"{date_column} IS NOT NULL"]
        if filters:
            total_conditions.insert(0, filters)
        total_where = "WHERE " + "\n            AND ".join(total_conditions)

        total_sql = f"""
            SELECT
                TO_CHAR({date_column}, 'YYYY-MM')  AS MONTH,
                COUNT(DISTINCT {count_column})      AS N_TOTAL
            FROM {source_table}
            {total_where}
            GROUP BY TO_CHAR({date_column}, 'YYYY-MM')
            ORDER BY MONTH
        """
        cursor.execute(total_sql)
        rows = cursor.fetchall()
        cols = [desc[0] for desc in cursor.description]
        monthly_total = pd.DataFrame(rows, columns=cols)

        return keyword_monthly, monthly_total

    # ──────────────────────────────────────────────
    # 2. 윈도우 월 리스트 생성
    # ──────────────────────────────────────────────
    def get_window_months(
        self,
        as_of_month: str,
        window: int,
    ) -> tuple[List[str], List[str]]:
        as_of_date = datetime.strptime(as_of_month, "%Y-%m")
        recent = [
            (as_of_date - relativedelta(months=i)).strftime("%Y-%m")
            for i in range(window)
        ]
        base = [
            (as_of_date - relativedelta(months=i)).strftime("%Y-%m")
            for i in range(window, window * 2)
        ]
        return recent, base

    # ──────────────────────────────────────────────
    # 3. 구간별 집계
    # ──────────────────────────────────────────────
    def aggregate_window(
        self,
        keyword_monthly: pd.DataFrame,
        monthly_total: pd.DataFrame,
        all_keywords: np.ndarray,
        recent_months: List[str],
        base_months: List[str],
    ) -> pd.DataFrame:
        km = keyword_monthly

        recent_df = (
            km[km["MONTH"].isin(recent_months)]
            .groupby("KEYWORD", as_index=False)
            .agg(C_RECENT=("N_REPORTS", "sum"), RECENT_MEAN=("N_REPORTS", "mean"))
        )

        base_filtered = km[km["MONTH"].isin(base_months)].copy()
        base_df = (
            base_filtered
            .groupby("KEYWORD", as_index=False)
            .agg(
                C_BASE=("N_REPORTS", "sum"),
                BASE_MEAN=("N_REPORTS", "mean"),
                BASE_STD=("N_REPORTS", "std"),
            )
        )

        base_filtered["_LOG"] = np.log(base_filtered["N_REPORTS"] + 1)
        log_stats = (
            base_filtered
            .groupby("KEYWORD", as_index=False)
            .agg(LOG_BASE_MEAN=("_LOG", "mean"), LOG_BASE_STD=("_LOG", "std"))
        )
        base_df = base_df.merge(log_stats, on="KEYWORD", how="left")

        mt = monthly_total
        n_recent = mt[mt["MONTH"].isin(recent_months)]["N_TOTAL"].sum()
        n_base   = mt[mt["MONTH"].isin(base_months)]["N_TOTAL"].sum()

        result = pd.DataFrame({"KEYWORD": all_keywords})
        result = result.merge(recent_df, on="KEYWORD", how="left")
        result = result.merge(base_df,   on="KEYWORD", how="left")
        result = result.fillna(0)
        result["N_RECENT"] = int(n_recent)
        result["N_BASE"]   = int(n_base)

        return result

    # ──────────────────────────────────────────────
    # 4. Ratio 지표 계산
    # ──────────────────────────────────────────────
    def calculate_ratio_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        r = (df["C_RECENT"] + 1) / (df["C_BASE"] + 1)
        df["RATIO"]       = r.round(2)
        df["SCORE_LOG"]   = (r * np.log(df["C_RECENT"] + 1)).round(2)
        df["SCORE_SQRT"]  = (r * np.sqrt(df["C_RECENT"])).round(2)
        df["SCORE_RATIO"] = (np.log(r) * np.log(df["C_RECENT"] + 1)).round(2)
        return df

    # ──────────────────────────────────────────────
    # 5. Z-score 지표 계산
    # ──────────────────────────────────────────────
    def calculate_zscore_metrics(self, df: pd.DataFrame, eps: float = 0.1) -> pd.DataFrame:
        df["Z_SCORE"] = (
            (df["RECENT_MEAN"] - df["BASE_MEAN"]) / (df["BASE_STD"] + eps)
        ).round(4)
        df["Z_LOG"] = (
            (np.log(df["C_RECENT"] + 1) - df["LOG_BASE_MEAN"]) / (df["LOG_BASE_STD"] + eps)
        ).round(4)
        return df

    # ──────────────────────────────────────────────
    # 6. 다중검정 보정 (내부 헬퍼)
    # ──────────────────────────────────────────────
    def _apply_correction(
        self,
        p_values: np.ndarray,
        method: Optional[str],
    ) -> np.ndarray:
        n = len(p_values)
        if method is None:
            return p_values
        elif method == "bonferroni":
            return np.minimum(p_values * n, 1.0)
        elif method == "sidak":
            return 1 - (1 - p_values) ** n
        elif method == "fdr_bh":
            sorted_idx = np.argsort(p_values)
            sorted_p   = p_values[sorted_idx]
            ranks      = np.arange(1, n + 1)
            adjusted   = np.minimum(sorted_p * n / ranks, 1.0)
            for i in range(n - 2, -1, -1):
                adjusted[i] = min(adjusted[i], adjusted[i + 1])
            result = np.empty(n)
            result[sorted_idx] = adjusted
            return result
        else:
            raise ValueError(f"Unknown correction method: {method}")

    # ──────────────────────────────────────────────
    # 7. Poisson 지표 계산
    # ──────────────────────────────────────────────
    def calculate_poisson_metrics(
        self,
        df: pd.DataFrame,
        alpha: float = 0.05,
        correction_method: Optional[str] = "fdr_bh",
    ) -> pd.DataFrame:
        c_recent = df["C_RECENT"].values
        n_base   = df["N_BASE"].values.astype(float)
        c_base   = df["C_BASE"].values.astype(float)
        n_recent = df["N_RECENT"].values.astype(float)

        lambda_pois = np.where(n_base > 0, (c_base / n_base) * n_recent, 0.001)
        lambda_pois = np.clip(lambda_pois, 0.001, None)

        p_pois = np.where(
            c_recent > 0,
            1 - stats.poisson.cdf(c_recent - 1, lambda_pois),
            1.0,
        )
        p_pois     = np.maximum(p_pois, 1e-300)
        p_adjusted = self._apply_correction(p_pois, correction_method)

        df["LAMBDA_POIS"] = np.round(lambda_pois, 6)
        df["P_POIS"]      = np.round(p_pois, 6)
        df["P_ADJUSTED"]  = np.round(p_adjusted, 6)
        df["SCORE_POIS"]  = np.round(-np.log10(p_pois), 4)
        return df

    # ──────────────────────────────────────────────
    # 8. 스파이크 판정
    # ──────────────────────────────────────────────
    def determine_spikes(
        self,
        df: pd.DataFrame,
        z_threshold: float = 2.0,
        min_c_recent: int = 20,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        increasing = df["C_RECENT"] > df["C_BASE"]
        enough     = df["C_RECENT"] >= min_c_recent

        ratio_thresholds = (
            df.groupby("WINDOW")["SCORE_RATIO"]
            .transform(lambda s: s.mean() + z_threshold * s.std())
        )

        df["IS_SPIKE"]   = increasing & enough & (df["SCORE_RATIO"] >= ratio_thresholds)
        df["IS_SPIKE_Z"] = increasing & enough & (df["Z_LOG"] >= z_threshold)
        df["IS_SPIKE_P"] = increasing & enough & (df["P_ADJUSTED"] <= alpha)
        return df

    # ──────────────────────────────────────────────
    # 9. 앙상블 판정
    # ──────────────────────────────────────────────
    def add_ensemble_pattern(self, df: pd.DataFrame) -> pd.DataFrame:
        spike_count = (
            df["IS_SPIKE"].astype(int)
            + df["IS_SPIKE_Z"].astype(int)
            + df["IS_SPIKE_P"].astype(int)
        )
        df["SPIKE_COUNT"] = spike_count
        df["PATTERN"]     = spike_count.map(self.PATTERN_MAP)
        return df

    # ──────────────────────────────────────────────
    # 10. 방법별 스파이크 필터링
    # ──────────────────────────────────────────────
    def detect_spikes(
        self,
        df: pd.DataFrame,
        window: int,
        spike_type: str = "ratio",
    ) -> pd.DataFrame:
        """특정 탐지 방법으로 스파이크만 반환.

        Args:
            df:         add_ensemble_pattern() 까지 완료된 combined DataFrame
            window:     필터링할 윈도우 (개월)
            spike_type: "ratio" | "z" | "poisson"
        """
        spike_col, sort_col = self.SPIKE_CONFIG[spike_type]
        return (
            df[(df["WINDOW"] == window) & df[spike_col]]
            .sort_values(sort_col, ascending=False)
            .reset_index(drop=True)
        )

    # ──────────────────────────────────────────────
    # 11. 앙상블 스파이크 필터링
    # ──────────────────────────────────────────────
    def detect_spike_ensemble(
        self,
        df: pd.DataFrame,
        window: int,
        min_methods: int = 2,
    ) -> pd.DataFrame:
        """앙상블 기준으로 스파이크만 반환.

        Args:
            df:          add_ensemble_pattern() 까지 완료된 combined DataFrame
            window:      필터링할 윈도우 (개월)
            min_methods: 최소 탐지 방법 수 (2 → alert+severe, 3 → severe만)
        """
        mask = (df["WINDOW"] == window) & (df["SPIKE_COUNT"] >= min_methods)
        return (
            df[mask]
            .sort_values("SCORE_POIS", ascending=False)
            .reset_index(drop=True)
        )


# ============================================================================
# 예시 실행 (__main__)
# 실제 파이프라인에서는 아래 흐름을 pipeline 파일에서 구현
# ============================================================================

if __name__ == "__main__":
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret

    # ── 설정 ────────────────────────────────────────
    SOURCE_TABLE   = "EVENT_STAGE_12_COMBINED"                  # 분석 대상 테이블
    KEYWORD_COLUMN = "DEFECT_TYPE"                              # 스파이크를 감지할 범주형 컬럼
    DATE_COLUMN    = "DATE_RECEIVED"                            # 월별 집계 기준 날짜 컬럼
    COUNT_COLUMN   = "MDR_REPORT_KEY"                           # 건수 집계 기준 컬럼 (COUNT DISTINCT)
    FILTERS        = "DEFECT_TYPE NOT IN ('Unknown', 'Other')"  # WHERE 조건. None = 전체
    WINDOWS        = [12]                                       # 비교 윈도우 크기(월). 복수 가능: [3, 6, 12]
    AS_OF_MONTH    = None                                       # 기준 월 "YYYY-MM". None = 자동으로 최신 월
    Z_THRESHOLD    = 2.0                                        # Z-score 스파이크 임계값. 높을수록 엄격
    MIN_C_RECENT   = 20                                         # recent 기간 최소 건수. 미만이면 스파이크 제외
    EPS            = 0.1                                        # Z-score 분모 안정화 상수 (log 0 방지)
    ALPHA          = 0.05                                       # 유의수준. Poisson p-value 임계값
    CORRECTION     = "fdr_bh"                                   # 다중검정 보정 방식
                                                                #   "fdr_bh"    : Benjamini-Hochberg (기본, 권장)
                                                                #   "fdr_by"    : Benjamini-Yekutieli (변수 간 상관 있을 때)
                                                                #   "bonferroni": 가장 보수적, 오탐 최소화
                                                                #   "holm"      : bonferroni보다 약간 관대

    # ── Snowflake 연결 ───────────────────────────────
    secret = get_secret("snowflake/de")
    conn = snowflake.connector.connect(
        user=secret["user"],
        password=secret["password"],
        account=secret["account"],
        warehouse=secret["warehouse"],
    )
    cursor = conn.cursor()

    try:
        detector = SpikeDetection(database="MAUDE", schema="SILVER")
        timings  = {}

        # 1. fetch
        t0 = time.time()
        keyword_monthly, monthly_total = detector.fetch_monthly_counts(
            cursor, SOURCE_TABLE, KEYWORD_COLUMN, DATE_COLUMN, COUNT_COLUMN, FILTERS
        )
        timings["fetch"] = time.time() - t0
        print(f"[fetch] 키워드 {keyword_monthly['KEYWORD'].nunique()}개, "
              f"월 {monthly_total['MONTH'].nunique()}개 ({timings['fetch']:.2f}s)")

        # 2. 기준 월 결정
        if AS_OF_MONTH is None:
            AS_OF_MONTH = monthly_total["MONTH"].sort_values(ascending=False).iloc[0]
        print(f"[config] 기준 월: {AS_OF_MONTH}, 윈도우: {WINDOWS}")

        # 3. 윈도우별 집계
        t0 = time.time()
        all_keywords = keyword_monthly["KEYWORD"].unique()
        results = []

        for w in WINDOWS:
            recent_months, base_months = detector.get_window_months(AS_OF_MONTH, w)
            print(f"  Window {w}: recent={recent_months}, base={base_months}")
            df = detector.aggregate_window(
                keyword_monthly, monthly_total, all_keywords, recent_months, base_months
            )
            df["AS_OF_MONTH"] = AS_OF_MONTH
            df["WINDOW"]      = w
            results.append(df)

        combined = pd.concat(results, ignore_index=True)
        timings["aggregate"] = time.time() - t0

        # 4. 지표 계산
        t0 = time.time()
        combined = detector.calculate_ratio_metrics(combined)
        combined = detector.calculate_zscore_metrics(combined, EPS)
        combined = detector.calculate_poisson_metrics(combined, ALPHA, CORRECTION)
        timings["metrics"] = time.time() - t0

        # 5. 스파이크 판정
        t0 = time.time()
        combined = detector.determine_spikes(combined, Z_THRESHOLD, MIN_C_RECENT, ALPHA)
        combined = detector.add_ensemble_pattern(combined)
        timings["spikes"] = time.time() - t0

        # 6. 컬럼 정리
        col_order = [
            "AS_OF_MONTH", "WINDOW", "KEYWORD",
            "C_RECENT", "C_BASE", "N_RECENT", "N_BASE",
            "RECENT_MEAN", "BASE_MEAN", "BASE_STD",
            "RATIO", "SCORE_LOG", "SCORE_SQRT", "SCORE_RATIO",
            "LOG_BASE_MEAN", "LOG_BASE_STD", "Z_SCORE", "Z_LOG",
            "LAMBDA_POIS", "P_POIS", "P_ADJUSTED", "SCORE_POIS",
            "SPIKE_COUNT", "PATTERN",
            "IS_SPIKE", "IS_SPIKE_Z", "IS_SPIKE_P",
        ]
        existing = [c for c in col_order if c in combined.columns]
        combined = combined[existing].sort_values(["KEYWORD", "WINDOW"]).reset_index(drop=True)
        # 7. 결과 출력
        timings["total"] = sum(timings.values())
        print(f"\n{'='*55}")
        print("스파이크 탐지 결과")
        print(f"{'='*55}")
        for w in WINDOWS:
            w_df = combined[combined["WINDOW"] == w]
            print(f"  Window {w}: "
                  f"severe={int((w_df['PATTERN'] == 'severe').sum())}, "
                  f"alert={int((w_df['PATTERN'] == 'alert').sum())}, "
                  f"attention={int((w_df['PATTERN'] == 'attention').sum())}")
        print("\n[소요 시간]")
        for step, sec in timings.items():
            print(f"  {step:20} -> {sec:.3f}s")

        # 8. 앙상블 스파이크 상세 출력
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", None)
        display_cols = ["KEYWORD", "WINDOW", "PATTERN", "C_RECENT", "C_BASE",
                        "RATIO", "Z_LOG", "P_ADJUSTED", "SPIKE_COUNT"]

        for w in WINDOWS:
            ensemble = detector.detect_spike_ensemble(combined, window=w, min_methods=2)
            print(f"\n[결과] Window {w}: baseline={len(combined[combined['WINDOW']==w])}건, "
                  f"spikes(alert+severe)={len(ensemble)}건")
            if len(ensemble) > 0:
                print(f"\n{'='*55}")
                print(f"앙상블 스파이크 상세 (Window {w})")
                print(f"{'='*55}")
                existing_cols = [c for c in display_cols if c in ensemble.columns]
                print(ensemble[existing_cols].to_string(index=False))

    finally:
        cursor.close()
        conn.close()
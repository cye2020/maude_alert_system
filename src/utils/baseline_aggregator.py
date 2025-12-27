import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Literal, Union
import numpy as np
from scipy import stats


class BaselineAggregator:
    """
    키워드별 베이스라인 집계 및 스파이크 탐지를 위한 클래스
    
    최근 구간과 기준 구간을 비교하여 키워드별 보고서 수의 변화를 분석합니다.
    """
    
    # 클래스 상수
    PATTERN_MAP = {3: "severe", 2: "alert", 1: "attention", 0: "general"}
    SPIKE_CONFIG = {
        "ratio": ("is_spike", "score_ratio"),
        "z": ("is_spike_z", "z_log"),
        "poisson": ("is_spike_p", "score_pois")
    }
    
    def __init__(self, lf: pl.LazyFrame):
        """
        Parameters
        ----------
        lf : pl.LazyFrame
            원본 보고서 데이터 (필수 컬럼: mdr_report_key, date_received, defect_type)
        """
        self.lf = lf
        self._keyword_monthly: Optional[pl.DataFrame] = None
        self._monthly_total: Optional[pl.DataFrame] = None
        self._as_of_month: Optional[str] = None
        
    def _prepare_monthly_data(self) -> None:
        """월별 집계 데이터를 준비합니다."""
        if self._keyword_monthly is not None:
            return  # 이미 준비된 경우 스킵
            
        lf_with_month = self.lf.with_columns(
            pl.col("date_received").cast(pl.Date).dt.strftime("%Y-%m").alias("month")
        )
        
        # 키워드별 월별 집계 (한 번에 collect)
        self._keyword_monthly = (
            lf_with_month
            .filter(pl.col("defect_type").is_not_null())
            .group_by(["defect_type", "month"])
            .agg(pl.col("mdr_report_key").n_unique().alias("n_reports"))
            .rename({"defect_type": "keyword"})
            .sort(["keyword", "month"])
            .collect()
        )
        
        # 월별 전체 보고서 수 집계
        self._monthly_total = (
            lf_with_month
            .filter(pl.col("date_received").is_not_null())
            .group_by("month")
            .agg(pl.col("mdr_report_key").n_unique().alias("n_total"))
            .sort("month")
            .collect()
        )
    
    def _get_latest_month(self) -> str:
        """가장 최신 월을 반환합니다."""
        months = self._monthly_total["month"].sort(descending=True)
        if months.is_empty():
            raise ValueError("사용 가능한 월 데이터가 없습니다.")
        return months[0]
    
    def _get_window_months(self, as_of_month: str, window: int) -> tuple[list[str], list[str]]:
        """
        윈도우별 recent/base 월 리스트를 반환합니다.
        
        기준 구간(base)은 최근 구간(recent)과 겹치지 않도록 설정합니다.
        - window=1: recent=[당월], base=[전월]
        - window=3: recent=[당월, -1, -2], base=[-3, -4, -5] (겹치지 않음)
        """
        as_of_date = datetime.strptime(as_of_month, "%Y-%m")
        
        if window == 1:
            recent = [as_of_month]
            base = [(as_of_date - relativedelta(months=1)).strftime("%Y-%m")]
        else:  # window == 3
            recent = [(as_of_date - relativedelta(months=i)).strftime("%Y-%m") for i in range(3)]
            # base는 recent 직전 3개월 (겹치지 않음)
            base = [(as_of_date - relativedelta(months=i)).strftime("%Y-%m") for i in range(3, 6)]
        
        return recent, base
    
    def _aggregate_window(
        self, 
        keywords: pl.Series,
        recent_months: list[str], 
        base_months: list[str]
    ) -> pl.DataFrame:
        """윈도우별 집계를 수행합니다."""
        # Recent/Base 필터링 및 집계
        recent_df = (
            self._keyword_monthly
            .filter(pl.col("month").is_in(recent_months))
            .group_by("keyword")
            .agg([
                pl.col("n_reports").sum().alias("C_recent"),
                pl.col("n_reports").mean().alias("recent_mean")
            ])
        )
        
        base_df = (
            self._keyword_monthly
            .filter(pl.col("month").is_in(base_months))
            .group_by("keyword")
            .agg([
                pl.col("n_reports").sum().alias("C_base"),
                pl.col("n_reports").mean().alias("base_mean"),
                pl.col("n_reports").std().alias("base_std"),
                (pl.col("n_reports") + 1).log().mean().alias("log_base_mean"),
                (pl.col("n_reports") + 1).log().std().alias("log_base_std")
            ])
        )
        
        # 전체 보고서 수
        N_recent = (
            self._monthly_total
            .filter(pl.col("month").is_in(recent_months))
            ["n_total"].sum()
        )
        N_base = (
            self._monthly_total
            .filter(pl.col("month").is_in(base_months))
            ["n_total"].sum()
        )
        
        # 모든 키워드와 조인
        result = (
            pl.DataFrame({"keyword": keywords})
            .join(recent_df, on="keyword", how="left")
            .join(base_df, on="keyword", how="left")
            .with_columns([
                pl.lit(N_recent).alias("N_recent"),
                pl.lit(N_base).alias("N_base")
            ])
            .fill_null(0)
        )
        
        return result
    
    def _calculate_ratio_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        """비율 기반 지표를 계산합니다."""
        return df.with_columns([
            # ratio
            ((pl.col("C_recent") + 1) / (pl.col("C_base") + 1)).round(2).alias("ratio"),
            # scores
            (((pl.col("C_recent") + 1) / (pl.col("C_base") + 1)) * (pl.col("C_recent") + 1).log()).round(2).alias("score_log"),
            (((pl.col("C_recent") + 1) / (pl.col("C_base") + 1)) * pl.col("C_recent").sqrt()).round(2).alias("score_sqrt"),
            (((pl.col("C_recent") + 1) / (pl.col("C_base") + 1)).log() * (pl.col("C_recent") + 1).log()).round(2).alias("score_ratio")
        ])
    
    def _calculate_zscore_metrics(self, df: pl.DataFrame, eps: float) -> pl.DataFrame:
        """Z-score 지표를 계산합니다."""
        return df.with_columns([
            # 일반 z-score
            ((pl.col("recent_mean") - pl.col("base_mean")) / (pl.col("base_std") + eps)).round(4).alias("z_score"),
            # log 변환 z-score
            (((pl.col("C_recent") + 1).log() - pl.col("log_base_mean")) / (pl.col("log_base_std") + eps)).round(4).alias("z_log")
        ])
    
    def _calculate_poisson_metrics(
        self, 
        df: pl.DataFrame, 
        alpha: float,
        correction_method: Optional[str]
    ) -> pl.DataFrame:
        """Poisson 기반 지표를 계산합니다."""
        # Lambda 계산
        df = df.with_columns(
            pl.when(pl.col("N_base") > 0)
            .then((pl.col("C_base") / pl.col("N_base")) * pl.col("N_recent"))
            .otherwise(0.001)
            .clip(lower_bound=0.001)
            .alias("lambda_pois")
        )
        
        # p-value 계산 (numpy/scipy)
        c_recent = df["C_recent"].to_numpy()
        lambda_pois = df["lambda_pois"].to_numpy()
        
        p_pois = np.where(
            c_recent > 0,
            1 - stats.poisson.cdf(c_recent - 1, lambda_pois),
            1.0
        )
        p_pois = np.maximum(p_pois, 1e-300)
        
        # 다중검정 보정
        p_adjusted = self._apply_correction(p_pois, correction_method)
        
        return df.with_columns([
            pl.Series("p_pois", p_pois).round(6),
            pl.Series("p_adjusted", p_adjusted).round(6),
            pl.Series("score_pois", -np.log10(p_pois)).round(4)
        ])
    
    def _apply_correction(self, p_values: np.ndarray, method: Optional[str]) -> np.ndarray:
        """다중검정 보정을 적용합니다."""
        n = len(p_values)
        
        if method is None:
            return p_values
        elif method == "bonferroni":
            return np.minimum(p_values * n, 1.0)
        elif method == "sidak":
            return 1 - (1 - p_values) ** n
        elif method == "fdr_bh":
            # Benjamini-Hochberg
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]
            ranks = np.arange(1, n + 1)
            adjusted = np.minimum(sorted_p * n / ranks, 1.0)
            
            # Cumulative minimum (뒤에서부터)
            for i in range(n - 2, -1, -1):
                adjusted[i] = min(adjusted[i], adjusted[i + 1])
            
            result = np.empty(n)
            result[sorted_idx] = adjusted
            return result
        else:
            raise ValueError(f"Unknown correction method: {method}")
    
    def _determine_spikes(
        self, 
        df: pl.DataFrame, 
        z_threshold: float, 
        min_c_recent: int,
        alpha: float
    ) -> pl.DataFrame:
        """스파이크 여부를 판정합니다."""
        # score_ratio 통계 (윈도우별)
        stats_df = df.group_by("window").agg([
            pl.col("score_ratio").mean().alias("_mean"),
            pl.col("score_ratio").std().alias("_std")
        ])
        
        df = df.join(stats_df, on="window", how="left")
        
        return df.with_columns([
            # is_spike (ratio 기반) - 증가만 탐지
            (
                (pl.col("C_recent") > pl.col("C_base")) &  # 증가 조건 추가
                (pl.col("score_ratio") >= (pl.col("_mean") + z_threshold * pl.col("_std"))) &
                (pl.col("C_recent") >= min_c_recent)
            ).alias("is_spike"),

            # is_spike_z (z-score 기반) - 증가만 탐지
            (
                (pl.col("C_recent") > pl.col("C_base")) &  # 증가 조건 추가
                (pl.col("z_log") >= z_threshold) &
                (pl.col("C_recent") >= min_c_recent)
            ).alias("is_spike_z"),

            # is_spike_p (Poisson 기반) - 증가만 탐지
            (
                (pl.col("C_recent") > pl.col("C_base")) &  # 증가 조건 추가
                (pl.col("p_adjusted") <= alpha) &
                (pl.col("C_recent") >= min_c_recent)
            ).alias("is_spike_p")
        ]).drop(["_mean", "_std"])
    
    def _add_ensemble_results(self, df: pl.DataFrame) -> pl.DataFrame:
        """앙상블 결과(pattern)를 추가합니다.

        증가하지 않은 키워드(is_spike + is_spike_z + is_spike_p == 0)는
        패턴을 null로 설정합니다.
        """
        return df.with_columns(
            pl.when(
                pl.col("is_spike").cast(pl.Int8) +
                pl.col("is_spike_z").cast(pl.Int8) +
                pl.col("is_spike_p").cast(pl.Int8) == 3
            ).then(pl.lit("severe"))
            .when(
                pl.col("is_spike").cast(pl.Int8) +
                pl.col("is_spike_z").cast(pl.Int8) +
                pl.col("is_spike_p").cast(pl.Int8) == 2
            ).then(pl.lit("alert"))
            .when(
                pl.col("is_spike").cast(pl.Int8) +
                pl.col("is_spike_z").cast(pl.Int8) +
                pl.col("is_spike_p").cast(pl.Int8) == 1
            ).then(pl.lit("attention"))
            .when(
                pl.col("is_spike").cast(pl.Int8) +
                pl.col("is_spike_z").cast(pl.Int8) +
                pl.col("is_spike_p").cast(pl.Int8) == 0
            ).then(pl.lit(None))  # 패턴 없음
            .otherwise(pl.lit("general"))  # 혹시 모를 예외 케이스
            .alias("pattern")
        )
    
    def _reorder_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """컬럼 순서를 정리합니다."""
        col_order = [
            "as_of_month", "window", "keyword",
            "C_recent", "C_base", "N_recent", "N_base",
            "recent_mean", "base_mean", "base_std",
            "ratio", "score_log", "score_sqrt", "score_ratio",
            "log_base_mean", "log_base_std", "z_score", "z_log",
            "lambda_pois", "p_pois", "p_adjusted", "score_pois",
            "pattern",
            "is_spike", "is_spike_z", "is_spike_p"
        ]
        # 존재하는 컬럼만 선택
        existing = [c for c in col_order if c in df.columns]
        return df.select(existing)
    
    def create_baseline_table(
        self,
        as_of_month: Optional[str] = None,
        z_threshold: float = 2.0,
        min_c_recent: int = 20,
        eps: float = 0.1,
        alpha: float = 0.05,
        correction_method: Optional[Literal["bonferroni", "sidak", "fdr_bh"]] = "fdr_bh",
        verbose: bool = True
    ) -> pl.LazyFrame:
        """
        베이스라인 집계 테이블을 생성합니다.
        
        Parameters
        ----------
        as_of_month : str, optional
            기준 월 (예: "2024-11"). None이면 최신 월 사용
        z_threshold : float, default=2.0
            스파이크 판단 z-score 임계값
        min_c_recent : int, default=20
            스파이크 판단 최소 C_recent 값
        eps : float, default=0.1
            z-score 분모 보정값
        alpha : float, default=0.05
            Poisson 유의수준
        correction_method : str or None, default="fdr_bh"
            다중검정 보정법
        verbose : bool, default=True
            진행 상황 출력
            
        Returns
        -------
        pl.LazyFrame
            베이스라인 집계 테이블
        """
        # 1. 데이터 준비
        if verbose:
            print("월별 집계 데이터 준비 중...")
        self._prepare_monthly_data()
        
        if as_of_month is None:
            as_of_month = self._get_latest_month()
        self._as_of_month = as_of_month
        
        if verbose:
            print(f"기준 월: {as_of_month}")
        
        # 2. 전체 키워드 목록
        all_keywords = self._keyword_monthly["keyword"].unique()
        
        # 3. 윈도우별 집계 및 통합
        results = []
        for window in [1, 3]:
            recent_months, base_months = self._get_window_months(as_of_month, window)
            
            if verbose:
                print(f"Window {window}: recent={recent_months}, base={base_months}")
            
            df = self._aggregate_window(all_keywords, recent_months, base_months)
            df = df.with_columns([
                pl.lit(as_of_month).alias("as_of_month"),
                pl.lit(window).alias("window")
            ])
            results.append(df)
        
        # 4. 통합
        combined = pl.concat(results)
        
        # 5. 지표 계산
        if verbose:
            print("지표 계산 중...")
        
        combined = self._calculate_ratio_metrics(combined)
        combined = self._calculate_zscore_metrics(combined, eps)
        combined = self._calculate_poisson_metrics(combined, alpha, correction_method)
        
        # 6. 스파이크 판정
        combined = self._determine_spikes(combined, z_threshold, min_c_recent, alpha)
        
        # 7. 앙상블 결과
        combined = self._add_ensemble_results(combined)
        
        # 8. 컬럼 정리
        combined = self._reorder_columns(combined)
        combined = combined.sort(["keyword", "window"])
        
        if verbose:
            n_keywords = combined["keyword"].n_unique()
            spike_w1 = combined.filter(
                (pl.col("window") == 1) & 
                (pl.col("pattern").is_in(["severe", "alert"]))
            ).height
            spike_w3 = combined.filter(
                (pl.col("window") == 3) & 
                (pl.col("pattern").is_in(["severe", "alert"]))
            ).height
            print(f"\n완료: 키워드 {n_keywords}개, 스파이크(2+) W1={spike_w1}, W3={spike_w3}")
        
        return combined.lazy()
    
    def detect_spikes(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1,
        spike_type: Literal["ratio", "z", "poisson"] = "ratio"
    ) -> pl.LazyFrame:
        """특정 방법으로 탐지된 스파이크만 반환합니다."""
        spike_col, sort_col = self.SPIKE_CONFIG[spike_type]
        
        return (
            baseline_lf
            .filter((pl.col("window") == window) & pl.col(spike_col))
            .sort(sort_col, descending=True)
        )
    
    def detect_spike_ensemble(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1,
        min_methods: int = 2
    ) -> pl.LazyFrame:
        """앙상블 기준 스파이크만 반환합니다."""
        return (
            baseline_lf
            .filter(pl.col("window") == window)
            .filter(
                (
                    pl.col("is_spike").cast(pl.Int8) +
                    pl.col("is_spike_z").cast(pl.Int8) +
                    pl.col("is_spike_p").cast(pl.Int8)
                ) >= min_methods
            )
            .sort("score_pois", descending=True)
        )
import polars as pl
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Optional, Tuple, List, Literal, Union


class BaselineAggregator:
    """
    키워드별 베이스라인 집계 및 스파이크 탐지를 위한 클래스
    
    최근 구간과 기준 구간을 비교하여 키워드별 보고서 수의 변화를 분석합니다.
    """
    
    def __init__(self, lf: pl.LazyFrame):
        """
        Parameters
        ----------
        lf : pl.LazyFrame
            원본 보고서 데이터 (필수 컬럼: mdr_report_key, date_received, defect_type)
        """
        self.lf = lf
        self._lf_keyword_monthly: Optional[pl.LazyFrame] = None
        self._lf_monthly_total: Optional[pl.LazyFrame] = None
        
    def _prepare_monthly_data(self) -> None:
        """월별 집계 데이터를 준비합니다."""
        lf_with_month = self.lf.with_columns(
            pl.col("date_received").cast(pl.Date).dt.strftime("%Y-%m").alias("as_of_month")
        )
        
        # 키워드 테이블 생성
        lf_keyword_table = (
            lf_with_month
            .filter(pl.col("defect_type").is_not_null())
            .select(
                pl.col("mdr_report_key"),
                pl.col("defect_type").alias("keyword"),
                pl.col("as_of_month").alias("month")
            )
            .unique(subset=["mdr_report_key", "keyword", "month"])
        )
        
        # 월별 키워드별 집계
        self._lf_keyword_monthly = (
            lf_keyword_table
            .group_by(["keyword", "month"])
            .agg(pl.col("mdr_report_key").n_unique().alias("n_reports_keyword"))
            .sort(["keyword", "month"])
        )
        
        # 월별 전체 보고서 수 집계
        self._lf_monthly_total = (
            self.lf
            .filter(pl.col("date_received").is_not_null())
            .with_columns(
                pl.col("date_received").cast(pl.Date).dt.strftime("%Y-%m").alias("month")
            )
            .group_by("month")
            .agg(pl.len().alias("n_total_reports"))
            .sort("month")
        )
    
    def _get_latest_month(self) -> str:
        """가장 최신 월을 반환합니다."""
        available_months = (
            self._lf_monthly_total
            .select("month")
            .sort("month", descending=True)
            .collect()
            .to_series()
            .to_list()
        )
        if not available_months:
            raise ValueError("사용 가능한 월 데이터가 없습니다.")
        return available_months[0]
    
    def _calculate_time_windows(
        self, 
        as_of_month: str
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        최근 구간과 기준 구간의 월 리스트를 계산합니다.
        
        Returns
        -------
        tuple
            (recent_1month, base_1month, recent_3month, base_3month)
        """
        as_of_date = datetime.strptime(as_of_month, "%Y-%m")
        
        recent_1month = [as_of_month]
        recent_3month = [
            (as_of_date - relativedelta(months=i)).strftime("%Y-%m")
            for i in range(3)
        ]
        base_1month = [(as_of_date - relativedelta(months=1)).strftime("%Y-%m")]
        base_3month = [
            (as_of_date - relativedelta(months=i)).strftime("%Y-%m")
            for i in range(1, 4)
        ]
        
        return recent_1month, base_1month, recent_3month, base_3month
    
    def _aggregate_keyword_window(
        self,
        recent_months: List[str],
        base_months: List[str],
        all_keywords: pl.DataFrame,
        window: int
    ) -> pl.DataFrame:
        """특정 윈도우에 대해 키워드별 보고서 수를 집계합니다."""
        recent_df = (
            self._lf_keyword_monthly
            .filter(pl.col("month").is_in(recent_months))
            .group_by("keyword")
            .agg(pl.col("n_reports_keyword").sum().alias(f"C_recent_{window}m"))
            .collect()
        )
        
        base_df = (
            self._lf_keyword_monthly
            .filter(pl.col("month").is_in(base_months))
            .group_by("keyword")
            .agg(pl.col("n_reports_keyword").sum().alias(f"C_base_{window}m"))
            .collect()
        )
        
        return (
            all_keywords
            .join(recent_df, on="keyword", how="left")
            .join(base_df, on="keyword", how="left")
            .with_columns(
                pl.col(f"C_recent_{window}m").fill_null(0),
                pl.col(f"C_base_{window}m").fill_null(0)
            )
        )
    
    def _aggregate_total_window(
        self,
        recent_months: List[str],
        base_months: List[str]
    ) -> Tuple[int, int]:
        """특정 윈도우에 대해 전체 보고서 수를 집계합니다."""
        N_recent = (
            self._lf_monthly_total
            .filter(pl.col("month").is_in(recent_months))
            .select(pl.sum('n_total_reports'))
            .collect()
            .item()
        ) or 0
        
        N_base = (
            self._lf_monthly_total
            .filter(pl.col("month").is_in(base_months))
            .select(pl.sum('n_total_reports'))
            .collect()
            .item()
        ) or 0
        
        return N_recent, N_base
    
    def _calculate_metrics(
        self,
        df: pl.DataFrame,
        z_threshold: float,
        min_c_recent: int,
        eps: float,
        alpha: float,
        correction_method: Optional[str]
    ) -> pl.DataFrame:
        """
        ratio, score, z_log, Poisson 등의 지표를 계산합니다.
        
        Parameters
        ----------
        alpha : float
            유의수준 (예: 0.05, 0.01, 0.001)
        correction_method : str or None
            다중검정 보정법
            - None: 보정 없음 (raw p-value 사용)
            - "bonferroni": Bonferroni 보정
            - "sidak": Šidák 보정
            - "fdr_bh": Benjamini-Hochberg FDR
        """
        from scipy import stats
        from scipy.stats import false_discovery_control
        import numpy as np
        
        result = (
            df
            # ratio 계산
            .with_columns(
                ((pl.col("C_recent") + 1) / (pl.col("C_base") + 1))
                .round(2)
                .alias("ratio")
            )
            # score 계산
            .with_columns(
                # log 가중
                (pl.col("ratio") * (pl.col("C_recent") + 1).log())
                .round(2)
                .alias("score_log"),
                # sqrt 가중
                (pl.col("ratio") * pl.col("C_recent").sqrt())
                .round(2)
                .alias("score_sqrt"),
                # 상대적 증가율
                (pl.col("ratio").log() * (pl.col("C_recent") + 1).log())
                .round(2)
                .alias("score_ratio")
            )
            # z_log 계산
            .with_columns(
                (pl.col("C_recent") + 1).log().alias("log_recent"),
                (pl.col("C_base") + 1).log().alias("log_base_mean")
            )
            .with_columns(
                ((pl.col("log_recent") - pl.col("log_base_mean")) / eps)
                .round(4)
                .alias("z_log")
            )
            # is_spike_z: z_log 기반 스파이크 판단
            .with_columns(
                (
                    (pl.col("z_log") >= z_threshold) &
                    (pl.col("C_recent") >= min_c_recent)
                ).alias("is_spike_z")
            )
            # is_spike: score_ratio 기반 스파이크 판단
            .with_columns(
                pl.col("score_ratio").mean().over("window").alias("_score_ratio_mean"),
                pl.col("score_ratio").std().over("window").alias("_score_ratio_std")
            )
            .with_columns(
                (
                    (pl.col("score_ratio") >= 
                     (pl.col("_score_ratio_mean") + z_threshold * pl.col("_score_ratio_std"))) &
                    (pl.col("C_recent") >= min_c_recent)
                ).alias("is_spike")
            )
            .drop(["_score_ratio_mean", "_score_ratio_std"])
            # Poisson lambda 계산: λ = (C_base / N_base) * N_recent
            .with_columns(
                pl.when(pl.col("N_base") > 0)
                .then((pl.col("C_base") / pl.col("N_base")) * pl.col("N_recent"))
                .otherwise(pl.col("C_base").cast(pl.Float64))
                .alias("lambda_pois")
            )
            .with_columns(
                pl.when(pl.col("lambda_pois") < 0.001)
                .then(0.001)
                .otherwise(pl.col("lambda_pois"))
                .alias("lambda_pois")
            )
        )
        
        # Poisson p-value 계산
        c_recent = result["C_recent"].to_numpy()
        lambda_pois = result["lambda_pois"].to_numpy()
        
        # P(X >= k | λ) = 1 - P(X <= k-1 | λ)
        p_pois = np.where(
            c_recent > 0,
            1 - stats.poisson.cdf(c_recent - 1, lambda_pois),
            1.0
        )
        p_pois = np.maximum(p_pois, 1e-300)
        score_pois = -np.log10(p_pois)
        
        # 다중검정 보정
        n = len(p_pois)
        if correction_method is None:
            # 보정 없음
            p_adjusted = p_pois
            alpha_adjusted = alpha
        elif correction_method == "bonferroni":
            # Bonferroni: α / n
            p_adjusted = np.minimum(p_pois * n, 1.0)
            alpha_adjusted = alpha
        elif correction_method == "sidak":
            # Šidák: 1 - (1-α)^(1/n)
            p_adjusted = 1 - (1 - p_pois) ** n
            alpha_adjusted = alpha
        elif correction_method == "fdr_bh":
            # Benjamini-Hochberg FDR
            # scipy의 false_discovery_control 사용
            reject = false_discovery_control(p_pois, axis=0, method='bh')
            # FDR에서는 adjusted p-value 대신 reject 여부를 직접 반환
            p_adjusted = p_pois  # raw p-value 유지
            # is_spike_p는 아래에서 별도 처리
        else:
            raise ValueError(f"Unknown correction method: {correction_method}")
        
        # 결과 추가
        result = result.with_columns(
            pl.Series("p_pois", p_pois).round(6),
            pl.Series("p_adjusted", p_adjusted).round(6),
            pl.Series("score_pois", score_pois).round(4)
        )
        
        # is_spike_p 계산
        if correction_method == "fdr_bh":
            # FDR: scipy의 reject 결과 사용 + min_c_recent 필터
            is_spike_p = reject & (c_recent >= min_c_recent)
            result = result.with_columns(
                pl.Series("is_spike_p", is_spike_p)
            )
        else:
            # 다른 방법: adjusted p-value와 alpha 비교
            result = result.with_columns(
                (
                    (pl.col("p_adjusted") <= alpha) &
                    (pl.col("C_recent") >= min_c_recent)
                ).alias("is_spike_p")
            )
        
        # 컬럼 순서 재배치: is_spike 시리즈를 마지막으로
        spike_cols = ["is_spike", "is_spike_z", "is_spike_p"]
        other_cols = [c for c in result.columns if c not in spike_cols]
        result = result.select(other_cols + spike_cols)
        
        return result.sort(["keyword", "window"])
    
    def _create_window_baseline(
        self,
        keyword_df: pl.DataFrame,
        as_of_month: str,
        window: int,
        N_recent: int,
        N_base: int,
        z_threshold: float,
        min_c_recent: int,
        eps: float,
        alpha: float,
        correction_method: Optional[str]
    ) -> pl.DataFrame:
        """
        윈도우별 베이스라인 데이터프레임을 생성합니다.
        
        모든 지표(ratio, score, z_log, Poisson, is_spike 등)를 포함합니다.
        """
        c_recent_col = f"C_recent_{window}m"
        c_base_col = f"C_base_{window}m"
        
        base_df = (
            keyword_df
            .with_columns(
                pl.lit(as_of_month).alias("as_of_month"),
                pl.lit(window).alias("window"),
                pl.col(c_recent_col).fill_null(0).cast(pl.Int64).alias("C_recent"),
                pl.col(c_base_col).fill_null(0).cast(pl.Int64).alias("C_base"),
                pl.lit(N_recent).cast(pl.Int64).alias("N_recent"),
                pl.lit(N_base).cast(pl.Int64).alias("N_base")
            )
            .select([
                "as_of_month", "window", "keyword",
                "C_recent", "C_base", "N_recent", "N_base"
            ])
        )
        
        # 지표 계산 적용
        return self._calculate_metrics(
            base_df, z_threshold, min_c_recent, eps, alpha, correction_method
        )
    
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
        베이스라인 집계 테이블을 Long Format으로 생성합니다.
        
        Parameters
        ----------
        as_of_month : str, optional
            기준 월 (예: "2025-11"). None이면 자동으로 최신 월 사용
        z_threshold : float, default=2.0
            스파이크 판단 기준 z-score 임계값
        min_c_recent : int, default=20
            스파이크 판단 최소 C_recent 값
        eps : float, default=0.1
            z_log 계산 시 분모에 더할 작은 값
        alpha : float, default=0.05
            Poisson 스파이크 판단 유의수준
            - 0.05: 일반적 유의수준 (*)
            - 0.01: 매우 유의 (**)
            - 0.001: 고도로 유의 (***)
        correction_method : str or None, default="fdr_bh"
            다중검정 보정법
            - None: 보정 없음 (raw p-value 사용)
            - "bonferroni": Bonferroni 보정 (가장 보수적)
            - "sidak": Šidák 보정
            - "fdr_bh": Benjamini-Hochberg FDR (권장, 거짓발견율 제어)
        verbose : bool, default=True
            진행 상황 출력 여부
            
        Returns
        -------
        pl.LazyFrame
            Long Format 베이스라인 집계 테이블
            컬럼: as_of_month, window(int), keyword, C_recent, C_base, N_recent, N_base,
                  ratio, score_log, score_sqrt, score_ratio, 
                  log_recent, log_base_mean, z_log, is_spike_z, is_spike,
                  lambda_pois, p_pois, p_adjusted, score_pois, is_spike_p
        """
        if verbose:
            print("월별 집계 데이터 준비 중...")
        self._prepare_monthly_data()
        
        if as_of_month is None:
            as_of_month = self._get_latest_month()
            if verbose:
                print(f"자동으로 최신 월을 사용합니다: {as_of_month}")
        
        recent_1m, base_1m, recent_3m, base_3m = self._calculate_time_windows(as_of_month)
        
        if verbose:
            print(f"\n기준 월: {as_of_month}")
            print(f"최근 1개월: {recent_1m}, 기준 1개월: {base_1m}")
            print(f"최근 3개월: {recent_3m}, 기준 3개월: {base_3m}")
            print(f"\nPoisson 설정: alpha={alpha}, correction={correction_method}")
        
        all_keywords = self._lf_keyword_monthly.select("keyword").unique().collect()
        
        if verbose:
            print(f"총 keyword 개수: {len(all_keywords)}")
        
        # 윈도우별 집계
        keyword_1m = self._aggregate_keyword_window(recent_1m, base_1m, all_keywords, 1)
        N_recent_1m, N_base_1m = self._aggregate_total_window(recent_1m, base_1m)
        
        keyword_3m = self._aggregate_keyword_window(recent_3m, base_3m, all_keywords, 3)
        N_recent_3m, N_base_3m = self._aggregate_total_window(recent_3m, base_3m)
        
        # 베이스라인 생성 (지표 포함)
        baseline_1m = self._create_window_baseline(
            keyword_1m, as_of_month, 1, N_recent_1m, N_base_1m,
            z_threshold, min_c_recent, eps, alpha, correction_method
        )
        baseline_3m = self._create_window_baseline(
            keyword_3m, as_of_month, 3, N_recent_3m, N_base_3m,
            z_threshold, min_c_recent, eps, alpha, correction_method
        )
        
        baseline_final = pl.concat([baseline_1m, baseline_3m]).sort(["keyword", "window"])
        
        if verbose:
            print("\n베이스라인 테이블 생성 완료")
        
        return baseline_final.lazy()
    
    def detect_spike_by_z_log_score(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1
    ) -> pl.LazyFrame:
        """
        Z-score 기반 키워드 스파이크 탐지 결과를 반환합니다.
        
        Parameters
        ----------
        baseline_lf : pl.LazyFrame
            create_baseline_table()로 생성된 베이스라인 테이블
        window : int, default=1
            분석할 윈도우 크기 (1 또는 3)
        
        Returns
        -------
        pl.LazyFrame
            해당 윈도우의 z_log 기반 스파이크 탐지 결과
        """
        return (
            baseline_lf
            .filter(pl.col("window") == window)
            .filter(pl.col("is_spike_z") == True)
            .sort("z_log", descending=True)
        )
    
    def detect_spike_by_poisson(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1
    ) -> pl.LazyFrame:
        """
        Poisson 분포 기반 키워드 스파이크 탐지 결과를 반환합니다.
        
        Parameters
        ----------
        baseline_lf : pl.LazyFrame
            create_baseline_table()로 생성된 베이스라인 테이블
        window : int, default=1
            분석할 윈도우 크기 (1 또는 3)
        
        Returns
        -------
        pl.LazyFrame
            해당 윈도우의 Poisson 기반 스파이크 탐지 결과
            - lambda_pois: Poisson 기대치 λ = (C_base / N_base) * N_recent
            - p_pois: P(X >= k | λ), tail probability
            - score_pois: -log10(p_pois), 이상도 점수
            - is_spike_p: 스파이크 여부
        """
        return (
            baseline_lf
            .filter(pl.col("window") == window)
            .filter(pl.col("is_spike_p") == True)
            .sort("score_pois", descending=True)
        )
    
    def detect_spikes(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1,
        spike_type: Literal["ratio", "z", "poisson"] = "ratio"
    ) -> pl.LazyFrame:
        """
        스파이크 키워드만 필터링하여 반환합니다.
        
        Parameters
        ----------
        baseline_lf : pl.LazyFrame
            create_baseline_table()로 생성된 베이스라인 테이블
        window : int, default=1
            분석할 윈도우 크기 (1 또는 3)
        spike_type : Literal["ratio", "z", "poisson"], default="ratio"
            "ratio": score_ratio 기반 is_spike
            "z": z_log 기반 is_spike_z
            "poisson": Poisson 기반 is_spike_p
        
        Returns
        -------
        pl.LazyFrame
            스파이크로 판단된 키워드만 포함
        """
        spike_map = {
            "ratio": ("is_spike", "score_ratio"),
            "z": ("is_spike_z", "z_log"),
            "poisson": ("is_spike_p", "score_pois")
        }
        spike_col, sort_col = spike_map[spike_type]
        
        return (
            baseline_lf
            .filter(
                (pl.col("window") == window) &
                (pl.col(spike_col) == True)
            )
            .sort(sort_col, descending=True)
        )
    
    def detect_spike_ensemble(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1,
        min_methods: int = 2,
        min_c_recent: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        앙상블 기반 스파이크 탐지 (다중 방법 통합)
        
        절대빈도 가드를 통과한 후, 여러 탐지 방법 중 
        min_methods개 이상 만족 시 최종 스파이크로 판정합니다.
        
        Parameters
        ----------
        baseline_lf : pl.LazyFrame
            create_baseline_table()로 생성된 베이스라인 테이블
        window : int, default=1
            분석할 윈도우 크기 (1 또는 3)
        min_methods : int, default=2
            스파이크 판정에 필요한 최소 방법 수 (1~3)
            - 1: 하나라도 만족하면 스파이크 (OR 조건, 민감)
            - 2: 2개 이상 만족 시 스파이크 (권장)
            - 3: 모두 만족해야 스파이크 (AND 조건, 보수적)
        min_c_recent : int, optional
            절대빈도 가드 (C_recent 최소값)
            None이면 baseline_table 생성 시 사용한 값 적용
        
        Returns
        -------
        pl.LazyFrame
            앙상블 스파이크 탐지 결과
            추가 컬럼:
            - n_methods: 만족한 방법 수 (0~3)
            - is_spike_ensemble: 최종 앙상블 스파이크 여부
        
        Notes
        -----
        3가지 방법:
        - (A) ratio 기반: is_spike (score_ratio 기준)
        - (B) z-score 기반: is_spike_z (z_log 기준)
        - (C) Poisson 기반: is_spike_p (p_pois 기준, 다중검정 보정 적용)
        """
        result = (
            baseline_lf
            .filter(pl.col("window") == window)
            # 만족한 방법 수 계산
            .with_columns(
                (
                    pl.col("is_spike").cast(pl.Int8) +
                    pl.col("is_spike_z").cast(pl.Int8) +
                    pl.col("is_spike_p").cast(pl.Int8)
                ).alias("n_methods")
            )
        )
        
        # 절대빈도 가드 적용
        if min_c_recent is not None:
            result = result.filter(pl.col("C_recent") >= min_c_recent)
        
        # 앙상블 스파이크 판정
        result = (
            result
            .with_columns(
                (pl.col("n_methods") >= min_methods).alias("is_spike_ensemble")
            )
            # 컬럼 순서 재배치: is_spike 시리즈를 마지막으로
            .select(
                pl.exclude(["is_spike", "is_spike_z", "is_spike_p", 
                           "n_methods", "is_spike_ensemble"]),
                "n_methods",
                "is_spike",
                "is_spike_z", 
                "is_spike_p",
                "is_spike_ensemble"
            )
            .sort("n_methods", descending=True)
        )
        
        return result
    
    def get_ensemble_spikes(
        self,
        baseline_lf: pl.LazyFrame,
        window: Union[int, Literal[1, 3]] = 1,
        min_methods: int = 2,
        min_c_recent: Optional[int] = None
    ) -> pl.LazyFrame:
        """
        앙상블 기준으로 스파이크인 키워드만 반환합니다.
        
        Parameters
        ----------
        baseline_lf : pl.LazyFrame
            create_baseline_table()로 생성된 베이스라인 테이블
        window : int, default=1
            분석할 윈도우 크기 (1 또는 3)
        min_methods : int, default=2
            스파이크 판정에 필요한 최소 방법 수
        min_c_recent : int, optional
            절대빈도 가드
        
        Returns
        -------
        pl.LazyFrame
            앙상블 스파이크로 판정된 키워드만 포함
        """
        return (
            self.detect_spike_ensemble(
                baseline_lf, 
                window=window, 
                min_methods=min_methods,
                min_c_recent=min_c_recent
            )
            .filter(pl.col("is_spike_ensemble") == True)
            .sort(["n_methods", "score_pois"], descending=True)
        )


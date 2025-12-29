# =========================================================
# 통계 검정 클래스
# ========================================================="


# 1. 표준 라이브러리
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Sequence

# 2. 서드파티 라이브러리
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike
from IPython import get_ipython
from IPython.display import display

# 2-1. 시각화 라이브러이
import matplotlib.pyplot as plt

# 2-2. 통계 분석
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, chi2_contingency
from scipy.stats import mannwhitneyu, f_oneway
import scikit_posthocs as sp
import pingouin as pg
from statsmodels.stats.multicomp import MultiComparison

# 3. 로컬 모듈
from src.utils import is_running_in_notebook

display = display if is_running_in_notebook() else print


@dataclass
class TestResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    effect_interpretation: str
    conclusion: str
    metadata: dict = None

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha
    
    def to_dict(self) -> dict:
        return self.__dict__


class StatisticalTest(ABC):
    """
    모델 통계 검정의 기반 추상 클래스
    """
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha
        
    def execute_all(
        self,
        data: pd.DataFrame,
        iv_col: str,
        dv_cols: List[str],
        *,
        labels: Optional[list] = None,
        alpha: Optional[float] = None,
        **kwargs
    ):
        alpha = alpha if alpha is not None else self.alpha
        
        if labels is None:
            labels = [0, 1]

        results = []
        for dv_col in dv_cols:
            result = self.execute(
                data=data,
                iv_col=iv_col,
                dv_col=dv_col,
                labels=labels,
                alpha=alpha,
                **kwargs
            )
            results.append(result)

        return results
    
    @abstractmethod
    def execute(self):
        pass
    
    @abstractmethod
    def interpret(self):
        pass

    def check_homogeneity(self, data: List[pd.Series], alpha: Optional[float] = None):
        """
        등분산성 검정 수행 (Levene's Test)

        ANOVA 및 평균 비교 분석의 가정사항 중 하나인
        등분산성(homoscedasticity)을 검증한다.

        Parameters
        ----------
        data : list of array-like
            각 그룹의 데이터를 담은 리스트
            예: [group1, group2, group3]
        alpha : float, optional
            유의수준 (기본값: 0.05)

        Returns
        -------
        stat : float
            Levene 검정 통계량
        p_value : float
            p-value
        equal_var : bool
            등분산성 만족 여부
            - True : 등분산성 만족
            - False : 등분산성 위반
        """
        alpha = alpha if alpha else self.alpha
        print("\n[등분산성 검정 - Levene's Test]")
        print("-" * 50)

        stat, p_value = levene(*data)

        print(f"Levene 통계량: {stat:.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value > self.alpha:
            print("✅ 등분산성 가정을 만족합니다.")
            is_equal_var = True
        else:
            print("⚠️ 등분산성 가정을 만족하지 않습니다.")
            print("   → Welch 방법 또는 비모수 검정 고려")
            is_equal_var = False

        return stat, p_value, is_equal_var
    
    def check_group_normality(self, data: List[ArrayLike], group_labels: List[str], mode: str, alpha: Optional[float] = None):
        alpha = alpha if alpha else self.alpha
        results = []
        for label, group_data in zip(group_labels, data):
            stat, p_value, is_normal = self.check_normality(group_data, group_labels, mode)
            results.append({
                '그룹': label,
                'W-통계량': round(stat, 4) if stat else None,
                'p-value': round(p_value, 4) if p_value else None,
                '판정': is_normal
            })
        result_df = pd.DataFrame(results)
        display(result_df)
        
        all_normal = all(r['판정'] for r in results)
        return all_normal

    def check_normality(self, data: ArrayLike, label: str, mode: str, alpha: Optional[float] = None):
        alpha = alpha if alpha else self.alpha
        if mode == 'soft':
            return self.check_normality_soft(data, label, alpha)
        elif mode == 'hard':
            return self.check_normality_hard(data, label, alpha)
        else:
            msg = f'Invalid mode: "{mode}". Expected one of ["soft", "hard"].'
            raise ValueError(msg)

    def check_normality_soft(self, data: ArrayLike, label: str = "데이터", alpha: Optional[float] = None):
        alpha = alpha if alpha else self.alpha
        """
        데이터의 정규성을 검정하는 함수: t-test용

        Parameters
        ----------
        data : array-like
            정규성을 검정할 데이터 (NaN은 자동 제거)
        label : str, default="데이터"
            출력 시 표시될 데이터 이름

        Returns
        -------
        bool
            정규분포 가정 충족 여부
            - True: 정규분포 가정 가능 (모수 검정)
            - False: 정규분포 가정 위반 (비모수 검정)

        검정 기준
        ---------
        - n < 30: Shapiro-Wilk 검정 (p > 0.05)
        - 30 ≤ n < 100: 왜도/첨도 우선, 필요시 Shapiro-Wilk
        - n ≥ 100: 왜도 기준 (|왜도| < 2, 중심극한정리)
        """
        # NaN 체크
        if pd.isna(data).any():
            print(f"⚠️ 경고: {label}에 NaN 값이 {pd.isna(data).sum()}개 포함됨")
            data = data.dropna()
            print(f"   → NaN 제거 후 n={len(data)}")

        n = len(data)

        print(f"\n[{label} 정규성 검정] n={n}")
        print("-"*40)

        # 왜도와 첨도
        skew = stats.skew(data)
        kurt = stats.kurtosis(data, fisher=True)
        print(f"왜도(Skewness): {skew:.3f}")
        print(f"첨도(Kurtosis): {kurt:.3f}")

        # 표본 크기에 따른 판단
        if n < 30:
            stat, p_value = shapiro(data)
            print(f"Shapiro-Wilk p-value: {p_value:.4f}")
            is_normal = p_value > self.alpha
            reason = f"Shapiro p={'>' if is_normal else '≤'}{self.alpha}"
        elif n < 100:
            if abs(skew) < 1 and abs(kurt) < 2:
                stat, p_value = None, None
                is_normal = True
                reason = "|왜도|<1, |첨도|<2"
            else:
                stat, p_value = shapiro(data)
                print(f"추가 Shapiro-Wilk p-value: {p_value:.4f}")
                is_normal = p_value > self.alpha
                reason = f"Shapiro p={'>' if is_normal else '≤'}{self.alpha}"
        else:
            stat, p_value = None, None
            is_normal = abs(skew) < 2
            reason = f"|왜도|{'<' if is_normal else '≥'}2 (중심극한정리)"

        print(f"결과: {'✅ 정규분포 가정 충족' if is_normal else '❌ 정규분포 가정 위반'} ({reason})")
        return stat, p_value, is_normal

    def check_normality_hard(self, data: ArrayLike, label: str = "데이터", alpha: float = None):
        alpha = alpha if alpha else self.alpha
        
        # NaN 체크
        if pd.isna(data).any():
            print(f"⚠️ 경고: {label}에 NaN 값이 {pd.isna(data).sum()}개 포함됨")
            data = data.dropna()
            print(f"   → NaN 제거 후 n={len(data)}")

        n = len(data)
        print(f"\n[{label} 정규성 검정] n={n}")
        print("-"*40)
        
        stat, p_value = shapiro(data)
        is_normal = p_value > alpha
        print(f"Shapiro-Wilk p-value: {p_value:.4f}")
        reason = f"Shapiro p={'>' if is_normal else '≤'}{self.alpha}"
        is_normal = True if p_value > self.alpha else False
        
        print(f"결과: {'✅ 정규분포 가정 충족' if is_normal else '❌ 정규분포 가정 위반'} ({reason})")
    
        return stat, p_value, is_normal

class TTest(StatisticalTest):
    """
    t-검정 (독립/대응)
    """
    
    def __init__(self):
        self.test_name = 't-검정'
    
    def execute(self, 
        data: pd.DataFrame, 
        iv_col: str, dv_col: str, 
        labels: list = [0, 1], 
        alpha: Optional[float] = None,
        mode: str = 'soft'
    ):
        """
        두 그룹 간 평균 차이에 대한 가설검정을 수행하는 함수.
        (정규성에 따라 t-검정 또는 Mann-Whitney U 검정을 자동 선택)

        Parameters
        ----------
        data: pd.DataFrame
            원본 데이터
        iv_col: str
            독립 변수 컬럼명
        dv_col: str
            종속 변수 컬럼명
        labels: list
            독립 변수의 값들
        alpha : float, optional
            유의수준 (default=0.05)

        Returns
        -------
        result : TestResult
        """
        alpha = alpha if alpha is not None else self.alpha
        
        class0_data = data[data[iv_col] == labels[0]][dv_col].dropna()
        class1_data = data[data[iv_col] == labels[1]][dv_col].dropna()
        
        self.plot(class0_data, class1_data, labels, iv_col, dv_col)
        
        stat, p_levene, is_equal_var = self.check_homogeneity([class0_data, class1_data], alpha)
        
        _, _, is_normal_0 = self.check_normality(class0_data, label=labels[0], mode=mode)
        _, _, is_normal_1 = self.check_normality(class0_data, label=labels[1], mode=mode)
        
        print("\n[가설검정]")
        print("-" * 40)

        if is_normal_0 and is_normal_1:
            print("H₀: μ₀ = μ₁ (두 클래스의 평균이 같다)")
            print("H₁: μ₀ ≠ μ₁ (두 클래스의 평균이 다르다)")
        else:
            print("H₀: 두 클래스의 분포가 같다 (중앙값 차이가 없다)")
            print("H₁: 두 클래스의 분포가 다르다 (중앙값 차이가 있다)")
            
        print(f"유의수준: α = {alpha}\n")

        # --- 검정 수행 ---
        if is_normal_0 and is_normal_1:
            # 모수 검정
            test_name = "Student's t-test" if is_equal_var else "Welch's t-test"
            t_stat, p_value = ttest_ind(class0_data, class1_data, is_equal_var=is_equal_var)
            print(f"{test_name} 결과:")
            print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
            
            # Cohen's d 계산
            pooled_std = np.sqrt((class0_data.var() + class1_data.var()) / 2)
            cohens_d = (class0_data.mean() - class1_data.mean()) / pooled_std
            abs_d = abs(cohens_d)

            if abs_d < 0.2:
                effect = "매우 작은 효과"
            elif abs_d < 0.5:
                effect = "작은 효과"
            elif abs_d < 0.8:
                effect = "중간 효과"
            else:
                effect = "큰 효과"

            print(f"Cohen's d = {cohens_d:.3f} ({effect})")

            test_stat = t_stat
            effect_size = cohens_d
            effect_interpretation = effect

        else:
            # 비모수 검정
            test_name = "Mann-Whitney U test"
            u_stat, p_value = mannwhitneyu(class0_data, class1_data, alternative='two-sided')
            print(f"{test_name} 결과:")
            print(f"U = {u_stat:.4f}, p = {p_value:.4f}")
            
            # 총 샘플 크기 (N)
            n0 = len(class0_data)
            n1 = len(class1_data)
            N = n0 + n1
            
            # 효과 크기 계산 (rank-biserial crrelation)
            r_rb = (2 * u_stat) / (n0 * n1) - 1
            abs_rb = abs(r_rb)
            
            if abs_rb < 0.1:
                effect = "매우 작은 효과"
            elif abs_rb < 0.3:
                effect = "작은 효과"
            elif abs_rb < 0.5:
                effect = "중간 효과"
            else:
                effect = "큰 효과"
            
            test_stat = u_stat
            effect_size = r_rb
            effect_interpretation = effect

        # --- 결론 ---
        print("\n[결론]")
        if p_value < alpha:
            conclusion = f"✅ p-value({p_value:.4f}) < {alpha} → 귀무가설 기각\n   두 클래스에 유의한 차이가 있음"
        else:
            conclusion = f"❌ p-value({p_value:.4f}) ≥ {alpha} → 귀무가설 채택\n   두 클래스에 유의한 차이가 없음"

        print(conclusion)
        
        metadata = {
            'f_stat': stat,
            'p_levene': p_levene
        }
        
        # 결과 반환
        return TestResult(
            test_name=test_name,
            statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            conclusion=conclusion,
            metadata=metadata
        )
    
    def interpret(self, result: TestResult, alpha: float = 0.05) -> None:
        if result.p_value >= alpha:
            pass
    
    @staticmethod
    def plot(class0_data: ArrayLike, class1_data: ArrayLike, labels: List[str], iv_col: str, dv_col: str):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes: Sequence[Axes]
        # 박스플롯
        bp = axes[0].boxplot([class0_data, class1_data],
                            labels=labels,
                            patch_artist=True)
        
        boxes: list[Patch] = bp["boxes"] 
        boxes[0].set_facecolor('lightblue')
        boxes[1].set_facecolor('lightcoral')
        axes[0].set_ylabel(dv_col)
        axes[0].set_title(f'{dv_col} 분포')
        axes[0].grid(True, alpha=0.3)

        # 히스토그램
        axes[1].hist(class0_data, bins=10, alpha=0.6, label=f'{iv_col} - {labels[0]}', 
                    color='blue', density=True, edgecolor='black')
        axes[1].hist(class1_data, bins=10, alpha=0.6, label=f'{iv_col} - {labels[1]}', 
                    color='red', density=True, edgecolor='black')
        axes[1].set_xlabel(dv_col)
        axes[1].set_ylabel('밀도')
        axes[1].set_title(f'{dv_col} 분포 비교')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Q-Q plot (Class 0)
        stats.probplot(class0_data, dist="norm", plot=axes[2])
        axes[2].set_title(f'Q-Q Plot ({labels[0]})')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


class Chi2Test(StatisticalTest):
    def __init__(self):
        super().__init__()
        self.test_name = '카이제곱 검정'
    
    def execute(self, 
        data: pd.DataFrame, 
        iv_col: str, dv_col: str, 
        alpha: Optional[float] = None
    ):
        alpha = alpha if alpha else self.alpha
        df = data[[iv_col, dv_col]]
        
        # 교차표
        contingency_table = pd.crosstab(df[iv_col], df[dv_col])
        
        # 기대빈도 확인
        is_valid = self.check_expected_frequencies(contingency_table)
        
        # 카이제곱 검정
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        n = contingency_table.values.sum()  # 전체 표본 수
        r, c = contingency_table.shape      # 행수, 열수
        
        test_stat = chi2_stat
        # ==============================================
        # Cramér's V 값 (0~1 사이)
        # - 0에 가까울수록: 독립적 (연관성 없음)
        # - 1에 가까울수록: 강한 연관성
        # ==============================================
        v, interpretation = self.cramers_v(chi2_stat, n, r, c)
        
        effect_size = v
        effect_interpretation = interpretation
        
        # 기대빈도 테이블
        expected_df = pd.DataFrame(
            expected, 
            index=contingency_table.index,
            columns=contingency_table.columns
        )
        print("\n[기대빈도]")
        print(expected_df.round(2))

        
        # ==============================================
        # 표준화 잔차
        # - |잔차| > 2: 해당 셀이 독립성에서 유의하게 벗어남
        # - |잔차| > 3: 매우 강한 연관성 (이상치 수준)
        # - 양수: 관측값이 기대값보다 큼 (과대 표현)
        # - 음수: 관측값이 기대값보다 작음 (과소 표현)
        # ==============================================
        std_residuals = (contingency_table.values - expected) / np.sqrt(expected)
        residuals_df = pd.DataFrame(
            std_residuals,
            index=contingency_table.index,
            columns=contingency_table.columns
        )
        print("\n[표준화 잔차]")
        display(residuals_df.round(2))
        print("(|잔차| > 2: 유의한 차이, |잔차| > 3: 매우 강한 연관성)")
        
        # 결과 요약
        print("\n[결론]")
        if p_value < alpha:
            conclusion = f"✅ p-value({p_value:.4f}) < {alpha} → 귀무가설 기각\n" \
                            f"   {iv_col}과(와) {dv_col}은(는) 관련이 있음" \
                            f"   효과 크기: {effect_interpretation}"
            print(conclusion)
            
            # 사후분석
            print("\n[사후분석]")
            print("   표준화 잔차 |값| > 2인 셀 해석:")
            post_hoc = "   표준화 잔차 |값| > 2인 셀 해석:"
            for i, row_label in enumerate(contingency_table.index):
                for j, col_label in enumerate(contingency_table.columns):
                    if abs(std_residuals[i, j]) > 2:
                        if std_residuals[i, j] > 0:
                            post_hoc += f"\n   • {row_label} - {col_label}: 예상보다 많음 (잔차={std_residuals[i, j]:.2f})"
                        else:
                            post_hoc += f"\n   • {row_label} - {col_label}: 예상보다 적음 (잔차={std_residuals[i, j]:.2f})"
            print(post_hoc)
            metadata = {'post_hoc': post_hoc}
        else:
            conclusion = f"❌ p-value({p_value:.4f}) ≥ {alpha} → 귀무가설 채택" \
                            f"   {iv_col}과(와) {dv_col}은(는) 독립적임 (연관 없음)"
            print(conclusion)
            metadata = {}
        
        return TestResult(
            test_name=self.test_name,
            statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            conclusion=conclusion,
            metadata=metadata
        )

    
    def interpret(self):
        return super().interpret()

    @staticmethod
    def cramers_v(chi2_stat, n, r, c):
        """
        Cramér's V 효과 크기 계산
        
        카이제곱 검정의 효과 크기를 측정하는 지표로, 두 범주형 변수 간
        연관성의 강도를 0~1 사이 값으로 표현합니다.
        
        Parameters
        ----------
        chi2_stat : float
            카이제곱 통계량 (χ²)
        n : int
            전체 표본 수 (분할표의 총합)
        r : int
            행(row)의 개수
        c : int
            열(column)의 개수
        
        Returns
        -------
        float
            Cramér's V 값 (0~1 사이)
            - 0에 가까울수록: 독립적 (연관성 없음)
            - 1에 가까울수록: 강한 연관성
        """
        v = np.sqrt(chi2_stat / (n * min(r-1, c-1)))
        
        # Cramér's V 값 효과 크기 해석
        if v < 0.1:
            interpretation = "매우 약한 관계"
        elif v < 0.3:
            interpretation = "약한 관계"
        elif v < 0.5:
            interpretation = "중간 관계"
        else:
            interpretation = "강한 관계"
        return v, interpretation
    
    @staticmethod
    def check_expected_frequencies(contingency_table):
        """
        카이제곱 검정의 기대빈도 가정 확인
        
        카이제곱 검정을 수행하기 전에 기대빈도가 충분한지 검사합니다.
        기대빈도가 너무 작으면 카이제곱 검정의 정확도가 떨어집니다.
        
        Parameters
        ----------
        contingency_table : array-like
            분할표 (관측 빈도)
        
        Returns
        -------
        bool
            카이제곱 검정 사용 가능 여부
            - True: 카이제곱 검정 사용 가능
            - False: Fisher's exact test 권장
        
        검정 기준
        ---------
        1. 모든 기대빈도 ≥ 5 (이상적)
        2. 기대빈도 < 5인 셀이 전체의 20% 이하 (허용 가능)
        
        Notes
        -----
        - 2×2 분할표에서 기대빈도 < 5인 경우: Fisher's exact test 필수
        - 큰 분할표에서 일부 셀만 < 5: 카이제곱 검정 여전히 사용 가능
        """
        # 카이제곱 검정으로 기대빈도 계산
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table)
        
        print("\n[기대빈도 확인]")
        print("-"*40)
        
        # -------------------------------------------------------------------------
        # 1. 최소 기대빈도 확인
        # -------------------------------------------------------------------------
        min_expected = expected.min()
        print(f"최소 기대빈도: {min_expected:.2f}")
        
        # -------------------------------------------------------------------------
        # 2. 기대빈도 < 5인 셀의 비율 계산
        # -------------------------------------------------------------------------
        cells_below_5 = (expected < 5).sum()  # 5 미만인 셀 개수
        total_cells = expected.size  # 전체 셀 개수
        percent_below_5 = (cells_below_5 / total_cells) * 100
        
        print(f"5 미만 셀: {cells_below_5}/{total_cells} ({percent_below_5:.1f}%)")
        
        # -------------------------------------------------------------------------
        # 3. 카이제곱 검정 적합성 판단
        # -------------------------------------------------------------------------
        # 조건: 최소 기대빈도 ≥ 5 AND 5 미만 셀 비율 ≤ 20%
        if min_expected < 5 or percent_below_5 > 20:
            print("⚠️ 주의: Fisher's exact test 사용 권장")
            print("   (기대빈도가 너무 작아 카이제곱 검정 부정확)")
            return False
        else:
            print("✅ 카이제곱검정 사용 가능")
            return True
        

class OneWayAnova(StatisticalTest):
    def __init__(self):
        super().__init__()
        self.test_name = '일원배치 ANOVA'
        
        self.configs = {
            'f_test': {
                'test_name': 'F-test',
                'stat_func': self._run_f_test,
                'effect_func': self.calculate_eta_squared,
                'effect_name': 'η²',
                'post_hoc_func': self.perform_tukey_hsd,
                'is_rank': False
            },
            'welch': {
                'test_name': "Welch's ANOVA",
                'stat_func': self._run_welch_test,
                'effect_func': None,  # 효과 크기 계산 안 함
                'effect_name': None,
                'post_hoc_func': self.perform_gameshowell,
                'is_rank': False
            },
            'kruskal': {
                'test_name': 'Kruskal-Wallis',
                'stat_func': self._run_kruskal_test,
                'effect_func': self.calculate_epsilon_squared,
                'effect_name': 'ε²',
                'post_hoc_func': self.perform_dunn_test,
                'is_rank': True
            }
        }

    def _select_config(
        self, is_normal, is_equal_var,
        data, iv_col, dv_col, 
        data_groups, group_labels, 
        k, N
    ):
        """조건에 따라 적절한 검정 설정 선택"""
        if not is_normal:
            base_config = self.configs['kruskal']
            test_args = [data_groups, k, N]
            post_hoc_args = [data, iv_col, dv_col]
        elif is_equal_var:
            base_config = self.configs['f_test']
            test_args = [data_groups, k, N]
            post_hoc_args = [data_groups, group_labels]
        else:
            base_config = self.configs['welch']
            test_args = [data, iv_col, dv_col]
            post_hoc_args = [data, iv_col, dv_col]

        return {
            **base_config,  # 기존 설정 복사
            'test_args': test_args,
            'post_hoc_args': post_hoc_args
        }
    
    def execute(self,
        data: pd.DataFrame, 
        iv_col: str,
        dv_col: str = None,
        alpha: Optional[float] = None,
        mode: str = 'hard'
    ):
        print("\n[일원배치 ANOVA]")
        print("-"*50)
        
        alpha = alpha if alpha else self.alpha
        metadata = {}
        
        # 가설 설정
        print(f"H₀: 모든 {iv_col}의 평균 {dv_col}이 같다")
        print(f"H₁: 적어도 한 {iv_col}의 평균 {dv_col}이 다르다")
        print(f"유의수준: α = {alpha}")
        
        group_labels = data[iv_col].unique().tolist()
        data_groups = [data[data[iv_col] == label][dv_col] for label in group_labels]
        
        # 전체 표본 크기 및 그룹 수
        k = len(data_groups)
        N = sum(len(group) for group in data_groups)
        
        # 정규성, 등분산성
        is_normal = self.check_group_normality(data_groups, group_labels, mode=mode, alpha=alpha)
        is_equal_var = self.check_homogeneity(data_groups, alpha=alpha)
        
        # 적절한 검정 선택
        config = self._select_config(is_normal, is_equal_var, data, iv_col, dv_col, data_groups, group_labels, k, N)

        test_stat, p_value, df_info = config['stat_func'](*config['test_args'])
        effect_size, effect_interpretation = self._calculate_effect_size(
            config['effect_func'], 
            test_stat, *df_info, 
            iv_col=iv_col, dv_col=dv_col,
            effect_name=config['effect_name'], is_rank=config['is_rank']
        )
        conclusion, post_hoc = self._print_conclusion_and_posthoc(p_value, iv_col, dv_col, config['post_hoc_func'], *config['post_hoc_args'], alpha=alpha)
        
        metadata.update(post_hoc)
        
        return TestResult(
            test_name=self.test_name + ' - ' + config['test_name'],
            statistic=test_stat,
            p_value=p_value,
            effect_size=effect_size,
            effect_interpretation=effect_interpretation,
            conclusion=conclusion,
            metadata=metadata
        )
    
    def interpret(self):
        return super().interpret()
        
    # 각 검정별 실행 함수
    def _run_f_test(self, data_groups, k, N):
        f_stat, p_value = f_oneway(*data_groups)
        df1, df2 = k - 1, N - k
        print(f"\nF-test 결과:")
        print(f"자유도: F({df1}, {df2})")
        print(f"f = {f_stat:.4f}, p = {p_value:.4f}")
        return f_stat, p_value, (df1, df2)
    
    def _run_welch_test(self, data, iv_col, dv_col):
        print("\n⚠️ 등분산성 가정이 위반되었으므로 Welch's ANOVA를 수행합니다.")
        welch_result = pg.welch_anova(dv=dv_col, between=iv_col, data=data)
        f_stat = welch_result['F'].values[0]
        df1 = welch_result['ddof1'].values[0]
        df2 = welch_result['ddof2'].values[0]
        p_value = welch_result['p-unc'].values[0]
        print(f"\nWelch's ANOVA 결과:")
        print(f"자유도: F({df1:.2f}, {df2:.2f})")
        print(f"f = {f_stat:.4f}, p = {p_value:.4f}")
        print("\n※ 효과 크기 계산 불가 (이분산 검정)")
        return f_stat, p_value, ()
    
    def _run_kruskal_test(self, data_groups, k, N):
        print("\n⚠️ 정규성 가정이 위반되었으므로 비모수 검정을 수행합니다.")
        print("\n[Kruskal-Wallis 검정]")
        h_stat, p_value = stats.kruskal(*data_groups)
        print(f"H-통계량: {h_stat:.4f}")
        print(f"자유도: {k-1}")
        print(f"p-value: {p_value:.6f}")
        return h_stat, p_value, (k, N)
    
    def _calculate_effect_size(self, effect_calc_func, *args, iv_col, dv_col, 
                        effect_name="η²", is_rank=False):
        """효과 크기 계산 및 출력 (공통 로직)"""
        if not effect_calc_func:
            return None, None
        
        effect_size, interpretation = effect_calc_func(*args)
        
        unit = "순위 변동" if is_rank else dv_col
        print(f"효과 크기 ({effect_name}): {effect_size:.4f} ({interpretation})")
        print(f"   → {iv_col} 차이가 전체 {unit}의 {effect_size*100:.1f}% 설명")
        
        return effect_size, interpretation

    def _print_conclusion_and_posthoc(self, p_value, iv_col, dv_col, 
                                    post_hoc_func, *post_hoc_args, alpha):
        """결론 출력 및 사후검정 수행 (공통 로직)"""
        print("\n[검정 결론]")
        
        if p_value < alpha:
            conclusion = (
                f"✅ p-value({p_value:.6f}) < {alpha} → 귀무가설 기각\n"
                f"   {iv_col}별 {dv_col}에 유의한 차이가 있음"
            )
            print(conclusion)
            
            # 사후검정
            result, post_hoc_text = post_hoc_func(*post_hoc_args)
            if result is not None:
                display(result)
            print(post_hoc_text)
            return conclusion, {'post_hoc': post_hoc_text}
        else:
            conclusion = (
                f"❌ p-value({p_value:.6f}) ≥ {alpha} → 귀무가설 채택\n"
                f"   {iv_col}별 {dv_col}에 유의한 차이가 없음"
            )
            print(conclusion)
            return conclusion, {}
    @staticmethod
    def calculate_eta_squared(f_stat, df1, df2):
        """
        에타제곱 (효과 크기) 계산
        
        ANOVA 결과의 실질적 중요성을 평가하는 효과 크기를 계산합니다.
        에타제곱은 집단 차이가 전체 변동의 몇 %를 설명하는지 나타냅니다.
        
        주의: 이 함수는 F 통계량을 이용한 근사 공식을 사용합니다.
        정확한 계산을 위해서는 SS(Sum of Squares) 값이 필요하지만,
        F 통계량만으로도 충분히 신뢰할 수 있는 근사치를 제공합니다.
        
        근사 공식: η² ≈ (F × df_between) / (F × df_between + df_within)
        정확한 공식: η² = SS_between / SS_total
        
        Parameters
        ----------
        f_stat : float
            F 통계량
        df1 : int
            집단 간 자유도
        df2 : int
            집단 내 자유도
        
        Returns
        -------
        tuple
            (에타제곱 값, 해석 문구)
        """
        
        # 근사 공식 사용
        eta_sq= (f_stat * df1) / (f_stat * df1 + df2)
        if eta_sq < 0.01:
            interpretation = "매우 작은 효과"
        elif eta_sq < 0.06:
            interpretation = "작은 효과"
        elif eta_sq < 0.14:
            interpretation = "중간 효과"
        else:
            interpretation = "큰 효과"
        
        return eta_sq, interpretation
    
    @staticmethod
    def calculate_epsilon_squared(h_stat, k, n):
        """
        엡실론제곱 (비모수 효과 크기) 계산
        
        Kruskal-Wallis 검정 결과의 실질적 중요성을 평가하는 효과 크기를 계산합니다.
        엡실론제곱은 집단 차이가 전체 순위 변동의 몇 %를 설명하는지 나타냅니다.
        
        Parameters
        ----------
        h_statistic : float
            Kruskal-Wallis H 통계량
        k : int
            집단(그룹) 수
        n : int
            전체 표본 크기
        
        Returns
        -------
        tuple
            (엡실론제곱 값, 해석 문구)
        
        Notes
        -----
        공식: ε² = (H - k + 1) / (n - k)
        - H: Kruskal-Wallis H 통계량
        - k: 그룹 수
        - n: 전체 표본 수
        
        해석 기준 (Cohen's 기준과 동일):
        - < 0.01: 매우 작은 효과
        - 0.01 ~ 0.06: 작은 효과
        - 0.06 ~ 0.14: 중간 효과
        - ≥ 0.14: 큰 효과
        """
        # 엡실론제곱 계산
        epsilon_sq = (h_stat - k + 1) / (n - k)
        
        # 효과 크기 해석
        if epsilon_sq < 0.01:
            interpretation = "매우 작은 효과"
        elif epsilon_sq < 0.06:
            interpretation = "작은 효과"
        elif epsilon_sq < 0.14:
            interpretation = "중간 효과"
        else:
            interpretation = "큰 효과"
        
        return epsilon_sq, interpretation
    
    def perform_tukey_hsd(self, data_groups: List[ArrayLike], group_labels: List[str], alpha: float = None):
        """
        Tukey HSD 사후검정 수행
        
        ANOVA에서 유의한 차이가 발견된 경우, 어느 집단 간에 차이가 있는지
        구체적으로 확인하기 위한 다중비교 검정을 수행합니다.
        
        Parameters
        ----------
        data : list of arrays
            각 그룹의 데이터
        labels : list
            각 그룹의 이름
        
        Returns
        -------
        TukeyHSDResults
            Tukey HSD 검정 결과 객체
        """
        alpha = alpha if alpha else self.alpha
        post_hoc = []

        print("\n[Tukey HSD 사후검정]")
        print("-" * 50)
        
        # 데이터를 긴 형식으로 변환
        all_data = []
        all_labels = []
        
        for label, group_data in zip(group_labels, data_groups):
            all_data.extend(group_data)
            all_labels.extend([label] * len(group_data))
        
        # Tukey HSD 수행
        mc = MultiComparison(all_data, all_labels)
        result = mc.tukeyhsd(alpha)
        display(result)
        
        # -----------------------------------------------------------------------------
        # 결과 해석
        # -----------------------------------------------------------------------------
        print("\n[결과 해석]")
        print("-" * 50)
        
        # 1. 각 그룹의 평균 계산 및 정렬
        group_means = {}
        for i, label in enumerate(group_labels):
            group_means[label] = np.mean(data_groups[i])
        
        sorted_groups = sorted(group_means.items(), key=lambda x: x[1], reverse=True)
        
        post_hoc.append("평균 순위:")
        for rank, (group, mean) in enumerate(sorted_groups, 1):
            post_hoc.append(f"  {rank}위: {group} (평균: {mean:.2f})")
        
        # 2. 유의성 관계 파악
        post_hoc.append("\n그룹 간 관계:")
        sig_matrix = {}
        
        # Tukey 결과에서 정보 추출
        for row in result.summary().data[1:]:  # 헤더 제외
            group1 = str(row[0]).strip()
            group2 = str(row[1]).strip()
            meandiff = float(row[2])
            p_adj = float(row[3])
            reject = str(row[6]).strip() == 'True'
            
            # 양방향으로 저장
            sig_matrix[(group1, group2)] = reject
            sig_matrix[(group2, group1)] = reject
            
            # 관계 출력
            if reject:
                post_hoc.append(
                    f"  • {group1} ≠ {group2} (p={p_adj:.4f}, 유의한 차이)"
                )
            else:
                post_hoc.append(
                    f"  • {group1} ≈ {group2} (p={p_adj:.4f}, 차이 없음)"
                )

        # 3. 시각화
        fig = result.plot_simultaneous(figsize=(10, 6))
        plt.title('Tukey HSD 95% 신뢰구간')
        plt.xlabel('그룹 간 평균 차이')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        post_hoc_text = '\n'.join(post_hoc)
        return result, post_hoc_text
    
    def perform_gameshowell(self, df: pd.DataFrame, iv_col: str, dv_col: str, alpha: float):
        """
        Games-Howell 사후검정 수행
        
        등분산성 가정을 만족하지 않을 때 사용하는 사후검정입니다.
        정규성은 만족하지만 등분산성이 위반된 경우에 적합합니다.
        
        Parameters
        ----------
        df : pandas.DataFrame
            분석할 데이터프레임
        iv_col : str
            집단변수(범주형) 컬럼명
        dv_col : str
            종속변수(연속형) 컬럼명
        
        Returns
        -------
        pandas.DataFrame
            Games-Howell 검정 결과
        """
        alpha = alpha if alpha else self.alpha
        post_hoc = []
        print("\n[Games-Howell 사후검정]")
        print("-"*50)
        print("※ 등분산성 가정을 만족하지 않아 Games-Howell 사용\n")
        
        # Games-Howell 수행
        result = pg.pairwise_gameshowell(dv=dv_col, between=iv_col, data=df)
        
        # =========================================================================
        # pingouin 버전에 따른 컬럼명 확인 및 처리
        # =========================================================================
        # 최신 버전: 'pval'과 'reject' 대신 'p-unc'와 'sig' 사용
        # 구버전: 'pval'과 'reject' 사용
        
        # p-value 컬럼 확인
        if 'pval' in result.columns:
            pval_col = 'pval'
        elif 'p-unc' in result.columns:
            pval_col = 'p-unc'
        else:
            raise ValueError("p-value 컬럼을 찾을 수 없습니다.")
        
        # reject/sig 컬럼 확인 (없으면 직접 생성)
        if 'reject' not in result.columns and 'sig' not in result.columns:
            result['reject'] = result[pval_col] < alpha
            reject_col = 'reject'
        elif 'reject' in result.columns:
            reject_col = 'reject'
        else:
            reject_col = 'sig'
            result['reject'] = result[reject_col]  # 호환성을 위해 'reject' 컬럼 추가
        
        # 결과 출력을 위한 컬럼 선택
        display_cols = ['A', 'B', 'mean(A)', 'mean(B)', 'diff', pval_col]
        if reject_col in result.columns:
            display_cols.append(reject_col)
        
        print("[사후검정 결과]")
        print("-"*50)
        
        display(result[display_cols].round(4))
        
        # -----------------------------------------------------------------------------
        # 결과 해석
        # -----------------------------------------------------------------------------
        print("\n[결과 해석]")
        print("-"*50)
        
        # 1. 각 그룹의 평균 계산 및 정렬
        group_means = df.groupby(iv_col)[dv_col].mean().sort_values(ascending=False)
        
        post_hoc.append("평균 순위:")
        for rank, (group, mean) in enumerate(group_means.items(), 1):
            post_hoc.append(
                f"  {rank}위: {group} (평균: {mean:.2f})"
            )
        
        # 2. 유의성 관계 파악
        post_hoc.append("\n그룹 간 관계:")
        for _, row in result.iterrows():
            is_significant = row['reject']
            p_value = row[pval_col]
            
            if row['reject']:
                post_hoc.append(
                    f"  • {row['A']} ≠ {row['B']} "
                    f"(p={p_value:.4f}, 유의한 차이)"
                )
            else:
                post_hoc.append(
                    f"  • {row['A']} ≈ {row['B']} "
                    f"(p={p_value:.4f}, 차이 없음)"
                )
        
        # 3. 시각화
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # 평균 차이와 신뢰구간 시각화
        y_pos = range(len(result))
        comparisons = [f"{row['A']}-{row['B']}" for _, row in result.iterrows()]
        diffs = result['diff'].values
        
        # 신뢰구간 계산 (SE * 1.96)
        errors = result['se'].values * 1.96
        
        colors = ['red' if reject else 'gray' for reject in result['reject']]
        
        ax.barh(y_pos, diffs, xerr=errors, color=colors, alpha=0.6, capsize=5)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(comparisons)
        ax.set_xlabel('평균 차이 (95% CI)')
        ax.set_title('Games-Howell 사후검정 결과')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 범례
        legend_elements = [Patch(facecolor='red', alpha=0.6, label=f'유의한 차이 (p<{alpha})'),
                        Patch(facecolor='gray', alpha=0.6, label=f'차이 없음 (p≥{alpha})')]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()
        
        post_hoc_text = "\n".join(post_hoc)

        return result, post_hoc_text
    
    def perform_dunn_test(self, df: pd.DataFrame, iv_col: str, dv_col: str, alpha: float = None):
        """
        Dunn's 사후검정 수행
        """
        alpha = alpha if alpha else self.alpha
        post_hoc = []

        print("\n[Dunn's Test 사후검정]")
        print("-" * 50)
        print("※ 정규성 가정을 만족하지 않아 비모수 사후검정 사용\n")

        # Dunn's test 수행 (Bonferroni 보정)
        dunn_result = sp.posthoc_dunn(
            df,
            val_col=dv_col,
            group_col=iv_col,
            p_adjust='bonferroni'
        )

        print("[사후검정 결과 - p-value 행렬 (Bonferroni 보정 적용됨)]")
        print("-" * 50)

        display(dunn_result.round(4))

        # -----------------------------------------------------------------------------
        # 결과 해석
        # -----------------------------------------------------------------------------
        print("\n[결과 해석]")
        print("-" * 50)

        # 1. 중앙값 순위
        group_medians = (
            df.groupby(iv_col)[dv_col]
            .median()
            .sort_values(ascending=False)
        )

        post_hoc.append(
            "중앙값 순위 (비모수 검정은 순위 기반이므로 중앙값 참조):"
        )
        group_means = df.groupby(iv_col)[dv_col].mean()

        for rank, (group, median) in enumerate(group_medians.items(), 1):
            post_hoc.append(
                f"  {rank}위: {group} "
                f"(중앙값: {median:.2f}, 참고-평균: {group_means[group]:.2f})"
            )

        # 2. 유의성 관계
        post_hoc.append("\n그룹 간 관계:")
        groups = dunn_result.columns.tolist()

        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                p_val = dunn_result.iloc[i, j]
                if p_val < alpha:
                    post_hoc.append(
                        f"  • {groups[i]} ≠ {groups[j]} "
                        f"(p={p_val:.4f}, 유의한 차이)"
                    )
                else:
                    post_hoc.append(
                        f"  • {groups[i]} ≈ {groups[j]} "
                        f"(p={p_val:.4f}, 차이 없음)"
                    )

        # -----------------------------------------------------------------------------
        # 시각화 (유지)
        # -----------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 8))

        import seaborn as sns

        mask = np.triu(np.ones_like(dunn_result, dtype=bool))

        sns.heatmap(
            dunn_result,
            mask=mask,
            annot=True,
            fmt=".4f",
            cmap="RdYlGn_r",
            center=0.05,
            vmin=0,
            vmax=0.2,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        ax.set_title(
            "Dunn's Test p-value 히트맵\n"
            "(낮을수록 유의한 차이, Bonferroni 보정 적용)"
        )
        plt.tight_layout()
        plt.show()

        # -----------------------------------------------------------------------------
        # post_hoc 문자열 결합 후 반환
        # -----------------------------------------------------------------------------
        post_hoc_text = "\n".join(post_hoc)

        return dunn_result, post_hoc_text


if __name__=='__main__':
    import pandas as pd
    import numpy as np

    # 테스트 데이터 생성
    np.random.seed(42)

    # 그룹 A, B, C의 데이터 생성 (정규분포, 등분산)
    group_a = np.random.normal(loc=50, scale=10, size=30)
    group_b = np.random.normal(loc=55, scale=10, size=30)
    group_c = np.random.normal(loc=60, scale=10, size=30)

    # 데이터프레임 생성
    test_data = pd.DataFrame({
        'group': ['A']*30 + ['B']*30 + ['C']*30,
        'score': np.concatenate([group_a, group_b, group_c])
    })

    # 테스트 실행
    print("=" * 60)
    print("테스트 1: 정규분포 + 등분산 데이터 (F-test 예상)")
    print("=" * 60)

    anova = OneWayAnova()
    result1 = anova.execute(
        data=test_data,
        iv_col='group',
        dv_col='score',
        alpha=0.05
    )

    print("\n반환된 결과:")
    print(f"- 검정명: {result1.test_name}")
    print(f"- 통계량: {result1.statistic:.4f}")
    print(f"- p-value: {result1.p_value:.6f}")
    print(f"- 효과크기: {result1.effect_size}")
    print(f"- 결론: {result1.conclusion}")

    # 테스트 2: 이분산 데이터 (Welch's ANOVA 예상)
    print("\n\n" + "=" * 60)
    print("테스트 2: 정규분포 + 이분산 데이터 (Welch's ANOVA 예상)")
    print("=" * 60)

    group_a2 = np.random.normal(loc=50, scale=5, size=30)
    group_b2 = np.random.normal(loc=55, scale=15, size=30)
    group_c2 = np.random.normal(loc=60, scale=25, size=30)

    test_data2 = pd.DataFrame({
        'group': ['A']*30 + ['B']*30 + ['C']*30,
        'score': np.concatenate([group_a2, group_b2, group_c2])
    })

    result2 = anova.execute(
        data=test_data2,
        iv_col='group',
        dv_col='score',
        alpha=0.05
    )

    print("\n반환된 결과:")
    print(f"- 검정명: {result2.test_name}")
    print(f"- 통계량: {result2.statistic:.4f}")
    print(f"- p-value: {result2.p_value:.6f}")

    # 테스트 3: 비정규 데이터 (Kruskal-Wallis 예상)
    print("\n\n" + "=" * 60)
    print("테스트 3: 비정규분포 데이터 (Kruskal-Wallis 예상)")
    print("=" * 60)

    group_a3 = np.random.exponential(scale=2, size=30)
    group_b3 = np.random.exponential(scale=3, size=30)
    group_c3 = np.random.exponential(scale=4, size=30)

    test_data3 = pd.DataFrame({
        'group': ['A']*30 + ['B']*30 + ['C']*30,
        'score': np.concatenate([group_a3, group_b3, group_c3])
    })

    result3 = anova.execute(
        data=test_data3,
        iv_col='group',
        dv_col='score',
        alpha=0.05,
        mode='hard'
    )

    print("\n반환된 결과:")
    print(f"- 검정명: {result3.test_name}")
    print(f"- 통계량: {result3.statistic:.4f}")
    print(f"- p-value: {result3.p_value:.6f}")
    print(f"- 효과크기: {result3.effect_size}")
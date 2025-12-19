# =========================================================
# 통계 검정 클래스
# ========================================================="


# 1. 표준 라이브러리
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

# 2. 서드파티 라이브러리
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
import pingouin as pg

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

    def check_homogeneity(self, data: List[pd.Series]):
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
    
    def check_group_normality(self, data: List[ArrayLike], group_labels: List[str], mode: str):
        results = []
        for label, group_data in zip(group_labels, data):
            stat, p_value, is_normal = self.check_normality(group_data, group_labels, mode)
            results.append({
                '그룹': label,
                'W-통계량': round(stat, 4),
                'p-value': round(p_value, 4),
                '판정': is_normal
            })
        result_df = pd.DataFrame(results)
        display(result_df)
        
        all_normal = all(r['is_normal'] for r in results)
        return all_normal

    def check_normality(self, data: ArrayLike, label: str, mode: str):
        if mode == 'soft':
            return self.check_normality_soft(data, label)
        elif mode == 'hard':
            return self.check_normality_hard(data, label)
        else:
            msg = f'Invalid mode: "{mode}". Expected one of ["soft", "hard"].'
            raise ValueError(msg)

    def check_normality_soft(self, data: ArrayLike, label: str = "데이터"):
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

    def check_normality_hard(self, data: ArrayLike, label: str = "데이터"):
        # NaN 체크
        if pd.isna(data).any():
            print(f"⚠️ 경고: {label}에 NaN 값이 {pd.isna(data).sum()}개 포함됨")
            data = data.dropna()
            print(f"   → NaN 제거 후 n={len(data)}")

        n = len(data)

        print(f"\n[{label} 정규성 검정] n={n}")
        print("-"*40)
        
        stat, p_value = shapiro(data)
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
    def plot(class0_data, class1_data, labels, iv_col, dv_col):
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        # 박스플롯
        bp = axes[0].boxplot([class0_data, class1_data],
                            labels=labels,
                            patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
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
        v = np.sqrt(chi2_stat / (n * min(r-1, c-1)))
        
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
        
        # 가설 설정
        print(f"H₀: 모든 {iv_col}의 평균 {dv_col}이 같다")
        print(f"H₁: 적어도 한 {iv_col}의 평균 {dv_col}이 다르다")
        print(f"유의수준: α = {alpha}")
        
        group_labels = data[iv_col].unique().tolist()
        data_groups = [data[data[iv_col] == label] for label in group_labels]
        
        is_normal = self.check_normality(data_groups, group_labels, mode=mode)
        is_equal_var = self.check_homogeneity(data_groups, group_labels)
        
        if is_normal:
            print("\n✅ 모든 그룹이 정규성 가정을 만족합니다.")
        else:
            print("\n⚠️ 일부 그룹이 정규성 가정을 만족하지 않습니다.")
            print("   → 비모수 검정(Kruskal-Wallis) 고려")
    
        # 전체 표본 크기 및 그룹 수
        k = len(data_groups)
        N = sum(len(group) for group in data_groups)
        
        if is_normal:
            test_name = "f-test" if is_equal_var else "Welch's ANOVA "
            
            if is_equal_var:
                f_stat, p_value  = f_oneway(*data_groups)
                df1 = k - 1  # 집단 간 자유도
                df2 = N - k   # 집단 내 자유도
                print(f"{test_name} 결과:")
                print(f"자유도: F({df1}, {df2})")
                print(f"f = {f_stat:.4f}, p = {p_value:.4f}")     
                
                # 효과 크기 계산
                eta_sq, interpretation = self.calculate_eta_squared(f_stat, df1, df2)
                print(f"효과 크기 (η²): {eta_sq:.4f} ({interpretation})")
                print(f"   → {iv_col} 차이가 전체 {dv_col}의 {eta_sq*100:.1f}% 설명")
                effect_size = eta_sq
                effect_interpretation = interpretation
                
                if p_value < alpha:
                    conclusion = f"✅ p-value({p_value:.6f}) < 0.05 → 귀무가설 기각\n" \
                    + "   배송 서비스별 만족도에 유의한 차이가 있음" 
                else:
                    conclusion = f"❌ p-value({p_value:.6f}) ≥ 0.05 → 귀무가설 채택\n" \
                    + "   배송 서비스별 만족도에 유의한 차이가 없음"
                
            else:
                # ==========================================================================
                # Welch's ANOVA (정규성 만족, 등분산성 위반)
                # ==========================================================================
                print("\n⚠️ 등분산성 가정이 위반되었으므로 Welch's ANOVA를 수행합니다.")
                print("\n[Welch's ANOVA (이분산 ANOVA)]")
                print("-"*50)
                welch_result = pg.welch_anova(dv=dv_col, between=iv_col, data=data)
                f_stat = welch_result['F'].values[0]
                df1 = welch_result['ddof1'].values[0]
                df2 = welch_result['ddof2'].values[0]
                p_value = welch_result['p-unc'].values[0]
            
                print(f"{test_name} 결과:")
                print(f"자유도: F({df1}, {df2})")
                print(f"f = {f_stat:.4f}, p = {p_value:.4f}")
                
                print("\n※ Welch's ANOVA는 등분산성 가정을 요구하지 않으므로")
                print("   전통적인 효과 크기(η², ω²)를 직접 계산하기 어렵습니다.")
                print("   F 통계량과 p-value로 효과의 유의성을 판단하세요.")
            
            
            # Cohen's d 계산
            pooled_std = np.sqrt((class0_data.var() + class1_data.var()) / 2)
            cohens_d = (class0_data.mean() - class1_data.mean()) / pooled_std
            abs_d = abs(cohens_d)
            
            # 결론
            print("\n[검정 결론]")
            if p_value < 0.05:
                print(f"✅ p-value({p_value:.6f}) < 0.05 → 귀무가설 기각")
                print(f"   {iv_col}별 {dv_col}에 유의한 차이가 있음")
                print("   → Games-Howell 사후검정으로 구체적인 차이 확인 필요")

            else:
                print(f"❌ p-value({p_value:.6f}) ≥ 0.05 → 귀무가설 채택")
                print(f"   {iv_col}별 {dv_col}에 유의한 차이가 없음")
    
    def calculate_eta_squared(self, f_stat, df1, df2):
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
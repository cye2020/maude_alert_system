"""
통계 분석 모듈 (Python 버전)
Snowflake에서 분할표를 가져와 scipy/statsmodels로 정확한 통계 계산 후 적재
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Optional, Dict, Any
from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier


class StatisticalAnalysisPython(SnowflakeBase):
    """
    Python 기반 통계 분석
    - scipy.stats.chi2_contingency → 정확한 카이제곱 검정
    - scipy.stats.fisher_exact → 오즈비 + 정확한 p-value
    - statsmodels.multipletests → 정확한 FDR 보정 (Benjamini-Hochberg)
    """

    def __init__(self, database: str, schema: str):
        super().__init__(database, schema)

    # ──────────────────────────────────────────────
    # 1. Snowflake에서 분할표 읽기
    # ──────────────────────────────────────────────
    @with_context
    def fetch_contingency(
        self,
        cursor: SnowflakeCursor,
        source_table: str,
        row_column: str,
        col_column: str,
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        validate_identifier(row_column)
        validate_identifier(col_column)
        where_clause = f"WHERE {filters}" if filters else ""
        sql = f"""
            SELECT {row_column}, {col_column}, COUNT(*) AS observed
            FROM {source_table}
            {where_clause}
            GROUP BY {row_column}, {col_column}
            ORDER BY {row_column}, {col_column}
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    # ──────────────────────────────────────────────
    # 2. 카이제곱 검정 (scipy)
    # ──────────────────────────────────────────────
    def chi2_test(
        self,
        df: pd.DataFrame,
        row_column: str,
        col_column: str,
    ) -> Dict[str, Any]:
        """제품코드별 2×N 카이제곱 검정.

        각 제품코드에 대해 [이 제품, 나머지 합산] × 결함유형 분할표를 만들어
        독립적으로 검정합니다. 제품코드마다 고유한 CHI2_STATISTIC / P_VALUE /
        CRAMERS_V 가 부여됩니다.
        """
        pivot = df.pivot_table(
            index=row_column, columns=col_column,
            values="OBSERVED", fill_value=0, aggfunc="sum",
        )
        matrix     = pivot.values
        row_labels = pivot.index.tolist()
        col_labels = pivot.columns.tolist()
        col_totals = matrix.sum(axis=0)

        def _effect(v: float) -> str:
            if v < 0.1:  return "Very Weak"
            if v < 0.3:  return "Weak"
            if v < 0.5:  return "Moderate"
            return "Strong"

        rows_list = []
        for i, r in enumerate(row_labels):
            this_row = matrix[i]
            rest_row = col_totals - this_row
            sub      = np.array([this_row, rest_row])

            chi2, p_value, dof, expected = stats.chi2_contingency(sub)
            n         = int(sub.sum())
            cramers_v = np.sqrt(chi2 / n) if n > 0 else 0.0  # k=1 (2행 분할표)

            for j, c in enumerate(col_labels):
                obs = this_row[j]
                exp = expected[0, j]
                if obs == 0 and exp == 0:
                    continue
                std_residual = (obs - exp) / np.sqrt(exp) if exp > 0 else 0.0
                chi2_comp    = ((obs - exp) ** 2) / exp if exp > 0 else 0.0
                rows_list.append({
                    row_column        : r,
                    col_column        : c,
                    "OBSERVED"        : int(obs),
                    "EXPECTED"        : round(exp, 6),
                    "STD_RESIDUAL"    : round(std_residual, 10),
                    "CHI2_COMPONENT"  : round(chi2_comp, 10),
                    "CHI2_STATISTIC"  : round(chi2, 10),
                    "DF"              : int(dof),
                    "P_VALUE"         : p_value,
                    "CRAMERS_V"       : round(cramers_v, 10),
                    "EFFECT_SIZE"     : _effect(cramers_v),
                    "TOTAL_N"         : n,
                })

        return {"detail": pd.DataFrame(rows_list)}

    # ──────────────────────────────────────────────
    # 3. 오즈비 계산
    # ──────────────────────────────────────────────
    def odds_ratios(
        self,
        df: pd.DataFrame,
        row_column: str,
        col_column: str,
        min_cell_count: int = 3,
    ) -> pd.DataFrame:
        pivot = df.pivot_table(
            index=row_column, columns=col_column,
            values="OBSERVED", fill_value=0, aggfunc="sum",
        )
        matrix = pivot.values
        total = matrix.sum()
        row_labels = pivot.index.tolist()
        col_labels = pivot.columns.tolist()
        row_sums = matrix.sum(axis=1)
        col_sums = matrix.sum(axis=0)

        rows_list = []
        for i, r in enumerate(row_labels):
            for j, c in enumerate(col_labels):
                a = int(matrix[i, j])
                b = int(row_sums[i] - a)
                c_val = int(col_sums[j] - a)
                d = int(total - a - b - c_val)

                if min(a, b, c_val, d) < min_cell_count:
                    continue

                # 2x2 테이블로 fisher exact
                table_2x2 = np.array([[a, b], [c_val, d]])
                odds_ratio, p_value = stats.fisher_exact(table_2x2)

                # Haldane-Anscombe correction (신뢰구간용)
                if a == 0 or b == 0 or c_val == 0 or d == 0:
                    aa, bb, cc, dd = a + 0.5, b + 0.5, c_val + 0.5, d + 0.5
                else:
                    aa, bb, cc, dd = a, b, c_val, d

                se_log_or = np.sqrt(1/aa + 1/bb + 1/cc + 1/dd)
                log_or = np.log(aa * dd / (bb * cc))
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)

                rows_list.append({
                    row_column: r,
                    col_column: c,
                    "A": a, "B": b, "C": c_val, "D": d,
                    "MIN_CELL_COUNT": min(a, b, c_val, d),
                    "ODDS_RATIO": round(odds_ratio, 10),
                    "P_VALUE": p_value,
                    "SE_LOG_OR": round(se_log_or, 10),
                    "CI_LOWER": round(ci_lower, 10),
                    "CI_UPPER": round(ci_upper, 10),
                })

        result_df = pd.DataFrame(rows_list)
        if not result_df.empty:
            result_df = result_df.sort_values("P_VALUE").reset_index(drop=True)
        return result_df

    # ──────────────────────────────────────────────
    # 4. FDR 보정 (Benjamini-Hochberg)
    # ──────────────────────────────────────────────
    def fdr_correction(
        self,
        odds_df: pd.DataFrame,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        if odds_df.empty:
            return odds_df

        reject, p_corrected, _, _ = multipletests(
            odds_df["P_VALUE"], alpha=alpha, method="fdr_bh"
        )
        result = odds_df.copy()
        result["P_VALUE_CORRECTED"] = p_corrected
        result["SIGNIFICANT_CORRECTED"] = reject
        result["SIGNIFICANT_RAW"] = result["P_VALUE"] < alpha
        result["RANK"] = range(1, len(result) + 1)
        return result

    # ──────────────────────────────────────────────
    # 5. 통계 분석 실행 (계산만, 적재 X)
    # ──────────────────────────────────────────────
    def run(
        self,
        cursor: SnowflakeCursor,
        source_table: str,
        row_column: str,
        col_column: str,
        filters: Optional[str] = None,
        min_cell_count: int = 3,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        timings = {}

        # 1. 분할표 가져오기
        t0 = time.time()
        contingency_df = self.fetch_contingency(
            cursor, source_table, row_column, col_column, filters
        )
        timings["fetch_contingency"] = time.time() - t0

        # 2. 카이제곱 검정
        t0 = time.time()
        chi2_result = self.chi2_test(contingency_df, row_column, col_column)
        timings["chi2_test"] = time.time() - t0

        # 3. 오즈비 계산
        t0 = time.time()
        odds_df = self.odds_ratios(
            contingency_df, row_column, col_column, min_cell_count
        )
        timings["odds_ratios"] = time.time() - t0

        # 4. FDR 보정
        t0 = time.time()
        final_df = self.fdr_correction(odds_df, alpha)
        timings["fdr_correction"] = time.time() - t0

        timings["total"] = sum(timings.values())

        return {
            "chi2"       : {"detail": chi2_result["detail"]},
            "odds_ratios": odds_df,
            "final"      : final_df,
            "timings"    : timings,
        }


# ============================================================================
# 테스트 실행 (__main__)
# ============================================================================

if __name__ == "__main__":
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret

    # ------------------------------------------------------------------
    # 1. Snowflake 연결 (silver/credentials는 읽기 전용 → snowflake/de 사용)
    # ------------------------------------------------------------------
    secret = get_secret('snowflake/de')
    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
    )
    cursor = conn.cursor()

    # ------------------------------------------------------------------
    # 2. 통계 분석 실행
    #    - EVENT_STAGE_12_COMBINED 에서 PRODUCT_CODE, DEFECT_TYPE 가져옴
    #    - Python(scipy/statsmodels)으로 chi2, 오즈비, FDR 계산
    # ------------------------------------------------------------------
    try:
        analyzer = StatisticalAnalysisPython(database="MAUDE", schema="SILVER")
        result = analyzer.run(
            cursor=cursor,
            source_table="EVENT_STAGE_12_COMBINED",
            row_column="PRODUCT_CODE",
            col_column="DEFECT_TYPE",
            filters="DEFECT_TYPE NOT IN ('Unknown', 'Other')",
            min_cell_count=3,
            alpha=0.05,
        )

        print("=" * 60)
        print("Python 통계 분석 결과")
        print("=" * 60)
        chi2_detail = result["chi2"]["detail"]
        print(f"  Chi2 검정 행 수: {len(chi2_detail)} (제품코드별 2×N 검정)")
        print(f"  오즈비 수:       {len(result['odds_ratios'])}")
        final_df  = result["final"]
        sig_count = int(final_df["SIGNIFICANT_CORRECTED"].sum()) if not final_df.empty else 0
        print(f"  유의미 수:       {sig_count} (FDR 보정)")

        print("\n[소요 시간]")
        for step, sec in result["timings"].items():
            print(f"  {step:20} → {sec:.3f}s")

    finally:
        cursor.close()
        conn.close()

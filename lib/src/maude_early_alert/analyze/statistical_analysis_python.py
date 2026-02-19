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


class StatisticalAnalysisPython:
    """
    Python 기반 통계 분석
    - scipy.stats.chi2_contingency → 정확한 카이제곱 검정
    - scipy.stats.fisher_exact → 오즈비 + 정확한 p-value
    - statsmodels.multipletests → 정확한 FDR 보정 (Benjamini-Hochberg)
    """

    # ──────────────────────────────────────────────
    # 1. Snowflake에서 분할표 읽기
    # ──────────────────────────────────────────────
    def fetch_contingency(
        self,
        cursor: SnowflakeCursor,
        source_table: str,
        row_column: str,
        col_column: str,
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
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
        pivot = df.pivot_table(
            index=row_column, columns=col_column,
            values="OBSERVED", fill_value=0, aggfunc="sum",
        )
        matrix = pivot.values

        chi2, p_value, dof, expected = stats.chi2_contingency(matrix)
        n = matrix.sum()
        k = min(matrix.shape) - 1
        cramers_v = np.sqrt(chi2 / (n * k)) if k > 0 else 0.0

        if cramers_v < 0.1:
            effect = "Very Weak"
        elif cramers_v < 0.3:
            effect = "Weak"
        elif cramers_v < 0.5:
            effect = "Moderate"
        else:
            effect = "Strong"

        # 셀별 상세 결과
        rows_list = []
        row_labels = pivot.index.tolist()
        col_labels = pivot.columns.tolist()
        for i, r in enumerate(row_labels):
            for j, c in enumerate(col_labels):
                obs = matrix[i, j]
                exp = expected[i, j]
                if obs == 0 and exp == 0:
                    continue
                std_residual = (obs - exp) / np.sqrt(exp) if exp > 0 else 0.0
                chi2_comp = ((obs - exp) ** 2) / exp if exp > 0 else 0.0
                rows_list.append({
                    row_column: r,
                    col_column: c,
                    "OBSERVED": int(obs),
                    "EXPECTED": round(exp, 6),
                    "STD_RESIDUAL": round(std_residual, 10),
                    "CHI2_COMPONENT": round(chi2_comp, 10),
                    "CHI2_STATISTIC": round(chi2, 10),
                    "DF": int(dof),
                    "P_VALUE": p_value,
                    "CRAMERS_V": round(cramers_v, 10),
                    "EFFECT_SIZE": effect,
                    "TOTAL_N": int(n),
                })

        result_df = pd.DataFrame(rows_list)
        return {
            "chi2_statistic": chi2,
            "p_value": p_value,
            "dof": dof,
            "cramers_v": cramers_v,
            "effect_size": effect,
            "detail": result_df,
        }

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
    # 5. 결과 Snowflake 적재
    # ──────────────────────────────────────────────
    def write_to_snowflake(
        self,
        cursor: SnowflakeCursor,
        df: pd.DataFrame,
        table_name: str,
    ) -> int:
        if df.empty:
            return 0

        cols = df.columns.tolist()
        col_defs = []
        for col in cols:
            dtype = df[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                col_defs.append(f"{col} BOOLEAN")
            elif pd.api.types.is_integer_dtype(dtype):
                col_defs.append(f"{col} INTEGER")
            elif pd.api.types.is_float_dtype(dtype):
                col_defs.append(f"{col} DOUBLE")
            else:
                col_defs.append(f"{col} VARCHAR")

        create_sql = f"CREATE OR REPLACE TABLE {table_name} ({', '.join(col_defs)})"
        cursor.execute(create_sql)

        placeholders = ", ".join(["%s"] * len(cols))
        insert_sql = f"INSERT INTO {table_name} ({', '.join(cols)}) VALUES ({placeholders})"

        data = []
        for _, row in df.iterrows():
            values = []
            for col in cols:
                val = row[col]
                if pd.isna(val):
                    values.append(None)
                elif isinstance(val, (np.bool_, bool)):
                    values.append(bool(val))
                elif isinstance(val, (np.integer,)):
                    values.append(int(val))
                elif isinstance(val, (np.floating,)):
                    values.append(float(val))
                else:
                    values.append(str(val))
            data.append(values)

        cursor.executemany(insert_sql, data)
        return len(data)

    # ──────────────────────────────────────────────
    # 6. 통계 분석 실행 (계산만, 적재 X)
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
            "chi2": {
                "chi2_statistic": chi2_result["chi2_statistic"],
                "p_value": chi2_result["p_value"],
                "dof": chi2_result["dof"],
                "cramers_v": chi2_result["cramers_v"],
                "effect_size": chi2_result["effect_size"],
                "detail": chi2_result["detail"],
            },
            "odds_ratios": odds_df,
            "final": final_df,
            "timings": timings,
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
    # 1-1. Snowflake 컨텍스트 설정 (파이프라인과 동일)
    # ------------------------------------------------------------------
    cursor.execute("USE DATABASE MAUDE")
    cursor.execute("USE SCHEMA SILVER")

    # ------------------------------------------------------------------
    # 2. 통계 분석 실행
    #    - EVENT_STAGE_12_COMBINED 에서 PRODUCT_CODE, DEFECT_TYPE 가져옴
    #    - Python(scipy/statsmodels)으로 chi2, 오즈비, FDR 계산
    # ------------------------------------------------------------------
    try:
        analyzer = StatisticalAnalysisPython()
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
        chi2 = result["chi2"]
        print(f"  Chi2 통계량: {chi2['chi2_statistic']:.4f}")
        print(f"  p-value:     {chi2['p_value']:.2e}")
        print(f"  자유도:      {chi2['dof']}")
        print(f"  Cramer's V:  {chi2['cramers_v']:.4f} ({chi2['effect_size']})")
        print(f"  오즈비 수:   {len(result['odds_ratios'])}")
        final_df = result["final"]
        sig_count = int(final_df["SIGNIFICANT_CORRECTED"].sum()) if not final_df.empty else 0
        print(f"  유의미 수:   {sig_count} (FDR 보정)")

        print("\n[소요 시간]")
        for step, sec in result["timings"].items():
            print(f"  {step:20} → {sec:.3f}s")

        # ------------------------------------------------------------------
        # 3. Snowflake 적재
        # ------------------------------------------------------------------
        output_prefix = "MAUDE_STAT"

        chi2_table = f"{output_prefix}_CHI2_RESULT"
        analyzer.write_to_snowflake(cursor, chi2["detail"], chi2_table)

        odds_table = f"{output_prefix}_ODDS_RATIOS"
        analyzer.write_to_snowflake(cursor, result["odds_ratios"], odds_table)

        final_table = f"{output_prefix}_FINAL"
        analyzer.write_to_snowflake(cursor, final_df, final_table)

        print("\n[적재 완료]")
        print(f"  chi2_result   → {chi2_table} ({len(chi2['detail'])}건)")
        print(f"  odds_ratios   → {odds_table} ({len(result['odds_ratios'])}건)")
        print(f"  final         → {final_table} ({len(final_df)}건)")

    finally:
        cursor.close()
        conn.close()

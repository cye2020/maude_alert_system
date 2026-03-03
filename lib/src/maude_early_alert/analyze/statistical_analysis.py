"""
통계 분석 SQL 생성 모듈
statistic.py의 Chi2Test 로직을 Snowflake SQL로 변환
"""
from typing import Optional, Dict
from textwrap import dedent


class StatisticalAnalysis:
    """
    - Chi2Test.execute() 
    - calculate_odds_ratio()
    - FDR 보정 
    """
    
    def build_chi2_test_sql(
        self,
        contingency_table: str,
        row_column: str,
        col_column: str,
        output_table: str = "CHI2_TEST_RESULT"
    ) -> str:
        """
        Parameters
        ----------
        contingency_table : str
            분할표 테이블 (observed 컬럼 필요)
        row_column : str
            독립변수 컬럼 (예: PRODUCT_CODE)
        col_column : str
            종속변수 컬럼 (예: DEFECT_TYPE)
        output_table : str
            결과 테이블명

        Returns
        -------
        str
            카이제곱 검정 결과 SQL
        """
        chi2_result_table = output_table

        # 2. 카이제곱 검정
        sql_chi2_result = dedent(f"""\
            CREATE OR REPLACE TABLE {chi2_result_table} AS
            WITH 
            -- 전체 샘플 수 
            total AS (
                SELECT SUM(observed) AS n 
                FROM {contingency_table}
            ),

            -- 행 / 열 합계
            row_totals AS (
                SELECT
                    {row_column},
                    SUM(observed) AS row_sum
                FROM {contingency_table}
                GROUP BY {row_column}    
            ),

            col_totals AS (
                SELECT
                    {col_column},
                    SUM(observed) AS col_sum
                FROM {contingency_table}
                GROUP BY {col_column}
            ),

            -- 기대빈도 계산: E = (row_sum * col_sum) / n
            expected_freq AS (
                SELECT 
                    c.{row_column},
                    c.{col_column},
                    c.observed,
                    (r.row_sum * col.col_sum * 1.0 / t.n) AS expected
                FROM {contingency_table} c
                JOIN row_totals r ON c.{row_column} = r.{row_column}
                JOIN col_totals col ON c.{col_column} = col.{col_column}
                CROSS JOIN total t
            ),

            -- 카이제곱 통계량
            chi2_components AS (
                SELECT
                    {row_column},
                    {col_column},
                    observed,
                    expected,
                    POWER(observed - expected, 2) / expected AS chi2_component,

                    -- 표준화 잔차
                    (observed - expected) / SQRT(expected) AS std_residual
                FROM expected_freq   
            ),

            -- 전체 카이제곱 통계량
            chi2_stat AS (
                SELECT
                    SUM(chi2_component) AS chi2_statistic,
                    -- 자유도
                    (COUNT(DISTINCT {row_column}) - 1) * (COUNT(DISTINCT {col_column}) - 1) AS df,
                    COUNT(DISTINCT {row_column}) AS n_rows,
                    COUNT(DISTINCT {col_column}) AS n_cols
                FROM chi2_components
            ),

            -- p-value 근사
            p_value_approx AS (
                SELECT
                    chi2_statistic,
                    df, 
                    n_rows,
                    n_cols,
                    CASE
                        WHEN df = 1 THEN
                            CASE
                                WHEN chi2_statistic > 10.83 THEN 0.001
                                WHEN chi2_statistic > 7.88 THEN 0.005
                                WHEN chi2_statistic > 6.63 THEN 0.01
                                WHEN chi2_statistic > 3.84 THEN 0.05
                                ELSE 0.5
                            END
                        WHEN df = 2 THEN
                            CASE
                                WHEN chi2_statistic > 13.82 THEN 0.001
                                WHEN chi2_statistic > 10.60 THEN 0.005
                                WHEN chi2_statistic > 9.21 THEN 0.01
                                WHEN chi2_statistic > 5.99 THEN 0.05
                                ELSE 0.5
                            END
                        WHEN df >= 3 THEN
                            CASE
                                WHEN chi2_statistic > (df + 3 * SQRT(2.0 * df)) THEN 0.001
                                WHEN chi2_statistic > (df + 2.5 * SQRT(2.0 * df)) THEN 0.01
                                WHEN chi2_statistic > (df + 2 * SQRT(2.0 * df)) THEN 0.05
                                ELSE 0.5
                            END
                    END AS p_value_approx
                FROM chi2_stat
            ),

            -- Cramer's V
            cramers_v_calc AS (
                SELECT
                    c.*,
                    p.p_value_approx,
                    SQRT(
                        c.chi2_statistic / (
                            (SELECT n FROM total) * LEAST(c.n_rows - 1, c.n_cols - 1)
                        )
                    ) AS cramers_v,

                    -- 효과 크기 해석
                    CASE
                        WHEN SQRT(c.chi2_statistic / ((SELECT n FROM total) * LEAST(c.n_rows - 1, c.n_cols - 1))) < 0.1 
                            THEN 'Very Weak'
                        WHEN SQRT(c.chi2_statistic / ((SELECT n FROM total) * LEAST(c.n_rows - 1, c.n_cols - 1))) < 0.3 
                            THEN 'Weak'
                        WHEN SQRT(c.chi2_statistic / ((SELECT n FROM total) * LEAST(c.n_rows - 1, c.n_cols - 1))) < 0.5 
                            THEN 'Moderate'
                        ELSE 'Strong'
                    END AS effect_size_interpretation
                FROM chi2_stat c
                CROSS JOIN p_value_approx p
            )

            -- 최종 결과 (각 셀별 정보 + 전체 통계)
            SELECT
                comp.{row_column},
                comp.{col_column},
                comp.observed,
                comp.expected,
                comp.std_residual,
                comp.chi2_component,
                cv.chi2_statistic,
                cv.df,
                cv.p_value_approx,
                cv.cramers_v,
                cv.effect_size_interpretation,
                (SELECT n FROM total) AS total_n
            FROM chi2_components comp
            CROSS JOIN cramers_v_calc cv
            ORDER BY comp.{row_column}, comp.{col_column}
        """)

        return sql_chi2_result

    def build_contingency_table_sql(
        self,
        source_table: str,
        row_column: str,
        col_column: str,
        output_table: str,
        filters: Optional[str] = None
    ) -> str:
        where_clause = f"WHERE\n        {filters}" if filters else ""
        
        sql = dedent(f"""\
            CREATE OR REPLACE TABLE {output_table} AS
            SELECT
                {row_column},
                {col_column},
                COUNT(*) AS observed
            FROM
                {source_table}
            {where_clause}
            GROUP BY
                {row_column},
                {col_column}
            ORDER BY
                {row_column},
                {col_column}
        """)
        
        return sql
    
    def build_odds_ratios_sql(
        self,
        contingency_table: str,
        row_column: str,
        col_column: str,
        output_table: str,
        min_cell_count: int = 3
    ) -> str:
        sql = dedent(f"""\
            CREATE OR REPLACE TABLE {output_table} AS
            WITH
            -- 전체 합계
            total AS (
                SELECT SUM(observed) AS total_count
                FROM {contingency_table}
            ),

            -- 각 조합별 a, b, c, d 계산
            odds_calc AS (
                SELECT
                    t.{row_column},
                    t.{col_column},

                    -- a: 타겟 제품 & 타겟 결함
                    t.observed AS a,

                    -- b: 타겟 제품 & 기타 결함
                    (SELECT SUM(observed)
                     FROM {contingency_table}
                     WHERE {row_column} = t.{row_column}
                    ) - t.observed AS b,

                    -- c: 기타 제품 & 타겟 결함
                    (SELECT SUM(observed)
                     FROM {contingency_table}
                     WHERE {col_column} = t.{col_column}
                    ) - t.observed AS c,

                    -- d: 기타 제품 & 기타 결함
                    total.total_count - (
                        t.observed +
                        ((SELECT SUM(observed) FROM {contingency_table} WHERE {row_column} = t.{row_column}) - t.observed) +
                        ((SELECT SUM(observed) FROM {contingency_table} WHERE {col_column} = t.{col_column}) - t.observed)
                    ) AS d,
                    
                    total.total_count
                FROM {contingency_table} t
                CROSS JOIN total
            ),
            
            -- 연속성 수정 (Haldane-Anscombe correction)
            adjusted AS (
                SELECT
                    {row_column},
                    {col_column},
                    a, b, c, d,
                    
                    CASE 
                        WHEN a = 0 OR b = 0 OR c = 0 OR d = 0 
                        THEN a + 0.5 
                        ELSE a 
                    END AS a_adj,
                    CASE 
                        WHEN a = 0 OR b = 0 OR c = 0 OR d = 0 
                        THEN b + 0.5 
                        ELSE b 
                    END AS b_adj,
                    CASE 
                        WHEN a = 0 OR b = 0 OR c = 0 OR d = 0 
                        THEN c + 0.5 
                        ELSE c 
                    END AS c_adj,
                    CASE 
                        WHEN a = 0 OR b = 0 OR c = 0 OR d = 0 
                        THEN d + 0.5 
                        ELSE d 
                    END AS d_adj,
                    
                    LEAST(a, b, c, d) AS min_cell_count
                FROM odds_calc
            )
            
            SELECT
                {row_column},
                {col_column},
                a, b, c, d,
                min_cell_count,
                
                -- 오즈비: OR = (a*d) / (b*c)
                (a_adj * d_adj) / (b_adj * c_adj) AS odds_ratio,
                
                -- 로그 오즈비의 표준오차
                SQRT(
                    1.0/a_adj + 1.0/b_adj + 1.0/c_adj + 1.0/d_adj
                ) AS se_log_or,
                
                -- 95% 신뢰구간
                EXP(
                    LN((a_adj * d_adj) / (b_adj * c_adj)) - 1.96 * SQRT(1.0/a_adj + 1.0/b_adj + 1.0/c_adj + 1.0/d_adj)
                ) AS ci_lower,
                
                EXP(
                    LN((a_adj * d_adj) / (b_adj * c_adj)) + 1.96 * SQRT(1.0/a_adj + 1.0/b_adj + 1.0/c_adj + 1.0/d_adj)
                ) AS ci_upper,
                
                -- 카이제곱 통계량 (Yates correction)
                POWER(ABS(a * d - b * c) - (a + b + c + d) / 2.0, 2) * (a + b + c + d) 
                / ((a + b) * (c + d) * (a + c) * (b + d)) AS chi2_statistic
                
            FROM adjusted
            WHERE min_cell_count >= {min_cell_count}
            ORDER BY chi2_statistic DESC
        """)
        
        return sql
    
    def build_fdr_correction_sql(
        self,
        odds_ratio_table: str,
        output_table: str,
        alpha: float = 0.05
    ) -> str:
        sql = dedent(f"""\
            CREATE OR REPLACE TABLE {output_table} AS
            WITH
            -- 카이제곱 통계량 → p-value 근사
            p_values AS (
                SELECT
                    *,
                    -- 근사 p-value (자유도 1)
                    CASE 
                        WHEN chi2_statistic > 10.83 THEN 0.001
                        WHEN chi2_statistic > 7.88 THEN 0.005
                        WHEN chi2_statistic > 6.63 THEN 0.01
                        WHEN chi2_statistic > 3.84 THEN 0.05
                        ELSE 0.5
                    END AS p_value_approx,
                    
                    ROW_NUMBER() OVER (ORDER BY chi2_statistic DESC) AS rank,
                    COUNT(*) OVER () AS total_tests
                FROM {odds_ratio_table}
            )
            
            SELECT
                *,
                -- FDR 보정: p_corrected = p * n / rank
                LEAST(p_value_approx * total_tests / rank, 1.0) AS p_value_corrected,
                
                -- 유의성 판단
                CASE 
                    WHEN p_value_approx * total_tests / rank < {alpha} THEN TRUE 
                    ELSE FALSE 
                END AS significant_corrected,
                
                CASE 
                    WHEN p_value_approx < {alpha} THEN TRUE 
                    ELSE FALSE 
                END AS significant_raw
                
            FROM p_values
            ORDER BY rank
        """)
        
        return sql
    
    def build_full_analysis_sql(
        self,
        source_table: str,
        row_column: str,
        col_column: str,
        output_prefix: str = "STATISTICAL_ANALYSIS",
        filters: Optional[str] = None,
        min_cell_count: int = 3,
        alpha: float = 0.05,
        include_chi2_test: bool = True
    ) -> Dict[str, str]:
        results = {}
        sql_statements = []

        # 1. 분할표 (공용 - chi2 + 오즈비에서 공유)
        contingency_table = f"{output_prefix}_CONTINGENCY"
        sql_contingency = self.build_contingency_table_sql(
            source_table, row_column, col_column,
            contingency_table, filters
        )
        results['contingency'] = sql_contingency
        sql_statements.append(sql_contingency)

        # 2. Chi2Test (전체 독립성 검정)
        if include_chi2_test:
            chi2_result_table = f"{output_prefix}_CHI2_RESULT"
            sql_chi2 = self.build_chi2_test_sql(
                contingency_table, row_column, col_column,
                chi2_result_table
            )
            results['chi2_test'] = sql_chi2
            sql_statements.append(sql_chi2)

        # 3. 개별 오즈비
        odds_table = f"{output_prefix}_ODDS_RATIOS"
        sql_odds = self.build_odds_ratios_sql(
            contingency_table, row_column, col_column,
            odds_table, min_cell_count
        )
        results['odds_ratios'] = sql_odds
        sql_statements.append(sql_odds)

        # 4. FDR 보정
        final_table = f"{output_prefix}_FINAL"
        sql_fdr = self.build_fdr_correction_sql(odds_table, final_table, alpha)
        results['fdr_correction'] = sql_fdr
        sql_statements.append(sql_fdr)

        # 5. 전체 SQL
        results['combined'] = ";\n\n".join(sql_statements) + ";"
        results['tables'] = {
            'contingency': contingency_table,
            'odds': odds_table,
            'final': final_table
        }

        if include_chi2_test:
            results['tables']['chi2_result'] = chi2_result_table

        return results

# =====================
# 사용 예시
# =====================

if __name__ == "__main__":
    print("=" * 80)
    print("통계 분석 SQL 생성 (statistic.py → SQL)")
    print("=" * 80)
    
    sql_builder = StatisticalAnalysis()
    
    # 전체 분석 SQL 생성
    sqls = sql_builder.build_full_analysis_sql(
        source_table="MAUDE.SILVER.EVENT_STAGE_06",
        row_column="PRODUCT_CODE",
        col_column="DEFECT_TYPE",
        output_prefix="MAUDE_STAT",
        filters="DEFECT_TYPE NOT IN ('Unknown', 'Other')",
        min_cell_count=3,
        alpha=0.05,
        include_chi2_test=True
    )
    
    print("\n[생성된 테이블]")
    print("=" * 80)
    for name, table in sqls['tables'].items():
        print(f"  {name:20} → {table}")
    
    # ============================================================
    # SQL만 깔끔하게 출력
    # ============================================================
    
    # print("\n" + "=" * 80)
    # print("[1/4] 분할표 (공용)")
    # print("=" * 80)
    # print(sqls['contingency'])

    # print("\n" + "=" * 80)
    # print("[2/4] Chi2 Test - 카이제곱 검정")
    # print("=" * 80)
    # print(sqls['chi2_test'])
    
    # print("\n" + "=" * 80)
    # print("[3/4] 오즈비 계산")
    # print("=" * 80)
    # print(sqls['odds_ratios'])
    
    print("\n" + "=" * 80)
    print("[4/4] FDR 보정")
    print("=" * 80)
    print(sqls['fdr_correction'])
    
    # print("\n" + "=" * 80)
    # print("[전체 SQL (순차 실행용)]")
    # print("=" * 80)
    # print(sqls['combined'])
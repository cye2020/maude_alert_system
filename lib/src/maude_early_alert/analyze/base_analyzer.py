from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier


def make_pairs(col_lst: List[str], fixed_col: str = None) -> List[tuple]:
    """
    분석할 열 쌍 목록 생성
    fixed_col이 있으면 fixed_col과 나머지 열의 조합만 생성
    없으면 col_lst 내 모든 조합 생성
    """
    if fixed_col:
        return [(fixed_col, col) for col in col_lst if col != fixed_col]
    else:
        return [(col_a, col_b) for i, col_a in enumerate(col_lst)
                               for col_b in col_lst[i+1:]]


class BaseAnalyzer(SnowflakeBase):
    """
    열 간 관계 분석 (비율 계산 + 시각화)
    - Snowflake에서 데이터를 직접 읽어와 분석
    """

    def __init__(self, database: str, schema: str):
        super().__init__(database, schema)

    @with_context
    def fetch_data(
        self,
        cursor,
        source_table: str,
        col_a: str,
        col_b: str,
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        """Snowflake에서 분석할 두 열만 읽어오기"""
        validate_identifier(col_a)
        validate_identifier(col_b)
        where_clause = f"WHERE {filters}" if filters else ""
        sql = f"""
            SELECT {col_a}, {col_b}
            FROM {source_table}
            {where_clause}
        """
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=columns)

    def run(
        self,
        cursor,
        source_table: str,
        col_a: str,
        col_b: str,
        filters: Optional[str] = None,
        top_n: int = None,
    ) -> pd.DataFrame:
        """한 쌍에 대한 비율 계산 및 시각화"""
        df = self.fetch_data(cursor, source_table, col_a, col_b, filters)
        ratio = self._calc_ratio(df, col_a, col_b)
        self._visualize(col_a, col_b, ratio, top_n)
        return ratio

    def _calc_ratio(self, df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
        """col_a 각 값 안에서 col_b의 구성 비율 계산"""
        count_df = (
            df.groupby([col_a, col_b])
            .size()
            .reset_index(name="count")
        )
        total_per_col_a = count_df.groupby(col_a)["count"].transform("sum")
        count_df["ratio"] = count_df["count"] / total_per_col_a
        return count_df

    def _visualize(self, col_a: str, col_b: str, ratio_df: pd.DataFrame, top_n: int = None):
        """col_a 각 값마다 col_b 구성 비율을 파이차트 + 막대그래프로 출력"""
        if top_n:
            top_values = (
                ratio_df.groupby(col_a)["count"]
                .sum()
                .nlargest(top_n)
                .index
            )
            ratio_df = ratio_df[ratio_df[col_a].isin(top_values)]

        col_a_values = ratio_df[col_a].unique()

        fig, axes = plt.subplots(
            nrows=len(col_a_values),
            ncols=2,
            figsize=(12, 4 * len(col_a_values))
        )

        if len(col_a_values) == 1:
            axes = [axes]

        for i, val in enumerate(col_a_values):
            subset = ratio_df[ratio_df[col_a] == val]

            axes[i][0].pie(
                subset["ratio"],
                labels=subset[col_b],
                autopct="%1.1f%%"
            )
            axes[i][0].set_title(f"{col_a} = {val}")

            axes[i][1].bar(subset[col_b], subset["ratio"])
            axes[i][1].set_title(f"{col_a} = {val}")
            axes[i][1].set_ylabel("ratio")
            axes[i][1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
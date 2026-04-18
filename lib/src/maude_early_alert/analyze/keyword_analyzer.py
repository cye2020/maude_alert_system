import ast
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.utils.helpers import validate_identifier


class KeywordAnalyzer(SnowflakeBase):
    """
    키워드 리스트 열과 클러스터 간 관계 분석
    - Snowflake에서 데이터를 직접 읽어와 분석
    """

    def __init__(self, database: str, schema: str):
        super().__init__(database, schema)

    @with_context
    def fetch_data(
        self,
        cursor,
        source_table: str,
        col_name: str,
        cluster_col: str = "cluster",
        filters: Optional[str] = None,
    ) -> pd.DataFrame:
        """Snowflake에서 키워드 열과 클러스터 열만 읽어오기"""
        validate_identifier(col_name)
        validate_identifier(cluster_col)
        where_clause = f"WHERE {filters}" if filters else ""
        sql = f"""
            SELECT {cluster_col}, {col_name}
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
        col_name: str,
        cluster_col: str = "cluster",
        filters: Optional[str] = None,
        top_n: int = 10,
    ) -> pd.DataFrame:
        """키워드 열과 클러스터 관계 분석 및 출력"""
        df = self.fetch_data(cursor, source_table, col_name, cluster_col, filters)
        exploded = self._explode_keyword_col(df, col_name, cluster_col)
        result = self._calc_keyword_freq(exploded, col_name, cluster_col)
        self._print_summary(result, col_name, cluster_col, top_n)
        self._visualize(result, col_name, cluster_col, top_n)
        return result

    def _explode_keyword_col(self, df: pd.DataFrame, col_name: str, cluster_col: str) -> pd.DataFrame:
        """리스트 열을 explode하여 키워드별로 분리"""
        df_temp = df[[cluster_col, col_name]].copy()
        if df_temp[col_name].dtype == object:
            df_temp[col_name] = df_temp[col_name].map(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        df_temp = df_temp.explode(col_name)
        df_temp[col_name] = df_temp[col_name].str.lower().str.strip()
        df_temp = df_temp[df_temp[col_name].notna() & (df_temp[col_name] != "")]
        return df_temp

    def _calc_keyword_freq(self, exploded_df: pd.DataFrame, col_name: str, cluster_col: str) -> pd.DataFrame:
        """클러스터별 키워드 빈도 계산"""
        result = (
            exploded_df
            .groupby([cluster_col, col_name])
            .size()
            .reset_index(name="count")
            .sort_values([cluster_col, "count"], ascending=[True, False])
        )
        return result

    def _print_summary(self, result: pd.DataFrame, col_name: str, cluster_col: str, top_n: int):
        """클러스터별 키워드 빈도 요약 출력"""
        for cluster_id in sorted(result[cluster_col].unique()):
            subset = result[result[cluster_col] == cluster_id]
            label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"

            print(f"\n{'='*60}")
            print(f"{label}")
            print(f"{'='*60}")
            print(f"총 키워드 수: {subset['count'].sum():,}")
            print(f"고유 키워드 수: {len(subset):,}")
            print(f"\nTop {top_n} 키워드:")
            for _, row in subset.head(top_n).iterrows():
                print(f"  {row[col_name]:30s}: {row['count']:>6,}")

    def _visualize(self, result: pd.DataFrame, col_name: str, cluster_col: str, top_n: int):
        """클러스터별 상위 키워드 막대그래프 출력"""
        cluster_ids = sorted(result[cluster_col].unique())

        fig, axes = plt.subplots(
            nrows=len(cluster_ids),
            ncols=1,
            figsize=(12, 4 * len(cluster_ids))
        )

        if len(cluster_ids) == 1:
            axes = [axes]

        for i, cluster_id in enumerate(cluster_ids):
            subset = result[result[cluster_col] == cluster_id].head(top_n)
            label = "NOISE" if cluster_id == -1 else f"Cluster {cluster_id}"

            axes[i].bar(subset[col_name], subset["count"])
            axes[i].set_title(f"{label} - Top {top_n} 키워드")
            axes[i].set_ylabel("count")
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.show()
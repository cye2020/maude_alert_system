from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
import pandas as pd
import pendulum
import structlog

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.utils.sql_builder import build_cte_sql, build_join_clause

if TYPE_CHECKING:
    from snowflake.connector.cursor import SnowflakeCursor

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ClusterPipeline(SnowflakeBase):
    """MAUDE clustering 전용 파이프라인."""

    def __init__(self, logical_date: pendulum.DateTime):
        self.cfg = get_config().silver
        self.logical_date = logical_date

        if not self.cfg.get_snowflake_enabled():
            logger.warning('Snowflake 로드 비활성화 상태, 건너뜀')
            return

        database = self.cfg.get_snowflake_transform_database()
        schema = self.cfg.get_snowflake_transform_schema()
        super().__init__(database, schema)

        llm_category = self.cfg.get_llm_source_category()
        self.llm_join_table = f'{database}.{schema}.{llm_category}{self.cfg.get_llm_join_suffix()}'.upper()

        clustering_cat = self.cfg.get_clustering_source_category()
        self.clustering_target_table = (
            f'{database}.{schema}.{clustering_cat}{self.cfg.get_clustering_output_suffix()}'.upper()
        )

        logger.info('ClusterPipeline 초기화 완료', database=database, schema=schema, logical_date=str(logical_date))

    @with_context
    def fetch_clustering_data(self, cursor: SnowflakeCursor) -> pd.DataFrame:
        """EVENT_LLM_EXTRACTED에서 logical_date 기준 최신분만 SELECT."""
        categorical_cols = self.cfg.get_clustering_categorical_columns()
        text_col = self.cfg.get_clustering_text_column()
        hover_cols = self.cfg.get_clustering_hover_cols()
        pk_cols = self.cfg.get_silver_primary_key(self.cfg.get_clustering_source_category())
        if isinstance(pk_cols, str):
            pk_cols = [pk_cols]
        select_cols = list(dict.fromkeys(pk_cols + hover_cols + categorical_cols + [text_col]))
        batch_id = f"maude_{self.logical_date.strftime('%Y%m')}"
        where = f"SOURCE_BATCH_ID = '{batch_id}'"

        logger.info(
            'clustering 데이터 로드 시작',
            source=self.llm_join_table,
            columns=select_cols,
            batch_id=batch_id,
        )
        sql = build_cte_sql(
            ctes=[],
            from_clause=self.llm_join_table,
            select_cols=select_cols,
            where=where,
        )
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        logger.info('clustering 데이터 로드 완료', rows=len(df), batch_id=batch_id)
        return df

    def prepare_clustering_embeddings(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """vocab filter -> embed (추론 전용 임베딩 준비)."""
        from maude_early_alert.preprocessors.clustering import (
            analyze_keywords,
            embed_texts,
            prepare_text_col,
        )

        text_col = self.cfg.get_clustering_text_column()
        sntc_col = f'{text_col}_FILTERED'

        vocab = analyze_keywords(df, text_col=text_col, min_freq=self.cfg.get_clustering_vocab_min_freq())
        df = prepare_text_col(df, text_col=text_col, output_col=sntc_col, vocab=vocab)
        embeddings = embed_texts(
            df[sntc_col].tolist(),
            model=self.cfg.get_clustering_embedding_model(),
            batch_size=self.cfg.get_clustering_embedding_batch_size(),
            normalize=self.cfg.get_clustering_embedding_normalize(),
        )

        logger.info('clustering 임베딩 준비 완료', rows=len(df))
        return embeddings, df

    def run_clustering_prediction(
        self, df: pd.DataFrame, embeddings: np.ndarray, run_dir: str | None = None
    ) -> Tuple[np.ndarray, Dict]:
        """저장된 베스트 모델로 전체 클러스터링."""
        from maude_early_alert.preprocessors.clustering import load_and_predict

        if run_dir is None:
            run_dir = self.cfg.get_clustering_inference_model_dir()
        logger.info('clustering 추론 모델 로드', model_dir=run_dir)
        labels, metadata = load_and_predict(
            embeddings=embeddings,
            model_dir=run_dir,
            df=df,
        )
        return labels, metadata

    @with_context
    def join_clustering_results(self, cursor: SnowflakeCursor, df: pd.DataFrame, labels: np.ndarray) -> None:
        """clustering labels를 primary key 기준 JOIN으로 원본 테이블에 붙여 _CLUSTERED 저장."""
        pk_cols = self.cfg.get_silver_primary_key(self.cfg.get_clustering_source_category())
        if isinstance(pk_cols, str):
            pk_cols = [pk_cols]

        temp_table = f'{self.clustering_target_table}_STG'
        col_defs = ', '.join(f'{c} VARCHAR' for c in pk_cols) + ', CLUSTER INT'
        cursor.execute(f'CREATE OR REPLACE TEMPORARY TABLE {temp_table} ({col_defs})')

        rows = [
            tuple(str(row[c]) for c in pk_cols) + (int(label),)
            for row, label in zip(df[pk_cols].to_dict('records'), labels.tolist())
        ]
        placeholders = ', '.join(['%s'] * (len(pk_cols) + 1))
        col_names = ', '.join(pk_cols) + ', CLUSTER'
        sql = f'INSERT INTO {temp_table} ({col_names}) VALUES ({placeholders})'
        chunk_size = 10_000
        for i in range(0, len(rows), chunk_size):
            cursor.executemany(sql, rows[i:i + chunk_size])

        join_clause = build_join_clause(
            left_table=self.llm_join_table,
            right_table=temp_table,
            on_columns=pk_cols,
            join_type='LEFT',
            left_alias='e',
            right_alias='c',
        )
        select_sql = build_cte_sql(
            ctes=[],
            from_clause=f'{self.llm_join_table} e\n{join_clause}',
            select_cols=['e.*', 'c.CLUSTER'],
        )
        cursor.execute(f'CREATE OR REPLACE TABLE {self.clustering_target_table} AS\n{select_sql}')
        logger.info('_CLUSTERED 테이블 생성 완료', table=self.clustering_target_table)

# ============================================================================
# llm_pipeline.py
# LLM 추출 파이프라인
#
# LLMPipeline: Airflow env에서 실행되는 Snowflake I/O 클래스
# run_llm_extraction / run_failure_model_retry: vllm-env에서 실행되는 배치 처리 함수
#
# NOTE: top-level에 snowflake 관련 import 금지.
#   @task.external_python이 vllm-env에서 이 모듈을 import할 때 모듈 전체가
#   로드되므로, vllm-env에 없는 패키지(snowflake 등)를 top-level에 두면
#   ImportError가 발생합니다. snowflake 관련 import는 __init__/메서드 내부에서
#   lazy하게 수행합니다.
# ============================================================================
from __future__ import annotations

import shutil
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd
import pendulum
import structlog

if TYPE_CHECKING:
    from maude_early_alert.preprocessors.mdr_extractor import MAUDEExtractor
    from snowflake.connector.cursor import SnowflakeCursor

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class LLMPipeline:
    """LLM 추출 파이프라인 (Snowflake I/O 담당).

    Airflow env에서만 인스턴스화됩니다.
    vllm-env에서는 run_llm_extraction, run_failure_model_retry를 사용합니다.
    """

    def __init__(self, logical_date: pendulum.DateTime):
        from maude_early_alert.pipelines.config import get_config

        self.cfg = get_config().silver
        self.logical_date = logical_date

        if not self.cfg.get_snowflake_enabled():
            logger.warning('Snowflake 로드 비활성화 상태, 건너뜀')
            return

        database = self.cfg.get_snowflake_transform_database()
        schema = self.cfg.get_snowflake_transform_schema()

        category = self.cfg.get_llm_source_category()
        self.llm_source_table = f'{database}.{schema}.{category}{self.cfg.get_llm_source_suffix()}'.upper()
        self.llm_extracted_table = f'{database}.{schema}.{category}{self.cfg.get_llm_extracted_suffix()}'.upper()
        self.llm_join_table = f'{database}.{schema}.{category}{self.cfg.get_llm_join_suffix()}'.upper()

        logger.info('LLMPipeline 초기화 완료', database=database, schema=schema, logical_date=str(logical_date))

    def extract_mdr_text(self, cursor: SnowflakeCursor) -> List[Dict]:
        """source 테이블에서 MDR_TEXT 추출 + unique 처리, dict 리스트 반환"""
        from maude_early_alert.preprocessors.text_extract import (
            build_ensure_extracted_table_sql,
            build_mdr_text_extract_sql,
        )

        logger.info('MDR_TEXT 추출 시작', source=self.llm_source_table)
        source_cols = self.cfg.get_llm_source_columns()
        pk_col = self.cfg.get_llm_extracted_pk_column()
        cursor.execute(build_ensure_extracted_table_sql(self.llm_extracted_table, self.cfg.get_llm_extracted_columns()))
        sql = build_mdr_text_extract_sql(
            table_name=self.llm_source_table,
            columns=source_cols,
            logical_date=self.logical_date,
            exclude_extracted_table=self.llm_extracted_table,
            pk_column=pk_col,
        )
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        total = len(df)
        df = df.drop_duplicates(subset=[pk_col])[source_cols].fillna('')
        unique_records = df.rename(columns={c: c.lower() for c in source_cols}).to_dict('records')
        logger.info('MDR_TEXT 추출 완료', total=total, unique=len(unique_records))
        return unique_records

    def fetch_failure_candidates(self, cursor: SnowflakeCursor) -> List[dict]:
        """Snowflake _EXTRACTED에서 failure 모델 재시도 대상 레코드 조회.

        재시도 조건 (OR):
            - _EXTRACTED에 없음 (1차 추출 실패)
            - PATIENT_HARM = 'Unknown'
            - DEFECT_TYPE = 'Unknown'

        Returns:
            재시도 대상 레코드 리스트
        """
        from maude_early_alert.preprocessors.text_extract import build_failure_candidates_sql

        sql = build_failure_candidates_sql(
            source_table=self.llm_source_table,
            extracted_table=self.llm_extracted_table,
            source_columns=self.cfg.get_llm_source_columns(),
            pk_column=self.cfg.get_llm_extracted_pk_column(),
            unknown_columns=self.cfg.get_llm_extracted_unknown_columns(),
            logical_date=self.logical_date,
        )
        cursor.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            logger.info('failure 모델 재시도 대상 없음')
            return []

        col_names = [desc[0].lower() for desc in cursor.description]
        records = [dict(zip(col_names, row)) for row in rows]
        pk_col = self.cfg.get_llm_extracted_pk_column().lower()
        records = pd.DataFrame(records).drop_duplicates(subset=[pk_col]).to_dict('records')
        logger.info('failure 모델 재시도 대상 조회 완료', retry_count=len(records))
        return records

    def load_extraction_results(self, cursor: SnowflakeCursor, results: List[dict]):
        """추출 결과를 Snowflake에 적재 (temp table → MERGE)"""
        from maude_early_alert.loaders.snowflake_load import SnowflakeLoader
        from maude_early_alert.preprocessors.text_extract import (
            build_create_extract_temp_sql,
            build_ensure_extracted_table_sql,
            build_extract_stage_insert_sql,
            prepare_insert_data,
        )

        columns = self.cfg.get_llm_extracted_columns()
        pk_col = self.cfg.get_llm_extracted_pk_column()
        non_pk_cols = self.cfg.get_llm_extracted_non_pk_columns()

        insert_data = prepare_insert_data(results, columns)
        if not insert_data:
            logger.warning('적재할 추출 데이터가 없습니다')
            return

        logger.info('추출 결과 적재 시작', extracted_table=self.llm_extracted_table, count=len(insert_data))
        cursor.execute(build_ensure_extracted_table_sql(self.llm_extracted_table, columns))

        temp_table = f"{self.llm_extracted_table}_STG_{self.logical_date.strftime('%Y%m%d')}"
        logger.debug('임시 스테이징 테이블 생성', temp_table=temp_table)
        cursor.execute(build_create_extract_temp_sql(temp_table, columns))
        stage_insert_sql = build_extract_stage_insert_sql(temp_table, columns)
        cursor.executemany(stage_insert_sql, insert_data)
        all_cols = [pk_col] + non_pk_cols
        cursor.execute(SnowflakeLoader.build_merge_sql(self.llm_extracted_table, temp_table, pk_col, all_cols))
        logger.info('추출 결과 적재 완료', count=len(insert_data))

    def cleanup_extraction_checkpoint(self) -> None:
        """LLM 추출 체크포인트 디렉토리 삭제."""
        checkpoint_dir = Path(self.cfg.get_llm_checkpoint_config()['dir'])
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info('체크포인트 삭제 완료', checkpoint_dir=str(checkpoint_dir))

    def join_extraction(self, cursor: SnowflakeCursor):
        """원본 EVENT + 추출 결과 LEFT JOIN → {category}_LLM_EXTRACTED 테이블 생성"""
        from maude_early_alert.preprocessors.text_extract import build_extracted_join_sql

        logger.info('추출 결과 JOIN 시작', source=self.llm_source_table, target=self.llm_join_table)
        sql = build_extracted_join_sql(
            base_table=self.llm_source_table,
            extracted_table=self.llm_extracted_table,
            non_pk_columns=self.cfg.get_llm_extracted_non_pk_columns(),
            pk_column=self.cfg.get_llm_extracted_pk_column(),
        )
        cursor.execute(f'CREATE OR REPLACE TABLE {self.llm_join_table} AS\n{sql}')
        logger.info('JOIN 결과 테이블 생성 완료', table=self.llm_join_table)

    @cached_property
    def _llm_extractor(self) -> 'MAUDEExtractor':
        """1차 추출용 extractor (최초 접근 시 모델 로드, 이후 재사용)."""
        from maude_early_alert.preprocessors.mdr_extractor import MAUDEExtractor
        from maude_early_alert.preprocessors.prompt import get_prompt

        return MAUDEExtractor(
            **self.cfg.get_llm_model_config(),
            sampling_config=self.cfg.get_llm_sampling_config(),
            prompt=get_prompt(self.cfg.get_llm_prompt_mode()),
        )

    @cached_property
    def _failure_extractor(self) -> 'MAUDEExtractor':
        """failure 모델용 extractor (최초 접근 시 모델 로드, 이후 재사용)."""
        from maude_early_alert.preprocessors.mdr_extractor import MAUDEExtractor
        from maude_early_alert.preprocessors.prompt import get_prompt

        return MAUDEExtractor(
            **self.cfg.get_llm_failure_model_config(),
            sampling_config=self.cfg.get_llm_sampling_config(),
            prompt=get_prompt(self.cfg.get_llm_prompt_mode()),
        )

    def run_llm_extraction(self, records: List[dict], chunk_idx: str = '0') -> List[dict]:
        """vLLM 배치 처리 (1차 추출용).

        vllm-env에서 LLMPipeline 인스턴스를 생성해 호출합니다.
        __init__은 snowflake를 import하지 않으므로 vllm-env에서 안전하게 실행됩니다.

        Args:
            records: {'mdr_text': ..., 'product_problems': ...} 형태의 dict 리스트
            chunk_idx: 청크 인덱스 (체크포인트 파일명 구분용)

        Returns:
            추출 결과 dict 리스트 (입력 순서 유지)
        """
        if not records:
            return []

        category = self.cfg.get_llm_source_category()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        return self._llm_extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_chunk{chunk_idx}",
        )

    def run_failure_model_retry(self, records: List[dict], chunk_idx: str = '0') -> List[dict]:
        """vLLM 배치 처리 (failure 재시도용).

        Args:
            records: fetch_failure_candidates로 조회한 재시도 대상 레코드
            chunk_idx: 청크 인덱스 (체크포인트 파일명 구분용)

        Returns:
            추출 결과 dict 리스트 (load_extraction_results로 UPSERT)
        """
        if not records:
            return []

        category = self.cfg.get_llm_source_category()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        return self._failure_extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_failure_chunk{chunk_idx}",
        )

    def run_llm_batch(self, records: List[dict], chunk_idx: str = '0', extractor: Any | None = None) -> List[dict]:
        """호환용 래퍼: run_llm_extraction 사용."""
        if extractor is not None:
            category = self.cfg.get_llm_source_category()
            checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
            return extractor.process_batch(
                records,
                checkpoint_dir=checkpoint_cfg['dir'],
                checkpoint_interval=checkpoint_cfg['interval'],
                checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_chunk{chunk_idx}",
            )
        return self.run_llm_extraction(records, chunk_idx=chunk_idx)

    def run_failure_batch(self, records: List[dict], chunk_idx: str = '0', extractor: Any | None = None) -> List[dict]:
        """호환용 래퍼: run_failure_model_retry 사용."""
        if extractor is not None:
            category = self.cfg.get_llm_source_category()
            checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
            return extractor.process_batch(
                records,
                checkpoint_dir=checkpoint_cfg['dir'],
                checkpoint_interval=checkpoint_cfg['interval'],
                checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_failure_chunk{chunk_idx}",
            )
        return self.run_failure_model_retry(records, chunk_idx=chunk_idx)

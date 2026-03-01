import json
import shutil
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pendulum
import structlog

from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.loaders.snowflake_load import SnowflakeLoader, get_staging_table_name
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.preprocessors.column_select import build_select_columns_sql
from maude_early_alert.utils.sql_builder import build_cte_sql, build_join_clause
from maude_early_alert.preprocessors.custom_transform import (
    build_combine_mdr_text_sql,
    build_primary_udi_di_sql,
    build_extract_udi_di_sql,
    build_apply_company_alias_sql,
    build_manufacturer_fuzzy_match_sql,
)
from maude_early_alert.preprocessors.flatten import (
    build_array_keys_sql,
    build_flatten_sql,
    build_top_keys_sql,
    parse_array_keys_result
)
from maude_early_alert.preprocessors.imputation import build_mode_fill_sql
from maude_early_alert.preprocessors.type_cast import build_type_cast_sql
from maude_early_alert.preprocessors.value_clean import build_clean_sql
from maude_early_alert.preprocessors.row_filter import build_filter_sql, build_filter_pipeline
from maude_early_alert.preprocessors.udi_match import build_matching_sql
from maude_early_alert.preprocessors.text_extract import (
    build_mdr_text_extract_sql,
    build_ensure_extracted_table_sql,
    build_create_extract_temp_sql,
    build_extract_stage_insert_sql,
    build_extract_merge_sql,
    build_failure_candidates_sql,
    build_extracted_join_sql,
    prepare_insert_data,
    MDRExtractor,
)
from maude_early_alert.preprocessors.metadata_add import (
    add_incremental_metadata,
    build_expire_old_records_sql,
    build_extract_bronze_metadata_sql,
)
from maude_early_alert.preprocessors.prompt import get_prompt

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

class SilverPipeline(SnowflakeBase):
    def __init__(self, stage: Dict[str, int], logical_date: pendulum.DateTime):
        self.cfg = get_config().silver
        self.stage = stage
        self.logical_date = logical_date

        if not self.cfg.get_snowflake_enabled():
            logger.warning('Snowflake 로드 비활성화 상태, 건너뜀')
            return

        database = self.cfg.get_snowflake_transform_database()
        schema = self.cfg.get_snowflake_transform_schema()
        super().__init__(database, schema)

        category = self.cfg.get_llm_source_category()
        self.llm_source_table = f'{database}.{schema}.{category}{self.cfg.get_llm_source_suffix()}'.upper()
        self.llm_extracted_table = f'{database}.{schema}.{category}{self.cfg.get_llm_extracted_suffix()}'.upper()
        self.llm_join_table = f'{database}.{schema}.{category}{self.cfg.get_llm_join_suffix()}'.upper()

        clustering_cat = self.cfg.get_clustering_source_category()
        self.clustering_target_table = (
            f'{database}.{schema}.{clustering_cat}{self.cfg.get_clustering_output_suffix()}'.upper()
        )

        logger.info('SilverPipeline 초기화 완료', database=database, schema=schema, logical_date=str(logical_date))

    def _stage_table(self, category: str) -> str:
        """현재 stage 테이블명 반환 (e.g. 'EVENT_STAGE_01')"""
        if self.stage[category] == 0:
            database = self.cfg.get_snowflake_load_database()
            schema = self.cfg.get_snowflake_load_schema()
            return f'{database}.{schema}.{category}'.upper()
        return f'{category}_stage_{self.stage[category]:02d}'.upper()

    def _create_next_stage(self, cursor: SnowflakeCursor, category: str, sql: str):
        """다음 stage 테이블 생성 후 self.stage 업데이트"""
        self.stage[category] += 1
        next_table = self._stage_table(category)
        logger.debug('stage 테이블 생성 중', table=next_table)
        full_sql = f'CREATE OR REPLACE TABLE {next_table} AS\n{sql}'
        cursor.execute(full_sql)
        logger.info('stage 테이블 생성 완료', category=category, table=next_table)

    def get_categories(self, step: str) -> List[str]:
        """DAG task_group expand용 - 스텝별 카테고리 목록 반환"""
        return {
            'filter_dedup_rows':     lambda: list(self.cfg.get_filtering_step('DEDUP').keys()),
            'filter_scoping':        lambda: list(self.cfg.get_filtering_step('SCOPING').keys()),
            'filter_quality':        lambda: list(self.cfg.get_filtering_step('QUALITY_FILTER').keys()),
            'flatten':               self.cfg.get_flatten_categories,
            'select_columns':        self.cfg.get_column_categories,
            'select_columns_final':  self.cfg.get_final_categories,
            'impute_missing_values': self.cfg.get_imputation_categories,
            'clean_values':          self.cfg.get_cleaning_categories,
            'cast_types':            self.cfg.get_column_categories,
            'add_scd2_metadata':     lambda: list(self.stage.keys()),
        }[step]()

    def _filter_step(self, key: str, cursor: SnowflakeCursor, category: str):
        """filtering.yaml의 key에 해당하는 스텝을 category별로 실행"""
        step = self.cfg.get_filtering_step(key)[category]
        source = self._stage_table(category)
        logger.info('필터 스텝 실행', key=key, category=category, type=step['type'], source=source)
        if step['type'] == 'standalone':
            sql = build_filter_sql(source, **{k: v for k, v in step.items() if k in ('where', 'qualify')})
        else:  # chain
            sql = build_filter_pipeline(source, step['ctes'], step['final'])
        self._create_next_stage(cursor, category, sql)

    @with_context
    def filter_dedup_rows(self, cursor: SnowflakeCursor, category: str):
        self._filter_step('DEDUP', cursor, category)

    @with_context
    def filter_scoping(self, cursor: SnowflakeCursor, category: str):
        self._filter_step('SCOPING', cursor, category)

    @with_context
    def filter_quality(self, cursor: SnowflakeCursor, category: str):
        self._filter_step('QUALITY_FILTER', cursor, category)

    def _fetch_schema(self, cursor: SnowflakeCursor, table_name: str):
        """top-level 키/타입 + 배열별 nested 스키마 조회"""
        logger.debug('스키마 조회 중', table=table_name)
        cursor.execute(build_top_keys_sql(table_name))
        top = {str(k): str(t) for k, t in cursor.fetchall() if k is not None}

        array_schemas = {}
        for name, typ in top.items():
            if typ.upper() == 'ARRAY':
                cursor.execute(build_array_keys_sql(table_name, name))
                array_schemas[name] = parse_array_keys_result(cursor.fetchall())

        logger.debug('스키마 조회 완료', table=table_name, top_keys=len(top), array_keys=len(array_schemas))
        return top, array_schemas

    @with_context
    def flatten(self, cursor: SnowflakeCursor, category: str):
        table_name = self._stage_table(category)
        logger.info('flatten 시작', category=category, source=table_name)
        flatten_cfg = self.cfg.get_flatten_config(category)
        top, array_schemas = self._fetch_schema(cursor, table_name)

        exclude = {name for names in flatten_cfg.values() for name in names}
        scalar_keys = {k: t.upper() for k, t in top.items() if k not in exclude}

        strategy_keys = {
            f'{strategy}_keys': {
                k: array_schemas[k]
                for k in names if k in array_schemas
            }
            for strategy, names in flatten_cfg.items()
        }

        flatten_sql = build_flatten_sql(
            table_name=table_name,
            scalar_keys=scalar_keys,
            **strategy_keys,
        )
        logger.debug('flatten SQL 생성 완료', category=category, sql=flatten_sql)
        self._create_next_stage(cursor, category, flatten_sql)

    @with_context
    def combine_mdr_text(self, cursor: SnowflakeCursor):
        category = self.cfg.get_combine_category()
        logger.info('MDR 텍스트 결합 시작', category=category, source=self._stage_table(category))
        sql = build_combine_mdr_text_sql(self._stage_table(category), **self.cfg.get_combine_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def extract_primary_udi_di(self, cursor: SnowflakeCursor):
        category = self.cfg.get_primary_category()
        logger.info('primary UDI-DI 추출 시작', category=category, source=self._stage_table(category))
        sql = build_primary_udi_di_sql(self._stage_table(category), **self.cfg.get_primary_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def extract_udi_di(self, cursor: SnowflakeCursor):
        category = self.cfg.get_extract_udi_di_category()
        logger.info('UDI-DI 추출 시작', category=category, source=self._stage_table(category))
        sql = build_extract_udi_di_sql(self._stage_table(category), **self.cfg.get_extract_udi_di_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def apply_company_alias(self, cursor: SnowflakeCursor):
        logger.info('회사 별칭 적용 시작', source=self._stage_table('event'))
        sql = build_apply_company_alias_sql(
            source=self._stage_table('event'),
            company_col=self.cfg.get_ma_company_col(),
            aliases=self.cfg.get_ma_aliases(),
        )
        self._create_next_stage(cursor, 'event', sql)

    @with_context
    def fuzzy_match_manufacturer(self, cursor: SnowflakeCursor):
        udf_schema = f"{self.cfg.get_snowflake_udf_database()}.{self.cfg.get_snowflake_udf_schema()}"
        target_category = self.cfg.get_fuzzy_match_target_category()
        threshold = self.cfg.get_fuzzy_match_threshold()
        logger.info('제조사 퍼지 매칭 시작', target=target_category, threshold=threshold)
        sql = build_manufacturer_fuzzy_match_sql(
            target=self._stage_table(target_category),
            source=self._stage_table(self.cfg.get_fuzzy_match_source_category()),
            mfr_col=self.cfg.get_fuzzy_match_mfr_col(),
            udf_schema=udf_schema,
            threshold=threshold,
        )
        self._create_next_stage(cursor, target_category, sql)

    @with_context
    def select_columns(self, cursor: SnowflakeCursor, category: str, final: bool = False):
        get_cols = self.cfg.get_final_cols if final else self.cfg.get_column_cols
        logger.info('컬럼 선택 시작', final=final, category=category)
        sql = build_select_columns_sql(get_cols(category), self._stage_table(category))
        self._create_next_stage(cursor, category, sql)

    @with_context
    def impute_missing_values(self, cursor: SnowflakeCursor, category: str):
        logger.info('결측값 대체 시작', category=category, source=self._stage_table(category))
        sql = build_mode_fill_sql(
            group_to_target=self.cfg.get_imputation_mode(category),
            table_name=self._stage_table(category),
            table_alias=self.cfg.get_imputation_alias(category),
        )
        self._create_next_stage(cursor, category, sql)

    @with_context
    def clean_values(self, cursor: SnowflakeCursor, category: str):
        udf_schema = f"{self.cfg.get_snowflake_udf_database()}.{self.cfg.get_snowflake_udf_schema()}"
        logger.info('값 정제 시작', category=category, source=self._stage_table(category))
        sql = build_clean_sql(
            table_name=self._stage_table(category),
            config=self.cfg.get_cleaning_config(category),
            udf_schema=udf_schema,
        )
        self._create_next_stage(cursor, category, sql)

    @with_context
    def cast_types(self, cursor: SnowflakeCursor, category: str):
        logger.info('타입 캐스팅 시작', category=category, source=self._stage_table(category))
        sql = build_type_cast_sql(
            columns=self.cfg.get_column_cols(category),
            input_table=self._stage_table(category),
        )
        self._create_next_stage(cursor, category, sql)

    @with_context
    def match_udi(self, cursor: SnowflakeCursor):
        target_cat = self.cfg.get_matching_target_category()
        source_cat = self.cfg.get_matching_source_category()
        logger.info('UDI 매칭 시작', target=target_cat, source=source_cat)
        sql = build_matching_sql(
            target=self._stage_table(target_cat),
            source=self._stage_table(source_cat),
            **self.cfg.get_matching_kwargs(),
        )
        self._create_next_stage(cursor, target_cat, sql)

    # ==================== LLM extraction (별도 단계) ====================

    @with_context
    def extract_mdr_text(self, cursor: SnowflakeCursor) -> List[Dict]:
        """source 테이블에서 MDR_TEXT 추출 + unique 처리, dict 리스트 반환"""
        logger.info('MDR_TEXT 추출 시작', source=self.llm_source_table)
        sql = build_mdr_text_extract_sql(
            table_name=self.llm_source_table,
            columns=self.cfg.get_llm_source_columns(),
        )
        source_cols = self.cfg.get_llm_source_columns()
        pk_col = self.cfg.get_llm_extracted_pk_column()

        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        total = len(df)
        df = df.drop_duplicates(subset=[pk_col])[source_cols].fillna('')
        unique_records = df.rename(columns={c: c.lower() for c in source_cols}).to_dict('records')
        logger.info('MDR_TEXT 추출 완료', total=total, unique=len(unique_records))
        return unique_records

    @cached_property
    def _llm_extractor(self) -> MDRExtractor:
        """1차 LLM 추출용 extractor (최초 접근 시 모델 로드, 이후 재사용)"""
        return MDRExtractor(
            **self.cfg.get_llm_model_config(),
            sampling_config=self.cfg.get_llm_sampling_config(),
            prompt=get_prompt(self.cfg.get_llm_prompt_mode()),
        )

    @cached_property
    def _failure_extractor(self) -> MDRExtractor:
        """failure 모델용 extractor (최초 접근 시 모델 로드, 이후 재사용)"""
        return MDRExtractor(
            **self.cfg.get_llm_failure_model_config(),
            sampling_config=self.cfg.get_llm_sampling_config(),
            prompt=get_prompt(self.cfg.get_llm_prompt_mode()),
        )

    def run_llm_extraction(self, records: List) -> List[dict]:
        """vLLM 배치 처리 (cursor 불필요, GPU 작업)"""
        category = self.cfg.get_llm_source_category()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        return self._llm_extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}",
        )

    @with_context
    def fetch_failure_candidates(self, cursor: SnowflakeCursor) -> List[dict]:
        """Snowflake _EXTRACTED에서 failure 모델 재시도 대상 레코드 조회.

        재시도 조건 (OR):
            - _EXTRACTED에 없음 (1차 추출 실패)
            - PATIENT_HARM = 'Unknown'
            - DEFECT_TYPE = 'Unknown'

        Returns:
            재시도 대상 레코드 리스트
        """
        sql = build_failure_candidates_sql(
            source_table=self.llm_source_table,
            extracted_table=self.llm_extracted_table,
            source_columns=self.cfg.get_llm_source_columns(),
            pk_column=self.cfg.get_llm_extracted_pk_column(),
            unknown_columns=self.cfg.get_llm_extracted_unknown_columns(),
        )
        cursor.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            logger.info('failure 모델 재시도 대상 없음')
            return []

        col_names = [desc[0].lower() for desc in cursor.description]
        records = [dict(zip(col_names, row)) for row in rows]
        logger.info('failure 모델 재시도 대상 조회 완료', retry_count=len(records))
        return records

    def run_failure_model_retry(self, records: List[dict]) -> List[dict]:
        """failure 모델로 재시도 (Snowflake 연결 불필요).

        Args:
            records: fetch_failure_candidates로 조회한 재시도 대상 레코드

        Returns:
            failure 모델 추출 결과 리스트 (load_extraction_results로 UPSERT)
        """
        if not records:
            logger.info('failure 모델 재시도 대상 없음 (스킵)')
            return []

        category = self.cfg.get_llm_source_category()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        return self._failure_extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_failure",
        )

    @with_context
    def load_extraction_results(self, cursor: SnowflakeCursor, results: List[dict]):
        """추출 결과를 Snowflake에 적재 (temp table -> MERGE)"""
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
        cursor.execute(build_extract_merge_sql(self.llm_extracted_table, temp_table, pk_col, non_pk_cols))
        logger.info('추출 결과 적재 완료', count=len(insert_data))

    def cleanup_extraction_checkpoint(self) -> None:
        """LLM 추출 체크포인트 디렉토리 삭제. load_extraction_results 성공 후 명시적으로 호출."""
        checkpoint_dir = Path(self.cfg.get_llm_checkpoint_config()['dir'])
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)
            logger.info('체크포인트 삭제 완료', checkpoint_dir=str(checkpoint_dir))

    @with_context
    def join_extraction(self, cursor: SnowflakeCursor):
        """원본 EVENT + 추출 결과 LEFT JOIN -> {category}_LLM_EXTRACTED 테이블 생성"""
        logger.info('추출 결과 JOIN 시작', source=self.llm_source_table, target=self.llm_join_table)
        sql = build_extracted_join_sql(
            base_table=self.llm_source_table,
            extracted_table=self.llm_extracted_table,
            non_pk_columns=self.cfg.get_llm_extracted_non_pk_columns(),
            pk_column=self.cfg.get_llm_extracted_pk_column(),
        )
        cursor.execute(f'CREATE OR REPLACE TABLE {self.llm_join_table} AS\n{sql}')
        logger.info('JOIN 결과 테이블 생성 완료', table=self.llm_join_table)

    @with_context
    def add_scd2_metadata(self, cursor: SnowflakeCursor, category: str):
        """증분 SCD2 메타데이터 추가 및 {CATEGORY}_CURRENT 테이블 적재

        신규 배치 레코드에 Silver SCD2 메타데이터를 추가하고
        MAUDE.SILVER.{CATEGORY}_CURRENT에 MERGE합니다.
        MERGE 이후 동일 business_key의 이전 배치 레코드를 만료 처리합니다.

        알고리즘:
            1. batch_id, source_system → 소스 테이블 컬럼 참조
            2. is_current = TRUE (신규 적재는 항상 현재 유효)
            3. effective_from = 카테고리별 컬럼
                   event: DATE_CHANGED
                   udi:   PUBLIC_VERSION_DATE
            4. ingest_time(logical_date), is_current, effective_from,
               source_batch_id(=batch_id), source_system 추가 후 MERGE
               primary_key 기준:
                   MATCHED     → SCD2 메타데이터 UPDATE (동일 배치 재처리 멱등성)
                   NOT MATCHED → 신규 행 INSERT
            5. 동일 business_key의 이전 배치 레코드 만료:
                   is_current  → FALSE
                   effective_to → 신규 effective_from - 1일
        """
        db = self.cfg.get_snowflake_transform_database()
        schema = self.cfg.get_snowflake_transform_schema()

        source = self._stage_table(category)
        pk_cols = self.cfg.get_silver_primary_key(category)
        business_key = self.cfg.get_silver_business_key(category)
        target = f'{db}.{schema}.{category}_CURRENT'.upper()
        stg_table = get_staging_table_name(target)

        logger.info(
            'SCD2 메타데이터 추가 시작 (증분)',
            category=category, source=source, target=target,
        )

        # 1-4. 메타데이터 추가 SQL 생성 (is_current=TRUE, effective_from=카테고리별 컬럼)
        bronze_db = self.cfg.get_snowflake_load_database()
        bronze_schema = self.cfg.get_snowflake_load_schema()
        bronze_table = f'{bronze_db}.{bronze_schema}.{category}'.upper()
        cursor.execute(build_extract_bronze_metadata_sql(bronze_table))
        row = cursor.fetchone()
        source_batch_id, source_system = row[0], row[1]

        effective_from_col = self.cfg.get_silver_effective_from_col(category)
        meta_sql = add_incremental_metadata(
            table_name=source,
            ingest_time=self.logical_date,
            effective_from_col=effective_from_col,
            source_batch_id=source_batch_id,
            source_system=source_system,
        )

        # staging 테이블 생성
        cursor.execute(f'CREATE OR REPLACE TEMPORARY TABLE {stg_table} AS\n{meta_sql}')

        # 컬럼 목록 조회 (MERGE INSERT 절 구성용)
        cursor.execute(f'SELECT * FROM {stg_table} LIMIT 0')
        all_cols = [desc[0] for desc in cursor.description]

        # 대상 테이블 없으면 구조만 복사해서 생성
        cursor.execute(
            f'CREATE TABLE IF NOT EXISTS {target} AS\n'
            f'SELECT * FROM {stg_table} WHERE FALSE'
        )

        # MERGE: 동일 primary_key → UPDATE (멱등성), 신규 → INSERT
        merge_sql = SnowflakeLoader.build_merge_sql(target, stg_table, pk_cols, all_cols)
        cursor.execute(merge_sql)

        # 5. 동일 business_key의 이전 레코드 만료 처리 (is_current=FALSE, effective_to 채움)
        expire_sql = build_expire_old_records_sql(target, business_key)
        cursor.execute(expire_sql)

        logger.info('SCD2 메타데이터 MERGE 및 만료 처리 완료', category=category, target=target)

    # ==================== Clustering ====================

    @with_context
    def fetch_clustering_data(self, cursor: SnowflakeCursor) -> pd.DataFrame:
        """Step 1: EVENT_LLM_EXTRACTED에서 clustering 대상 컬럼 SELECT"""
        categorical_cols = self.cfg.get_clustering_categorical_columns()
        text_col = self.cfg.get_clustering_text_column()
        hover_cols = self.cfg.get_clustering_hover_cols()
        pk_cols = self.cfg.get_silver_primary_key(self.cfg.get_clustering_source_category())
        if isinstance(pk_cols, str):
            pk_cols = [pk_cols]
        # 중복 없이 순서 유지 (pk를 앞에 배치)
        select_cols = list(dict.fromkeys(pk_cols + hover_cols + categorical_cols + [text_col]))

        logger.info('clustering 데이터 로드 시작', source=self.llm_join_table, columns=select_cols)
        sql = build_cte_sql(ctes=[], from_clause=self.llm_join_table, select_cols=select_cols)
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        logger.info('clustering 데이터 로드 완료', rows=len(df))
        return df

    def run_clustering_tuning(
        self, df: pd.DataFrame, run_dir: str = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """Step 2-5: vocab filter → embed → (train.enabled시) Optuna 튜닝 + 베스트 모델 저장.

        train.enabled=false이면 Optuna 튜닝을 스킵하고 임베딩만 반환.
        임베딩은 enabled 여부와 무관하게 항상 수행 (run_clustering_prediction에 필요).

        Args:
            run_dir: 모델 저장 경로. None이면 base_dir/runs/{timestamp}로 자동 생성.
                     train.enabled=false이면 무시됨.

        Returns:
            (embeddings, df_processed) — run_clustering_prediction에 재사용
        """
        from maude_early_alert.preprocessors.clustering import (
            analyze_keywords, prepare_text_col, embed_texts, train_and_save,
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

        selected_trial = self.cfg.get_clustering_selected_trial()

        if not self.cfg.get_clustering_train_enabled() and selected_trial is None:
            logger.info('clustering 학습 비활성화 (train.enabled=false), Optuna 튜닝 스킵')
            return embeddings, df

        if run_dir is None:
            resume_dir = self.cfg.get_clustering_resume_dir()
            if resume_dir:
                run_dir = resume_dir
                logger.info('clustering Optuna 재개 모드', run_dir=run_dir)
            else:
                timestamp = pendulum.now().strftime('%Y%m%d_%H%M%S')
                run_dir = f"{self.cfg.get_clustering_base_dir()}/runs/{timestamp}"
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        logger.info('clustering 학습 run 디렉토리', run_dir=run_dir)

        selected_trial = self.cfg.get_clustering_selected_trial()
        if selected_trial is not None:
            # selected_trial 모드: resume_run의 JSON에서 해당 trial 파라미터로 1회 재훈련
            from maude_early_alert.preprocessors.clustering import fit_with_params
            resume_dir = self.cfg.get_clustering_resume_dir()
            if not resume_dir:
                raise ValueError("selected_trial 설정 시 resume_run도 지정해야 합니다.")
            log_file = Path(resume_dir) / self.cfg._clustering['train']['optuna']['log_file']
            logs = json.loads(log_file.read_text())
            trial_entry = next((t for t in logs if t['trial'] == selected_trial), None)
            if trial_entry is None:
                raise ValueError(f"trial {selected_trial}을 {log_file}에서 찾을 수 없습니다.")
            logger.info('선택된 trial로 재훈련', trial=selected_trial,
                        hyperparams=trial_entry['hyperparams'])
            fit_with_params(
                hyperparams=trial_entry['hyperparams'],
                embeddings=embeddings,
                save_dir=run_dir,
                umap_fixed_params=self.cfg.get_clustering_umap_params(),
                hdbscan_params=self.cfg.get_clustering_hdbscan_params(),
                categorical_cols=self.cfg.get_clustering_categorical_columns(),
                df=df,
            )
        else:
            # Optuna 모드: 자동 최적화 후 베스트 모델 저장
            train_and_save(
                embeddings=embeddings,
                save_dir=run_dir,
                df=df,
                categorical_cols=self.cfg.get_clustering_categorical_columns(),
                umap_params=self.cfg.get_clustering_umap_params(),
                hdbscan_params=self.cfg.get_clustering_hdbscan_params(),
                validity_params=self.cfg.get_clustering_validity_params(),
                scoring_params=self.cfg.get_clustering_scoring_params(),
                optuna_params=self.cfg.get_clustering_optuna_params(run_dir),
            )
        return embeddings, df

    def run_clustering_prediction(
        self, df: pd.DataFrame, embeddings: np.ndarray, run_dir: str = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Step 6: 저장된 베스트 모델로 전체 클러스터링.

        Args:
            run_dir: 모델 경로. None이면 yaml active_run 사용 (null이면 ValueError).
        """
        from maude_early_alert.preprocessors.clustering import load_and_predict
        if run_dir is None:
            run_dir = self.cfg.get_clustering_active_dir()
        labels, metadata = load_and_predict(
            embeddings=embeddings,
            model_dir=run_dir,
            df=df,
        )
        return labels, metadata

    @with_context
    def join_clustering_results(
        self, cursor: SnowflakeCursor, df: pd.DataFrame, labels: np.ndarray,
    ):
        """Step 5: clustering labels를 primary key 기준 JOIN으로 원본 테이블에 붙여 _CLUSTERED 저장"""
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

    def cleanup_stages(self, cursor: SnowflakeCursor):
        """중간 stage 테이블 전부 삭제"""
        logger.info('중간 stage 테이블 정리 시작', stage=dict(self.stage))
        for category, current in self.stage.items():
            for i in range(1, current + 1):
                table = f'{category}_stage_{i:02d}'.upper()
                logger.debug('stage 테이블 삭제', table=table)
                cursor.execute(f'DROP TABLE IF EXISTS {table}')
        logger.info('중간 stage 테이블 정리 완료')


if __name__ == '__main__':
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret
    from maude_early_alert.logging_config import configure_logging

    configure_logging(level='DEBUG', log_file='silver.log')

    secret = get_secret('snowflake/de')

    def _connect():
        return snowflake.connector.connect(
            user=secret['user'],
            password=secret['password'],
            account=secret['account'],
            warehouse=secret['warehouse'],
        )

    pipeline = SilverPipeline(stage={'event': 14, 'udi': 7}, logical_date=pendulum.now())

    logger.info('Silver 파이프라인 시작')

    # ── Silver 14단계 ──────────────────────────────────────────────
    # conn = _connect()
    # cursor = conn.cursor()
    # try:
    #     for cat in pipeline.get_categories('filter_dedup_rows'):
    #         pipeline.filter_dedup_rows(cursor, category=cat)
    #     for cat in pipeline.get_categories('flatten'):
    #         pipeline.flatten(cursor, category=cat)
    #     for cat in pipeline.get_categories('filter_scoping'):
    #         pipeline.filter_scoping(cursor, category=cat)
    #     pipeline.combine_mdr_text(cursor)
    #     pipeline.extract_primary_udi_di(cursor)
    #     for cat in pipeline.get_categories('select_columns'):
    #         pipeline.select_columns(cursor, category=cat)
    #     for cat in pipeline.get_categories('impute_missing_values'):
    #         pipeline.impute_missing_values(cursor, category=cat)
    #     for cat in pipeline.get_categories('clean_values'):
    #         pipeline.clean_values(cursor, category=cat)
    #     pipeline.apply_company_alias(cursor)
    #     pipeline.fuzzy_match_manufacturer(cursor)
    #     for cat in pipeline.get_categories('cast_types'):
    #         pipeline.cast_types(cursor, category=cat)
    #     pipeline.extract_udi_di(cursor)
    #     pipeline.match_udi(cursor)
    #     for cat in pipeline.get_categories('filter_quality'):
    #         pipeline.filter_quality(cursor, category=cat)
    #     for cat in pipeline.get_categories('select_columns_final'):
    #         pipeline.select_columns(cursor, category=cat, final=True)
    #     for cat in pipeline.get_categories('add_scd2_metadata'):
    #         pipeline.add_scd2_metadata(cursor, category=cat)
    # except Exception:
    #     logger.error('전처리 실패', exc_info=True)
    #     raise
    # finally:
    #     cursor.close()
    #     conn.close()

    logger.info('Silver 14단계 완료')

    # ── LLM 추출 (Design 2) ───────────────────────────────────────
    # 1. source → records 추출
    # 2. 1차 LLM 추출 → checkpoint.db (중단 시 재개 가능)
    # 3. _EXTRACTED 청크 적재 (5000건 단위, 연결별 독립)
    # 4. failure 대상 조회
    # 5. failure 모델 재시도 → checkpoint_failure.db (중단 시 재개 가능)
    # 6. _EXTRACTED 증분 청크 적재
    # 7. source + _EXTRACTED LEFT JOIN → _LLM_EXTRACTED 생성

    CHUNK_SIZE = 5000

    # ── 1단계: source 읽기 ────────────────────────────────────────
    logger.info('LLM 추출 단계 시작: source 읽기')
    # conn = _connect()
    # cursor = conn.cursor()
    # try:
    #     records = pipeline.extract_mdr_text(cursor)
    # except Exception:
    #     logger.error('source 읽기 실패', exc_info=True)
    #     raise
    # finally:
    #     cursor.close()
    #     conn.close()

    # ── 2-3단계: 1차 LLM 추출 + _EXTRACTED 청크 적재 ────────────
    # total_chunks = (len(records) - 1) // CHUNK_SIZE + 1
    # logger.info('LLM 추출 단계 시작: 1차 추출 + 적재', total=len(records), chunks=total_chunks)
    # session_start = pendulum.now()
    # for chunk_idx in range(total_chunks):
    #     chunk = records[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
    #     logger.info('1차 청크 추출 시작', chunk=chunk_idx + 1, total_chunks=total_chunks, size=len(chunk))
    #     chunk_start = pendulum.now()
    #     results = pipeline.run_llm_extraction(chunk)
    #     elapsed = (pendulum.now() - chunk_start).in_seconds()
    #     elapsed_session = (pendulum.now() - session_start).in_seconds()
    #     avg_per_chunk = elapsed_session / (chunk_idx + 1)
    #     eta_seconds = avg_per_chunk * (total_chunks - chunk_idx - 1)
    #     success = sum(1 for r in results if r.get('_success', False))
    #     logger.info(
    #         '1차 청크 추출 완료',
    #         chunk=chunk_idx + 1, total_chunks=total_chunks,
    #         success=success, failed=len(results) - success,
    #         success_rate=f'{100 * success / len(results):.1f}%' if results else 'N/A',
    #         elapsed=f'{elapsed:.1f}s',
    #         elapsed_session=f'{elapsed_session / 3600:.2f}h',
    #         remaining_time=f'{eta_seconds / 3600:.1f}h',
    #         eta_kst=pendulum.now('Asia/Seoul').add(seconds=int(eta_seconds)).format('MM-DD HH:mm'),
    #     )
    #     conn = _connect()
    #     cursor = conn.cursor()
    #     try:
    #         pipeline.load_extraction_results(cursor, results)
    #     except Exception:
    #         logger.error('1차 청크 적재 실패', chunk=chunk_idx + 1, exc_info=True)
    #         raise
    #     finally:
    #         cursor.close()
    #         conn.close()

    # ── 4단계: failure 대상 조회 ──────────────────────────────────
    # logger.info('LLM 추출 단계 시작: failure 대상 조회')
    # conn = _connect()
    # cursor = conn.cursor()
    # try:
    #     failure_records = pipeline.fetch_failure_candidates(cursor)
    # except Exception:
    #     logger.error('failure 대상 조회 실패', exc_info=True)
    #     raise
    # finally:
    #     cursor.close()
    #     conn.close()

    # ── 5-6단계: failure 모델 재시도 + _EXTRACTED 증분 청크 적재 ─
    # total_chunks = (len(failure_records) - 1) // CHUNK_SIZE + 1 if failure_records else 0
    # logger.info('LLM 추출 단계 시작: failure 추출 + 적재', total=len(failure_records), chunks=total_chunks)
    # session_start = pendulum.now()
    # for chunk_idx in range(total_chunks):
    #     chunk = failure_records[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
    #     logger.info('failure 청크 추출 시작', chunk=chunk_idx + 1, total_chunks=total_chunks, size=len(chunk))
    #     chunk_start = pendulum.now()
    #     failure_results = pipeline.run_failure_model_retry(chunk)
    #     elapsed = (pendulum.now() - chunk_start).in_seconds()
    #     elapsed_session = (pendulum.now() - session_start).in_seconds()
    #     avg_per_chunk = elapsed_session / (chunk_idx + 1)
    #     eta_seconds = avg_per_chunk * (total_chunks - chunk_idx - 1)
    #     success = sum(1 for r in failure_results if r.get('_success', False))
    #     logger.info(
    #         'failure 청크 추출 완료',
    #         chunk=chunk_idx + 1, total_chunks=total_chunks,
    #         success=success, failed=len(failure_results) - success,
    #         success_rate=f'{100 * success / len(failure_results):.1f}%' if failure_results else 'N/A',
    #         elapsed=f'{elapsed:.1f}s',
    #         elapsed_session=f'{elapsed_session / 3600:.2f}h',
    #         remaining_time=f'{eta_seconds / 3600:.1f}h',
    #         eta_kst=pendulum.now('Asia/Seoul').add(seconds=int(eta_seconds)).format('MM-DD HH:mm'),
    #     )
    #     conn = _connect()
    #     cursor = conn.cursor()
    #     try:
    #         pipeline.load_extraction_results(cursor, failure_results)
    #     except Exception:
    #         logger.error('failure 청크 적재 실패', chunk=chunk_idx + 1, exc_info=True)
    #         raise
    #     finally:
    #         cursor.close()
    #         conn.close()

    # ── 7단계: _LLM_EXTRACTED JOIN ────────────────────────────────
    # logger.info('LLM 추출 단계 시작: _LLM_EXTRACTED JOIN')
    # conn = _connect()
    # cursor = conn.cursor()
    # try:
    #     pipeline.join_extraction(cursor)
    # except Exception:
    #     logger.error('JOIN 실패', exc_info=True)
    #     raise
    # finally:
    #     cursor.close()
    #     conn.close()

    # ── Clustering ────────────────────────────────────────────────
    # 1단계: 데이터 로드 (_LLM_EXTRACTED에서 clustering 대상 컬럼 SELECT)
    logger.info('Clustering 단계 시작: 데이터 로드')
    conn = _connect()
    cursor = conn.cursor()
    try:
        df = pipeline.fetch_clustering_data(cursor)
    except Exception:
        logger.error('clustering 데이터 로드 실패', exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

    # 2-5단계: vocab filter → 임베딩 → (train.enabled=true시) Optuna 튜닝 + 저장
    logger.info(
        'Clustering 단계 시작: 임베딩 + 학습',
        train_enabled=pipeline.cfg.get_clustering_train_enabled(),
    )
    embeddings, df = pipeline.run_clustering_tuning(df)

    # 6단계: active_run 모델로 전체 클러스터링 예측
    logger.info('Clustering 단계 시작: 예측')
    labels, metadata = pipeline.run_clustering_prediction(df, embeddings)
    n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
    logger.info(
        'Clustering 예측 완료',
        n_clusters=n_clusters,
        noise_ratio=f'{float((labels == -1).mean()):.2%}',
    )

    # 7단계: 결과 저장 (_CLUSTERED 테이블 생성)
    logger.info('Clustering 단계 시작: 결과 저장')
    conn = _connect()
    cursor = conn.cursor()
    try:
        pipeline.join_clustering_results(cursor, df, labels)
    except Exception:
        logger.error('clustering 결과 저장 실패', exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

    logger.info('Silver 파이프라인 완료')

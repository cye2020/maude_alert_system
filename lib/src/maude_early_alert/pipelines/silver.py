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
        logger.info(f'{category}stage 테이블 생성 완료', table=next_table)

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

    def _filter_step(self, key: str, cursor: SnowflakeCursor, category: str = None):
        """filtering.yaml의 key에 해당하는 스텝을 category별로 실행"""
        all_steps = self.cfg.get_filtering_step(key)
        run_steps = {category: all_steps[category]} if category else all_steps
        for category, step in run_steps.items():
            source = self._stage_table(category)
            logger.info('필터 스텝 실행', key=key, category=category, type=step['type'], source=source)
            if step['type'] == 'standalone':
                sql = build_filter_sql(source, **{k: v for k, v in step.items() if k in ('where', 'qualify')})
            else:  # chain
                sql = build_filter_pipeline(source, step['ctes'], step['final'])
            self._create_next_stage(cursor, category, sql)

    @with_context
    def filter_dedup_rows(self, cursor: SnowflakeCursor, category: str = None):
        logger.info('중복 행 필터링 시작')
        self._filter_step('DEDUP', cursor, category)

    @with_context
    def filter_scoping(self, cursor: SnowflakeCursor, category: str = None):
        logger.info('스코핑 필터링 시작')
        self._filter_step('SCOPING', cursor, category)

    @with_context
    def filter_quality(self, cursor: SnowflakeCursor, category: str = None):
        logger.info('품질 필터링 시작')
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
    def flatten(self, cursor: SnowflakeCursor, category: str = None):
        run_cats = [category] if category else self.cfg.get_flatten_categories()
        for category in run_cats:
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
    def select_columns(self, cursor: SnowflakeCursor, final: bool = False, category: str = None):
        all_cats = self.cfg.get_final_categories() if final else self.cfg.get_column_categories()
        get_cols = self.cfg.get_final_cols if final else self.cfg.get_column_cols
        run_cats = [category] if category else all_cats
        logger.info('컬럼 선택 시작', final=final, categories=run_cats)
        for category in run_cats:
            sql = build_select_columns_sql(get_cols(category), self._stage_table(category))
            self._create_next_stage(cursor, category, sql)

    @with_context
    def impute_missing_values(self, cursor: SnowflakeCursor, category: str = None):
        run_cats = [category] if category else self.cfg.get_imputation_categories()
        for category in run_cats:
            logger.info('결측값 대체 시작', category=category, source=self._stage_table(category))
            sql = build_mode_fill_sql(
                group_to_target=self.cfg.get_imputation_mode(category),
                table_name=self._stage_table(category),
                table_alias=self.cfg.get_imputation_alias(category),
            )
            self._create_next_stage(cursor, category, sql)

    @with_context
    def clean_values(self, cursor: SnowflakeCursor, category: str = None):
        udf_schema = f"{self.cfg.get_snowflake_udf_database()}.{self.cfg.get_snowflake_udf_schema()}"
        run_cats = [category] if category else self.cfg.get_cleaning_categories()
        for category in run_cats:
            logger.info('값 정제 시작', category=category, source=self._stage_table(category))
            sql = build_clean_sql(
                table_name=self._stage_table(category),
                config=self.cfg.get_cleaning_config(category),
                udf_schema=udf_schema,
            )
            self._create_next_stage(cursor, category, sql)

    @with_context
    def cast_types(self, cursor: SnowflakeCursor, category: str = None):
        run_cats = [category] if category else self.cfg.get_column_categories()
        for category in run_cats:
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

    def run_llm_extraction(self, records: List) -> List[dict]:
        """vLLM 배치 처리 (cursor 불필요, GPU 작업)"""
        logger.info('LLM 추출 시작', record_count=len(records))
        category = self.cfg.get_llm_source_category()
        model_cfg = self.cfg.get_llm_model_config()
        sampling_cfg = self.cfg.get_llm_sampling_config()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        prompt = get_prompt(self.cfg.get_llm_prompt_mode())

        logger.debug('LLM 모델 설정', model_cfg=model_cfg, checkpoint_dir=checkpoint_cfg['dir'])
        extractor = MDRExtractor(
            **model_cfg,
            sampling_config=sampling_cfg,
            prompt=prompt,
        )
        results = extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}",
        )
        logger.info('LLM 추출 완료', result_count=len(results))
        return results

    @with_context
    def run_failure_model_retry(self, cursor: SnowflakeCursor) -> List[dict]:
        """Snowflake _EXTRACTED에서 실패/UNKNOWN 레코드를 조회해 failure 모델로 재시도.

        재시도 조건 (OR):
            - _EXTRACTED에 없음 (1차 추출 실패)
            - PATIENT_HARM = 'Unknown'
            - DEFECT_TYPE = 'Unknown'

        Returns:
            failure 모델 추출 결과 리스트 (load_extraction_results로 UPSERT)
        """
        sql = build_failure_candidates_sql(
            source_table=self.llm_source_table,
            extracted_table=self.llm_extracted_table,
            source_columns=self.cfg.get_llm_source_columns(),
            pk_column=self.cfg.get_llm_extracted_pk_column(),
            unknown_columns=['PATIENT_HARM', 'DEFECT_TYPE'],
            source_where=self.cfg.get_llm_source_where(),
        )
        cursor.execute(sql)
        rows = cursor.fetchall()

        if not rows:
            logger.info('failure 모델 재시도 대상 없음 (스킵)')
            return []

        col_names = [desc[0].lower() for desc in cursor.description]
        records = [dict(zip(col_names, row)) for row in rows]

        logger.info('failure 모델 재시도 시작', retry_count=len(records))

        failure_model_cfg = self.cfg.get_llm_failure_model_config()
        sampling_cfg = self.cfg.get_llm_sampling_config()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        prompt = get_prompt(self.cfg.get_llm_prompt_mode())
        category = self.cfg.get_llm_source_category()

        extractor = MDRExtractor(
            **failure_model_cfg,
            sampling_config=sampling_cfg,
            prompt=prompt,
        )
        results = extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_name=f"{checkpoint_cfg['prefix']}_{category}_failure",
        )

        logger.info('failure 모델 재시도 완료', result_count=len(results))
        return results

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
    def add_scd2_metadata(self, cursor: SnowflakeCursor, category: str = None):
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

        run_cats = [category] if category else list(self.stage.keys())
        for category in run_cats:
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
        # 중복 없이 순서 유지
        select_cols = list(dict.fromkeys(hover_cols + categorical_cols + [text_col]))

        logger.info('clustering 데이터 로드 시작', source=self.llm_join_table, columns=select_cols)
        sql = build_cte_sql(ctes=[], from_clause=self.llm_join_table, select_cols=select_cols)
        cursor.execute(sql)
        df = cursor.fetch_pandas_all()
        logger.info('clustering 데이터 로드 완료', rows=len(df))
        return df

    def run_clustering_tuning(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Step 2-5: vocab filter → embed → Optuna 튜닝 + 베스트 모델 저장.

        Returns:
            (embeddings, df_processed) — load_and_predict에 재사용
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
        train_and_save(
            embeddings=embeddings,
            save_dir=self.cfg.get_clustering_model_dir(),
            df=df,
            categorical_cols=self.cfg.get_clustering_categorical_columns(),
            onehot_weight=self.cfg.get_clustering_onehot_weight(),
            umap_params=self.cfg.get_clustering_umap_params(),
            hdbscan_params=self.cfg.get_clustering_hdbscan_params(),
            validity_params=self.cfg.get_clustering_validity_params(),
            scoring_params=self.cfg.get_clustering_scoring_params(),
            optuna_params=self.cfg.get_clustering_optuna_params(),
        )
        return embeddings, df

    def run_clustering_prediction(
        self, df: pd.DataFrame, embeddings: np.ndarray,
    ) -> Tuple[np.ndarray, Dict]:
        """Step 6: 저장된 베스트 모델로 전체 클러스터링"""
        from maude_early_alert.preprocessors.clustering import load_and_predict
        labels, metadata = load_and_predict(
            embeddings=embeddings,
            model_dir=self.cfg.get_clustering_model_dir(),
            df=df,
        )
        return labels, metadata

    @with_context
    def save_clustering_results(
        self, cursor: SnowflakeCursor, df: pd.DataFrame, labels: np.ndarray,
    ):
        """Step 5: clustering labels를 네 컬럼 기준 JOIN으로 원본 테이블에 붙여 _CLUSTERED 저장"""
        categorical_cols = self.cfg.get_clustering_categorical_columns()
        text_col = self.cfg.get_clustering_text_column()
        join_keys = categorical_cols + [text_col]

        temp_table = f'{self.clustering_target_table}_STG'
        col_defs = ', '.join(f'{c} VARCHAR' for c in join_keys) + ', CLUSTER INT'
        cursor.execute(f'CREATE OR REPLACE TEMPORARY TABLE {temp_table} ({col_defs})')

        rows = [
            tuple(row[c] for c in join_keys) + (int(label),)
            for row, label in zip(df[join_keys].to_dict('records'), labels.tolist())
        ]
        placeholders = ', '.join(['%s'] * (len(join_keys) + 1))
        col_names = ', '.join(join_keys) + ', CLUSTER'
        cursor.executemany(f'INSERT INTO {temp_table} ({col_names}) VALUES ({placeholders})', rows)

        join_clause = build_join_clause(
            left_table=self.llm_join_table,
            right_table=temp_table,
            on_columns=join_keys,
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

    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
    )

    pipeline = SilverPipeline(stage={'event': 14, 'udi': 7}, logical_date=pendulum.now())

    cursor = conn.cursor()
    try:
        logger.info('Silver 파이프라인 시작')
        # ── Silver 14단계 ──────────────────────────────────────────────
        # pipeline.filter_dedup_rows(cursor)
        # pipeline.flatten(cursor)
        # pipeline.filter_scoping(cursor)
        # pipeline.combine_mdr_text(cursor)
        # pipeline.extract_primary_udi_di(cursor)
        # pipeline.select_columns(cursor)
        # pipeline.impute_missing_values(cursor)
        # pipeline.clean_values(cursor)
        # pipeline.apply_company_alias(cursor)
        # pipeline.fuzzy_match_manufacturer(cursor)
        # pipeline.cast_types(cursor)
        # pipeline.extract_udi_di(cursor)
        # pipeline.match_udi(cursor)
        # pipeline.filter_quality(cursor)
        # pipeline.select_columns(cursor, final=True)
        # pipeline.add_scd2_metadata(cursor)
        logger.info('Silver 14단계 완료')

        # ── LLM 추출 (Design 2) ───────────────────────────────────────
        # 1. source → records 추출
        # 2. 1차 LLM 추출 → checkpoint_event.db (중단 시 재개 가능)
        # 3. _EXTRACTED UPSERT (Snowflake 영속화)
        # 4. failure 모델 재시도 → checkpoint_event_failure.db (중단 시 재개 가능)
        # 5. _EXTRACTED 증분 UPSERT
        # 6. source + _EXTRACTED LEFT JOIN → _LLM_EXTRACTED 생성
        logger.info('LLM 추출 단계 시작')
        records = pipeline.extract_mdr_text(cursor)
        results = pipeline.run_llm_extraction(records)
        pipeline.load_extraction_results(cursor, results)
        results = pipeline.run_failure_model_retry(cursor)
        pipeline.load_extraction_results(cursor, results)
        pipeline.join_extraction(cursor)

        logger.info('Silver 파이프라인 완료')

    except Exception:
        logger.error('Silver 파이프라인 실패', exc_info=True)
        raise
    finally:
        cursor.close()
        conn.close()

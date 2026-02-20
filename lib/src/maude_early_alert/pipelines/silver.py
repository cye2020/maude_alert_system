from typing import Dict, List

import pendulum
import structlog

from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase, with_context
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.preprocessors.column_select import build_select_columns_sql
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
    build_extracted_join_sql,
    prepare_insert_data,
    MDRExtractor,
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
        cursor.execute(f'CREATE OR REPLACE TABLE {next_table} AS\n{sql}')

    def _filter_step(self, key: str, cursor: SnowflakeCursor):
        """filtering.yaml의 key에 해당하는 스텝을 category별로 실행"""
        for category, step in self.cfg.get_filtering_step(key).items():
            source = self._stage_table(category)
            if step['type'] == 'standalone':
                sql = build_filter_sql(source, **{k: v for k, v in step.items() if k in ('where', 'qualify')})
            else:  # chain
                sql = build_filter_pipeline(source, step['ctes'], step['final'])
            self._create_next_stage(cursor, category, sql)

    @with_context
    def filter_dedup_rows(self, cursor: SnowflakeCursor):
        self._filter_step('DEDUP', cursor)

    @with_context
    def filter_scoping(self, cursor: SnowflakeCursor):
        self._filter_step('SCOPING', cursor)

    @with_context
    def filter_quality(self, cursor: SnowflakeCursor):
        self._filter_step('QUALITY_FILTER', cursor)

    def _fetch_schema(self, cursor: SnowflakeCursor, table_name: str):
        """top-level 키/타입 + 배열별 nested 스키마 조회"""
        cursor.execute(build_top_keys_sql(table_name))
        top = {str(k): str(t) for k, t in cursor.fetchall() if k is not None}

        array_schemas = {}
        for name, typ in top.items():
            if typ.upper() == 'ARRAY':
                cursor.execute(build_array_keys_sql(table_name, name))
                array_schemas[name] = parse_array_keys_result(cursor.fetchall())

        return top, array_schemas

    @with_context
    def flatten(self, cursor: SnowflakeCursor):
        for category in self.cfg.get_flatten_categories():
            table_name = self._stage_table(category)
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
            print(flatten_sql)
            self._create_next_stage(cursor, category, flatten_sql)

    @with_context
    def combine_mdr_text(self, cursor: SnowflakeCursor):
        category = self.cfg.get_combine_category()
        sql = build_combine_mdr_text_sql(self._stage_table(category), **self.cfg.get_combine_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def extract_primary_udi_di(self, cursor: SnowflakeCursor):
        category = self.cfg.get_primary_category()
        sql = build_primary_udi_di_sql(self._stage_table(category), **self.cfg.get_primary_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def extract_udi_di(self, cursor: SnowflakeCursor):
        category = self.cfg.get_extract_udi_di_category()
        sql = build_extract_udi_di_sql(self._stage_table(category), **self.cfg.get_extract_udi_di_columns())
        self._create_next_stage(cursor, category, sql)

    @with_context
    def apply_company_alias(self, cursor: SnowflakeCursor):
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
        sql = build_manufacturer_fuzzy_match_sql(
            target=self._stage_table(target_category),
            source=self._stage_table(self.cfg.get_fuzzy_match_source_category()),
            mfr_col=self.cfg.get_fuzzy_match_mfr_col(),
            udf_schema=udf_schema,
            threshold=self.cfg.get_fuzzy_match_threshold(),
        )
        self._create_next_stage(cursor, target_category, sql)

    @with_context
    def select_columns(self, cursor: SnowflakeCursor, final: bool = False):
        categories = self.cfg.get_final_categories() if final else self.cfg.get_column_categories()
        get_cols = self.cfg.get_final_cols if final else self.cfg.get_column_cols
        for category in categories:
            sql = build_select_columns_sql(get_cols(category), self._stage_table(category))
            self._create_next_stage(cursor, category, sql)

    @with_context
    def impute_missing_values(self, cursor: SnowflakeCursor):
        for category in self.cfg.get_imputation_categories():
            sql = build_mode_fill_sql(
                group_to_target=self.cfg.get_imputation_mode(category),
                table_name=self._stage_table(category),
                table_alias=self.cfg.get_imputation_alias(category),
            )
            self._create_next_stage(cursor, category, sql)

    @with_context
    def clean_values(self, cursor: SnowflakeCursor):
        udf_schema = f"{self.cfg.get_snowflake_udf_database()}.{self.cfg.get_snowflake_udf_schema()}"
        for category in self.cfg.get_cleaning_categories():
            sql = build_clean_sql(
                table_name=self._stage_table(category),
                config=self.cfg.get_cleaning_config(category),
                udf_schema=udf_schema,
            )
            self._create_next_stage(cursor, category, sql)

    @with_context
    def cast_types(self, cursor: SnowflakeCursor):
        for category in self.cfg.get_column_categories():
            sql = build_type_cast_sql(
                columns=self.cfg.get_column_cols(category),
                input_table=self._stage_table(category),
            )
            self._create_next_stage(cursor, category, sql)

    @with_context
    def match_udi(self, cursor: SnowflakeCursor):
        target_cat = self.cfg.get_matching_target_category()
        source_cat = self.cfg.get_matching_source_category()
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
        category = self.cfg.get_llm_source_category()
        sql = build_mdr_text_extract_sql(
            table_name=self._stage_table(category),
            columns=self.cfg.get_llm_source_columns(),
            where=self.cfg.get_llm_source_where(),
        )
        cursor.execute(sql)
        rows = cursor.fetchall()
        seen = {}
        for r in rows:
            if r[0] and r[0] not in seen:
                seen[r[0]] = {'mdr_text': r[0], 'product_problems': r[1] or ''}
        unique_records = list(seen.values())
        logger.info('MDR_TEXT 추출 완료', total=len(rows), unique=len(unique_records))
        return unique_records

    def run_llm_extraction(self, records: List) -> List[dict]:
        """vLLM 배치 처리 (cursor 불필요, GPU 작업)"""
        model_cfg = self.cfg.get_llm_model_config()
        sampling_cfg = self.cfg.get_llm_sampling_config()
        checkpoint_cfg = self.cfg.get_llm_checkpoint_config()
        prompt = get_prompt(self.cfg.get_llm_prompt_mode())

        extractor = MDRExtractor(
            **model_cfg,
            sampling_config=sampling_cfg,
            prompt=prompt,
        )
        return extractor.process_batch(
            records,
            checkpoint_dir=checkpoint_cfg['dir'],
            checkpoint_interval=checkpoint_cfg['interval'],
            checkpoint_prefix=checkpoint_cfg['prefix'],
        )

    @with_context
    def load_extraction_results(self, cursor: SnowflakeCursor, results: List[dict]):
        """추출 결과를 Snowflake에 적재 (temp table -> MERGE)"""
        columns = self.cfg.get_llm_extracted_columns()
        pk_col = self.cfg.get_llm_extracted_pk_column()
        non_pk_cols = self.cfg.get_llm_extracted_non_pk_columns()

        category = self.cfg.get_llm_source_category()
        extracted_table = f"{self._stage_table(category)}{self.cfg.get_llm_extracted_suffix()}"

        insert_data = prepare_insert_data(results, columns)
        if not insert_data:
            logger.warning('적재할 추출 데이터가 없습니다')
            return

        cursor.execute(build_ensure_extracted_table_sql(extracted_table, columns))

        temp_table = f"{extracted_table}_STG_{self.logical_date.strftime('%Y%m%d')}"
        cursor.execute(build_create_extract_temp_sql(temp_table, columns))
        stage_insert_sql = build_extract_stage_insert_sql(temp_table, columns)
        cursor.executemany(stage_insert_sql, insert_data)
        cursor.execute(build_extract_merge_sql(extracted_table, temp_table, pk_col, non_pk_cols))
        logger.info('추출 결과 적재 완료', count=len(insert_data))

    @with_context
    def join_extraction(self, cursor: SnowflakeCursor):
        """원본 EVENT + 추출 결과 LEFT JOIN -> 최종 stage 생성"""
        category = self.cfg.get_llm_source_category()
        pk_col = self.cfg.get_llm_extracted_pk_column()
        non_pk_cols = self.cfg.get_llm_extracted_non_pk_columns()

        sql = build_extracted_join_sql(
            base_table=self._stage_table(category),
            extracted_suffix=self.cfg.get_llm_extracted_suffix(),
            non_pk_columns=non_pk_cols,
            pk_column=pk_col,
        )
        self._create_next_stage(cursor, category, sql)

    def _cleanup_stages(self, cursor: SnowflakeCursor):
        """중간 stage 테이블 전부 삭제"""
        for category, current in self.stage.items():
            for i in range(1, current + 1):
                table = f'{category}_stage_{i:02d}'.upper()
                cursor.execute(f'DROP TABLE IF EXISTS {table}')


if __name__ == '__main__':
    import snowflake.connector
    from maude_early_alert.utils.secrets import get_secret

    secret = get_secret('snowflake/de')

    conn = snowflake.connector.connect(
        user=secret['user'],
        password=secret['password'],
        account=secret['account'],
        warehouse=secret['warehouse'],
    )

    pipeline = SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=pendulum.now())

    try:
        cursor = conn.cursor()

        # ── Silver 14단계 ──────────────────────────────────────────────
        pipeline.filter_dedup_rows(cursor)
        pipeline.flatten(cursor)
        pipeline.filter_scoping(cursor)
        pipeline.combine_mdr_text(cursor)
        pipeline.extract_primary_udi_di(cursor)
        pipeline.select_columns(cursor)
        pipeline.impute_missing_values(cursor)
        pipeline.clean_values(cursor)
        pipeline.apply_company_alias(cursor)
        pipeline.fuzzy_match_manufacturer(cursor)
        pipeline.cast_types(cursor)
        pipeline.extract_udi_di(cursor)
        pipeline.match_udi(cursor)
        pipeline.select_columns(cursor, final=True)

        # ── LLM 추출 4단계 ────────────────────────────────────────────
        records = pipeline.extract_mdr_text(cursor)
        results = pipeline.run_llm_extraction(records)
        pipeline.load_extraction_results(cursor, results)
        pipeline.join_extraction(cursor)

    finally:
        cursor.close()
        conn.close()

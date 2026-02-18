from typing import Dict
import pendulum
import structlog

from snowflake.connector.cursor import SnowflakeCursor

from maude_early_alert.loaders.snowflake_base import SnowflakeBase
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.preprocessors.flatten import (
    build_array_keys_sql, 
    build_flatten_sql,
    build_top_keys_sql,
    parse_array_keys_result
)
from maude_early_alert.preprocessors.row_filter import build_filter_sql, build_filter_pipeline

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
        return f'{category}_stage_{self.stage[category]:02d}'.upper()

    def _create_next_stage(self, category: str, sql: str, cursor: SnowflakeCursor):
        """다음 stage 테이블 생성 후 self.stage 업데이트"""
        self.stage[category] += 1
        next_table = self._stage_table(category)
        cursor.execute(f'CREATE OR REPLACE TABLE {next_table} AS\n{sql}')

    def _filter_step(self, key: str, cursor: SnowflakeCursor, source: str = None):
        """filtering.yaml의 key에 해당하는 스텝을 category별로 실행"""
        for category, step in self.cfg.get_filtering_step(key).items():
            source = source if source else self._stage_table(category)
            if step['type'] == 'standalone':
                sql = build_filter_sql(source, **{k: v for k, v in step.items() if k in ('where', 'qualify')})
            else:  # chain
                sql = build_filter_pipeline(source, step['ctes'], step['final'])
            self._create_next_stage(category, sql, cursor)

    def filter_dedup_rows(self, cursor: SnowflakeCursor, source: str = None):
        self._filter_step('DEDUP', cursor, source)

    def filter_scoping(self, cursor: SnowflakeCursor, source: str = None):
        self._filter_step('SCOPING', cursor, source)

    def filter_quality(self, cursor: SnowflakeCursor, source: str = None):
        self._filter_step('QUALITY_FILTER', cursor, source)

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
            self._create_next_stage(category, flatten_sql, cursor)
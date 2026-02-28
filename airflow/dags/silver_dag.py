from airflow.sdk import dag, task, task_group, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars
from typing import Dict, List

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.silver import SilverPipeline
from maude_early_alert.assets import MAUDE_BRONZE_ASSETS, MAUDE_SILVER_ASSETS

configure_logging(level='INFO', log_file='silver.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_default'


def _step_cat(
    stage: Dict[str, int],
    logical_date: pendulum.DateTime,
    run_id: str,
    dag_id: str,
    method_name: str,
    category: str,
    **method_kwargs,
) -> Dict[str, int]:
    """카테고리 단위 파이프라인 실행 → {category: updated_stage_value} 반환"""
    bind_contextvars(dag_id=dag_id, run_id=run_id)
    try:
        pipeline = SilverPipeline(stage=stage, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
            getattr(pipeline, method_name)(cursor, category=category, **method_kwargs)
        logger.info(f'{method_name}[{category}] completed', stage_value=pipeline.stage[category])
        return {category: pipeline.stage[category]}
    except Exception as e:
        logger.error(f'{method_name}[{category}] failed', error=str(e), exc_info=True)
        raise AirflowException(f'{method_name}[{category}] 실패: {e}') from e


def _step(
    stage: Dict[str, int],
    logical_date: pendulum.DateTime,
    run_id: str,
    dag_id: str,
    method_name: str,
    **method_kwargs,
) -> Dict[str, int]:
    """단일 카테고리 파이프라인 실행 → 전체 stage dict 반환"""
    bind_contextvars(dag_id=dag_id, run_id=run_id)
    try:
        pipeline = SilverPipeline(stage=stage, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
            getattr(pipeline, method_name)(cursor, **method_kwargs)
        logger.info(f'{method_name} completed', stage=dict(pipeline.stage))
        return dict(pipeline.stage)
    except Exception as e:
        logger.error(f'{method_name} failed', error=str(e), exc_info=True)
        raise AirflowException(f'{method_name} 실패: {e}') from e


@dag(
    dag_id='maude_silver',
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    schedule=MAUDE_BRONZE_ASSETS,
    catchup=False,
    max_active_runs=1,
    tags=['maude', 'silver', 'snowflake'],
    description='Snowflake Bronze 데이터를 Silver 레이어로 전처리 및 적재',
    default_args={
        'retries': 2,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_silver():
    """Bronze 데이터를 14단계 전처리 후 Silver SCD2 테이블에 적재하는 파이프라인"""

    # ── task_group 팩토리 ────────────────────────────────────────────
    def _cat_group(group_id: str, method_name: str, is_last: bool = False, **method_kwargs):
        """카테고리별 병렬 실행 task_group 생성.
        내부 구조: get_cats → run(expand) → merge
        반환값: merged stage dict (또는 is_last=True 시 MAUDE_SILVER_ASSETS emit)
        """
        @task_group(group_id=group_id)
        def _group(stage: dict) -> dict:

            @task
            def get_cats(stage: dict, logical_date: pendulum.DateTime) -> List[str]:
                return SilverPipeline(
                    stage=stage, logical_date=logical_date
                ).get_categories(method_name if not method_kwargs.get('final') else 'select_columns_final')

            @task
            def run_step(category: str, stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
                return _step_cat(stage, logical_date, run_id, dag.dag_id, method_name, category, **method_kwargs)

            @task(outlets=MAUDE_SILVER_ASSETS if is_last else [])
            def merge(updates: List[dict]) -> dict:
                return {k: v for d in updates for k, v in d.items()}

            cats = get_cats(stage)
            results = run_step.partial(stage=stage).expand(category=cats)
            return merge(results)

        return _group

    # ── 단일 카테고리 task 정의 ──────────────────────────────────────
    @task
    def combine_mdr_text(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'combine_mdr_text')

    @task
    def extract_primary_udi_di(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'extract_primary_udi_di')

    @task
    def apply_company_alias(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'apply_company_alias')

    @task
    def fuzzy_match_manufacturer(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'fuzzy_match_manufacturer')

    @task
    def extract_udi_di(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'extract_udi_di')

    @task
    def match_udi(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> dict:
        return _step(stage, logical_date, run_id, dag.dag_id, 'match_udi')

    @task
    def merge_s3_s4(a: dict, b: dict) -> dict:
        return {k: max(a.get(k, 0), b.get(k, 0)) for k in set(a) | set(b)}

    @task(trigger_rule='all_done')
    def cleanup_stages(stage: dict, logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> None:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        try:
            pipeline = SilverPipeline(stage=stage, logical_date=logical_date)
            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.cleanup_stages(cursor)
            logger.info('Stage 정리 완료', stage=stage)
        except Exception as e:
            logger.error('Stage 정리 실패', error=str(e), exc_info=True)
            raise AirflowException(f'Stage 정리 실패: {e}') from e

    # ── 다중 카테고리 task_group 생성 ────────────────────────────────
    dedup          = _cat_group('filter_dedup_rows',     'filter_dedup_rows')
    flatten        = _cat_group('flatten',               'flatten')
    scoping        = _cat_group('filter_scoping',        'filter_scoping')
    select_cols    = _cat_group('select_columns',        'select_columns')
    impute         = _cat_group('impute_missing_values', 'impute_missing_values')
    clean          = _cat_group('clean_values',          'clean_values')
    cast           = _cat_group('cast_types',            'cast_types')
    quality        = _cat_group('filter_quality',        'filter_quality')
    select_final   = _cat_group('select_columns_final',  'select_columns', final=True)
    scd2           = _cat_group('add_scd2_metadata',     'add_scd2_metadata', is_last=True)

    # ── Task 의존성 체인 (stage XCom 전달) ───────────────────────────
    init = {'event': 0, 'udi': 0}

    s0  = dedup(init)
    s1  = flatten(s0)
    s2  = scoping(s1)
    s3  = combine_mdr_text(s2)
    s4  = extract_primary_udi_di(s2)
    s5  = select_cols(merge_s3_s4(s3, s4))
    s6  = impute(s5)
    s7  = clean(s6)
    s8  = apply_company_alias(s7)
    s9  = fuzzy_match_manufacturer(s8)
    s10 = cast(s9)
    s11 = extract_udi_di(s10)
    s12 = match_udi(s11)
    s13 = quality(s12)
    s14 = select_final(s13)
    s15 = scd2(s14)
    cleanup_stages(s15)


maude_silver()

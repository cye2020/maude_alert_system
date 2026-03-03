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
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.assets import MAUDE_BRONZE_ASSETS, MAUDE_SILVER_ASSETS

configure_logging(level='INFO', log_file='silver.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_de'


def _run_pipeline_step(
    stage: Dict[str, int],
    logical_date: pendulum.DateTime,
    run_id: str,
    dag_id: str,
    step: str,
    category: str | None,
) -> Dict[str, int]:
    """SilverPipeline.run_step() 단일 진입점 호출 → 업데이트된 stage dict 반환"""
    bind_contextvars(dag_id=dag_id, run_id=run_id)
    try:
        pipeline = SilverPipeline(stage=stage, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
            pipeline.run_step(cursor, step, category=category)
        logger.info(f'{step}[{category or "-"}] completed', stage=dict(pipeline.stage))
        return dict(pipeline.stage)
    except Exception as e:
        label = f'{step}[{category or "-"}]'
        logger.error(f'{label} failed', error=str(e), exc_info=True)
        raise AirflowException(f'{label} 실패: {e}') from e


def _make_pipeline_task(step: str, category: str, is_last: bool):
    """단일 파이프라인 스텝 task 생성 (루프 closure 안전을 위해 팩토리 함수로 분리)"""
    @task(task_id=step, outlets=MAUDE_SILVER_ASSETS if is_last else [])
    def _run(stage: dict, run_id: str, dag: DAG) -> dict:
        return _run_pipeline_step(
            stage, pendulum.now('Asia/Seoul'), run_id, dag.dag_id, step, category,
        )
    return _run


def _pipeline_group(
    group_id: str,
    category: str,
    steps: List[str],
    is_last: bool = False,
):
    """YAML 스텝 목록으로 카테고리 순차 task_group 동적 생성"""
    @task_group(group_id=group_id)
    def _group(stage: dict) -> dict:
        cur = stage
        for step in steps:
            is_step_last = is_last and (step == steps[-1])
            cur = _make_pipeline_task(step, category, is_step_last)(cur)
        return cur
    return _group


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
    """Bronze 데이터를 카테고리별 순차 파이프라인으로 처리 후 Silver SCD2 테이블에 적재.

    event_pre / udi 는 병렬 실행.
    event_post 는 udi add_scd2_metadata 완료 후 실행
    (fuzzy_match_manufacturer, match_udi 는 udi_CURRENT 를 소스로 사용).
    """

    # ── pipeline.yaml 로드 (DAG parse time) ─────────────────────
    pipeline_cfg = get_config().silver.get_pipeline_config()
    event_pre_steps  = pipeline_cfg['event']['pre_sync']
    event_post_steps = pipeline_cfg['event']['post_sync']
    udi_steps        = pipeline_cfg['udi']

    # ── 카테고리별 task_group ─────────────────────────────────────
    event_pre  = _pipeline_group('event_pre',  'event', event_pre_steps)
    event_post = _pipeline_group('event_post', 'event', event_post_steps, is_last=True)
    udi_pipe   = _pipeline_group('udi',        'udi',   udi_steps)

    # ── 동기화 태스크 ─────────────────────────────────────────────
    @task
    def sync_after_udi(event_stage: dict, _udi_done: dict) -> dict:
        """udi add_scd2_metadata 완료 신호를 받아 event stage 반환 (Airflow 의존성 동기화)"""
        return event_stage

    # ── Stage 정리 ────────────────────────────────────────────────
    @task
    def merge_stages(a: dict, b: dict) -> dict:
        """event + udi stage dict 합산 (cleanup_stages 용)"""
        return {**a, **b}

    @task(trigger_rule='all_done')
    def cleanup_stages(stage: dict, run_id: str, dag: DAG) -> None:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        try:
            pipeline = SilverPipeline(stage=stage, logical_date=pendulum.now('Asia/Seoul'))
            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.cleanup_stages(cursor)
            logger.info('Stage 정리 완료', stage=stage)
        except Exception as e:
            logger.error('Stage 정리 실패', error=str(e), exc_info=True)
            raise AirflowException(f'Stage 정리 실패: {e}') from e

    # ── 의존성 체인 ───────────────────────────────────────────────
    event_init = {'event': 0}
    udi_init   = {'udi': 0}

    u_out    = udi_pipe(udi_init)                  # udi 브랜치 (독립 실행)
    e_pre    = event_pre(event_init)               # event pre-sync 브랜치
    e_synced = sync_after_udi(e_pre, u_out)        # udi 완료 대기
    e_post   = event_post(e_synced)                # event post-sync (udi_CURRENT 소스)

    cleanup_stages(merge_stages(e_post, u_out))


maude_silver()

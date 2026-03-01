from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.gold import GoldPipeline
from maude_early_alert.assets import MAUDE_CLUSTERED_ASSET
from maude_early_alert.utils.config_loader import load_config

configure_logging(level='INFO', log_file='gold.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_default'

_storage = load_config('storage')
_DATABASE    = _storage['snowflake']['transform']['database']
_SCHEMA      = _storage['snowflake']['transform']['schema']
_GOLD_SCHEMA = _storage['snowflake']['gold']['schema']
_cfg         = load_config('gold')


@dag(
    dag_id='maude_gold',
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    schedule=[MAUDE_CLUSTERED_ASSET],
    catchup=False,
    max_active_runs=1,
    tags=['maude', 'gold', 'snowflake'],
    description='EVENT_CLUSTERED 기반 Gold 레이어 집계 및 이상 탐지',
    default_args={
        'retries': 2,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_gold():
    """EVENT_CLUSTERED → Gold 집계(dashboard) + 이상징후시그널(spike/stat) 파이프라인"""

    @task
    def aggregate(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> None:
        """클러스터/결함/제품/전체 월간 지표 집계 → Gold dashboard 테이블 적재"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        snapshot_date = logical_date.strftime('%Y-%m-%d')
        try:
            pipeline = GoldPipeline(
                database=_DATABASE,
                schema=_SCHEMA,
                gold_schema=_GOLD_SCHEMA,
            )
            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.run_cluster_dashboard  (cursor, _cfg['source_table'], snapshot_date)
                pipeline.run_defect_spike       (cursor, _cfg['source_table'], snapshot_date)
                pipeline.run_product_dashboard  (cursor, _cfg['source_table'], snapshot_date)
                pipeline.run_overview_dashboard (cursor, _cfg['source_table'], snapshot_date)
            logger.info('집계 완료', snapshot_date=snapshot_date)
        except Exception as e:
            logger.error('집계 실패', error=str(e), exc_info=True)
            raise AirflowException(f'집계 실패: {e}') from e

    @task
    def anomaly_signal(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> None:
        """Spike Detection + Statistical Analysis → Gold 이상 탐지 테이블 적재"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        snapshot_date = logical_date.strftime('%Y-%m-%d')
        try:
            pipeline = GoldPipeline(
                database=_DATABASE,
                schema=_SCHEMA,
                gold_schema=_GOLD_SCHEMA,
            )
            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.run_spike_detection     (cursor, _cfg['source_table'], snapshot_date)
                pipeline.run_statistical_analysis(cursor, _cfg['source_table'], snapshot_date)
            logger.info('이상징후 탐지 완료', snapshot_date=snapshot_date)
        except Exception as e:
            logger.error('이상징후 탐지 실패', error=str(e), exc_info=True)
            raise AirflowException(f'이상징후 탐지 실패: {e}') from e

    aggregate()
    anomaly_signal()


maude_gold()

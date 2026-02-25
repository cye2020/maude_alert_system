from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars
from typing import Any, Dict, List

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.bronze import BronzePipeline
from maude_early_alert.assets import MAUDE_S3_ASSET, MAUDE_BRONZE_ASSETS

configure_logging(level='INFO', log_file='bronze.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_default'


@dag(
    dag_id='maude_bronze',
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    schedule=[MAUDE_S3_ASSET],
    catchup=False,
    max_active_runs=1,
    tags=['maude', 'bronze', 'snowflake'],
    description='S3 MAUDE 데이터를 Snowflake Bronze 레이어로 적재',
    default_args={
        'retries': 2,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_bronze():
    """S3에 적재된 MAUDE 데이터를 Snowflake Bronze 레이어로 로드하는 파이프라인"""

    @task(outlets=MAUDE_BRONZE_ASSETS)
    def load_all(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> List[Dict[str, Any]]:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)

        try:
            pipeline = BronzePipeline(logical_date)
            batch_id = f"bronze_{logical_date.strftime('%Y%m%d_%H%M%S')}"

            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                results = pipeline.load_all(cursor, batch_id=batch_id)

            logger.info('Bronze load completed', table_count=len(results), batch_id=batch_id)
            return results

        except Exception as e:
            logger.error('Bronze load failed', error=str(e), exc_info=True)
            raise AirflowException(f'Bronze 적재 실패: {e}') from e

    load_all()


maude_bronze()

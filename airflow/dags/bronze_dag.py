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

SNOWFLAKE_CONN_ID = 'snowflake_de'


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

    @task
    def get_tables() -> List[str]:
        return BronzePipeline(pendulum.now('Asia/Seoul')).get_tables()

    @task
    def load_table(table: str, run_id: str, dag: DAG) -> Dict[str, Any]:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id, table=table)
        try:
            kst = pendulum.now('Asia/Seoul')
            batch_id = f"maude_{kst.strftime('%Y%m')}"
            pipeline = BronzePipeline(kst)

            hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                result = pipeline.load_table(cursor, table_name=table, batch_id=batch_id)

            logger.info('Table load completed', table=table, **result)
            return result

        except Exception as e:
            logger.error('Table load failed', table=table, error=str(e), exc_info=True)
            raise AirflowException(f'Bronze 적재 실패 [{table}]: {e}') from e

    @task(outlets=MAUDE_BRONZE_ASSETS)
    def finalize(results: List[Dict[str, Any]], run_id: str, dag: DAG) -> None:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        logger.info('Bronze load all completed', table_count=len(results))

    tables = get_tables()
    loaded = load_table.expand(table=tables)
    finalize(loaded)


maude_bronze()

from airflow.sdk import dag, task
from airflow.exceptions import AirflowException
import pendulum
import structlog
from structlog.contextvars import bind_contextvars
from typing import List

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.ingest import IngestPipeline

configure_logging(level='INFO', log_file='ingest.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


@dag(
    dag_id='maude_ingest',
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    schedule='@monthly',
    catchup=False,
    tags=['maude', 'ingest', 'bronze'],
    description='FDA MAUDE 데이터 추출 및 S3 적재',
    default_args={
        'retries': 2,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_ingest():
    """FDA MAUDE 데이터를 추출하여 S3에 적재하는 파이프라인"""

    @task
    def extract(logical_date: pendulum.DateTime, run_id: str, dag_id: str) -> List[str]:
        bind_contextvars(dag_id=dag_id, run_id=run_id)

        try:
            import requests
            pipeline = IngestPipeline(logical_date)

            with requests.Session() as session:
                data_urls = pipeline.extract(session)

            logger.info('Extract completed', count=len(data_urls), data_urls=data_urls)
            return data_urls

        except Exception as e:
            logger.error('Extract failed', error=str(e), exc_info=True)
            raise AirflowException(f'추출 실패: {e}') from e

    @task
    def s3_load(data_urls: List[str], logical_date: pendulum.DateTime, run_id: str, dag_id: str) -> None:
        bind_contextvars(dag_id=dag_id, run_id=run_id)

        try:
            import requests
            from airflow.providers.amazon.aws.hooks.s3 import S3Hook
            pipeline = IngestPipeline(logical_date)

            hook = S3Hook(aws_conn_id='aws_default')
            client = hook.get_conn()
            with requests.Session() as session:
                pipeline.s3_load(data_urls, client, session)

            logger.info('S3 load completed', count=len(data_urls))

        except Exception as e:
            logger.error('S3 load failed', error=str(e), exc_info=True)
            raise AirflowException(f'S3 적재 실패: {e}') from e

    # Task 의존성 정의
    data_urls = extract()
    s3_load(data_urls)


maude_ingest()

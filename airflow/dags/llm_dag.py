from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.silver import SilverPipeline
from maude_early_alert.assets import MAUDE_SILVER_ASSETS, MAUDE_LLM_ASSET

configure_logging(level='INFO', log_file='llm.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_default'
CHUNK_SIZE = 5000


@dag(
    dag_id='maude_llm',
    schedule=MAUDE_SILVER_ASSETS,
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    catchup=False,
    max_active_runs=1,
    tags=['maude', 'llm'],
    default_args={
        'retries': 1,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_llm():
    """MAUDE LLM 추출 파이프라인
    1. MDR 텍스트 추출 → LLM 청크 처리 → 적재
    2. Failure 후보 재시도 → 적재
    3. 추출 결과 JOIN
    """

    @task
    def llm_extract_and_load(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> int:
        """MDR 텍스트 추출 → LLM 청크 처리 → Snowflake 적재"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            # 1단계: MDR 텍스트 추출
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                records = pipeline.extract_mdr_text(cursor)

            # 2-3단계: LLM 청크 처리 + 적재
            total_chunks = (len(records) - 1) // CHUNK_SIZE + 1 if records else 0
            logger.info('LLM 청크 처리 시작', total=len(records), chunks=total_chunks)
            for chunk_idx in range(total_chunks):
                chunk = records[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
                logger.info('LLM 청크 처리', chunk=chunk_idx + 1, total_chunks=total_chunks, size=len(chunk))
                results = pipeline.run_llm_extraction(chunk)
                with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                    pipeline.load_extraction_results(cursor, results)

            logger.info('LLM 추출 완료', total=len(records), chunks=total_chunks)
            return len(records)
        except Exception as e:
            logger.error('LLM 추출 실패', error=str(e), exc_info=True)
            raise AirflowException(f'LLM 추출 실패: {e}') from e

    @task
    def failure_retry_and_load(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> int:
        """Failure 후보 조회 → failure 모델 청크 처리 → Snowflake 적재"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            # 4단계: failure 후보 조회
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                failure_records = pipeline.fetch_failure_candidates(cursor)

            if not failure_records:
                logger.info('failure 대상 없음, 스킵')
                return 0

            # 5-6단계: failure 모델 청크 처리 + 적재
            total_chunks = (len(failure_records) - 1) // CHUNK_SIZE + 1
            logger.info('failure 청크 처리 시작', total=len(failure_records), chunks=total_chunks)
            for chunk_idx in range(total_chunks):
                chunk = failure_records[chunk_idx * CHUNK_SIZE:(chunk_idx + 1) * CHUNK_SIZE]
                logger.info('failure 청크 처리', chunk=chunk_idx + 1, total_chunks=total_chunks, size=len(chunk))
                results = pipeline.run_failure_model_retry(chunk)
                with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                    pipeline.load_extraction_results(cursor, results)

            logger.info('failure 재시도 완료', total=len(failure_records), chunks=total_chunks)
            return len(failure_records)
        except Exception as e:
            logger.error('failure 재시도 실패', error=str(e), exc_info=True)
            raise AirflowException(f'failure 재시도 실패: {e}') from e

    @task(outlets=[MAUDE_LLM_ASSET])
    def join_extraction(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> None:
        """추출 결과 JOIN → {category}_LLM_EXTRACTED 테이블 생성"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.join_extraction(cursor)
            logger.info('LLM 추출 JOIN 완료')
        except Exception as e:
            logger.error('JOIN 실패', error=str(e), exc_info=True)
            raise AirflowException(f'JOIN 실패: {e}') from e

    @task(trigger_rule='all_done')
    def cleanup_checkpoint(logical_date: pendulum.DateTime) -> None:
        """체크포인트 삭제 (성공/실패 무관하게 항상 실행)"""
        SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=logical_date).cleanup_extraction_checkpoint()

    s0 = llm_extract_and_load()
    s1 = failure_retry_and_load()
    s2 = join_extraction()
    s3 = cleanup_checkpoint()

    s0 >> s1 >> s2 >> s3


maude_llm()

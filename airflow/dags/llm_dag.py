import json
from pathlib import Path

from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.llm_pipeline import LLMPipeline
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.assets import MAUDE_SILVER_ASSETS, MAUDE_LLM_ASSET

configure_logging(level='INFO', log_file='llm.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_de'
CHUNK_SIZE = 5000

_cfg = get_config().silver
VLLM_PYTHON = _cfg.get_llm_vllm_python()


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
    """MAUDE LLM 추출 파이프라인 (청크 기반 Dynamic Task Mapping)

    1. MDR 텍스트 추출 → 청크별 /tmp 파일 저장
    2. vllm-env에서 청크별 LLM 배치 추출 (순차 실행, GPU 충돌 방지)
    3. 청크별 Snowflake 적재
    4. Failure 후보 조회 → 청크별 /tmp 파일 저장
    5. vllm-env에서 청크별 failure 모델 추출 (순차 실행)
    6. 청크별 Snowflake 적재
    7. 추출 결과 JOIN
    """

    @task
    def extract_records(run_id: str, dag: DAG) -> list[str]:
        """Snowflake에서 MDR 텍스트 추출 → 청크별 /tmp 파일 저장. 경로 리스트 반환."""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = LLMPipeline(logical_date=pendulum.now('Asia/Seoul'))
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                records = pipeline.extract_mdr_text(cursor)
            if not records:
                logger.info('추출할 MDR 레코드 없음')
                return []
            paths = []
            for i, start in enumerate(range(0, len(records), CHUNK_SIZE)):
                chunk = records[start:start + CHUNK_SIZE]
                path = f'/tmp/llm_records_{run_id}_chunk{i}.json'
                Path(path).write_text(json.dumps(chunk, ensure_ascii=False))
                paths.append(path)
            logger.info('MDR 텍스트 추출 완료', total=len(records), chunks=len(paths))
            return paths
        except Exception as e:
            logger.error('MDR 텍스트 추출 실패', error=str(e), exc_info=True)
            raise AirflowException(f'MDR 텍스트 추출 실패: {e}') from e

    @task.external_python(python=VLLM_PYTHON, expect_airflow=False, max_active_tis_per_dagrun=1)
    def llm_extract_chunk(input_path: str, logical_date: str) -> str:
        """vllm-env에서 청크 단위 LLM 배치 추출. 결과를 /tmp 파일로 저장 후 경로 반환.

        max_active_tis_per_dagrun=1: 동시에 하나의 vllm 프로세스만 실행 (GPU 경합 방지)
        """
        import json
        import re
        from pathlib import Path
        import pendulum
        from maude_early_alert.pipelines.llm_pipeline import LLMPipeline

        m = re.search(r'_chunk(\d+)\.json$', input_path)
        chunk_idx = m.group(1) if m else '0'
        records = json.loads(Path(input_path).read_text())
        output_path = input_path.replace('llm_records_', 'llm_results_')
        pipeline = LLMPipeline(logical_date=pendulum.parse(logical_date))
        results = pipeline.run_llm_batch(records, chunk_idx=chunk_idx)
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, default=str))
        return output_path

    @task
    def load_chunk_results(output_path: str, run_id: str, dag: DAG) -> int:
        """LLM 청크 결과 파일 → Snowflake 적재 후 임시 파일 삭제"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = LLMPipeline(logical_date=pendulum.now('Asia/Seoul'))
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            results = json.loads(Path(output_path).read_text())
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.load_extraction_results(cursor, results)
            for p in [output_path, output_path.replace('llm_results_', 'llm_records_')]:
                if Path(p).exists():
                    Path(p).unlink()
            logger.info('LLM 청크 결과 적재 완료', count=len(results), path=output_path)
            return len(results)
        except Exception as e:
            logger.error('LLM 청크 결과 적재 실패', error=str(e), exc_info=True)
            raise AirflowException(f'LLM 청크 결과 적재 실패: {e}') from e

    @task
    def fetch_failures(run_id: str, dag: DAG) -> list[str]:
        """failure 후보 조회 → 청크별 /tmp 파일 저장. 대상 없으면 [] 반환."""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = LLMPipeline(logical_date=pendulum.now('Asia/Seoul'))
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                records = pipeline.fetch_failure_candidates(cursor)
            if not records:
                logger.info('failure 대상 없음, 스킵')
                return []
            paths = []
            for i, start in enumerate(range(0, len(records), CHUNK_SIZE)):
                chunk = records[start:start + CHUNK_SIZE]
                path = f'/tmp/failure_records_{run_id}_chunk{i}.json'
                Path(path).write_text(json.dumps(chunk, ensure_ascii=False))
                paths.append(path)
            logger.info('failure 후보 조회 완료', total=len(records), chunks=len(paths))
            return paths
        except Exception as e:
            logger.error('failure 후보 조회 실패', error=str(e), exc_info=True)
            raise AirflowException(f'failure 후보 조회 실패: {e}') from e

    @task.external_python(python=VLLM_PYTHON, expect_airflow=False, max_active_tis_per_dagrun=1)
    def failure_extract_chunk(input_path: str, logical_date: str) -> str:
        """vllm-env에서 청크 단위 failure 모델 배치 추출."""
        import json
        import re
        from pathlib import Path
        import pendulum
        from maude_early_alert.pipelines.llm_pipeline import LLMPipeline

        m = re.search(r'_chunk(\d+)\.json$', input_path)
        chunk_idx = m.group(1) if m else '0'
        records = json.loads(Path(input_path).read_text())
        output_path = input_path.replace('failure_records_', 'failure_results_')
        pipeline = LLMPipeline(logical_date=pendulum.parse(logical_date))
        results = pipeline.run_failure_batch(records, chunk_idx=chunk_idx)
        Path(output_path).write_text(json.dumps(results, ensure_ascii=False, default=str))
        return output_path

    @task
    def load_failure_chunk(output_path: str, run_id: str, dag: DAG) -> int:
        """failure 청크 결과 파일 → Snowflake 적재 후 임시 파일 삭제"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = LLMPipeline(logical_date=pendulum.now('Asia/Seoul'))
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            results = json.loads(Path(output_path).read_text())
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.load_extraction_results(cursor, results)
            for p in [output_path, output_path.replace('failure_results_', 'failure_records_')]:
                if Path(p).exists():
                    Path(p).unlink()
            logger.info('failure 청크 결과 적재 완료', count=len(results), path=output_path)
            return len(results)
        except Exception as e:
            logger.error('failure 청크 결과 적재 실패', error=str(e), exc_info=True)
            raise AirflowException(f'failure 청크 결과 적재 실패: {e}') from e

    @task(outlets=[MAUDE_LLM_ASSET])
    def join_extraction(run_id: str, dag: DAG) -> None:
        """추출 결과 JOIN → {category}_LLM_EXTRACTED 테이블 생성"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = LLMPipeline(logical_date=pendulum.now('Asia/Seoul'))
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.join_extraction(cursor)
            logger.info('LLM 추출 JOIN 완료')
        except Exception as e:
            logger.error('JOIN 실패', error=str(e), exc_info=True)
            raise AirflowException(f'JOIN 실패: {e}') from e

    @task(trigger_rule='all_done')
    def cleanup_checkpoint() -> None:
        """체크포인트 삭제 (성공/실패 무관하게 항상 실행)"""
        LLMPipeline(logical_date=pendulum.now('Asia/Seoul')).cleanup_extraction_checkpoint()

    # ── 1차 추출: Dynamic Task Mapping ──────────────────────────────────────
    chunk_paths = extract_records()
    results_paths = llm_extract_chunk.partial(
        logical_date=pendulum.now('Asia/Seoul').isoformat(),
    ).expand(input_path=chunk_paths)
    loaded = load_chunk_results.expand(output_path=results_paths)

    # ── failure retry: Dynamic Task Mapping ─────────────────────────────────
    # loaded 전체 완료 후 failure 후보 조회
    failure_paths = fetch_failures()
    failure_results = failure_extract_chunk.partial(
        logical_date=pendulum.now('Asia/Seoul').isoformat(),
    ).expand(input_path=failure_paths)
    failure_loaded = load_failure_chunk.expand(output_path=failure_results)

    p_join = join_extraction()
    p_cleanup = cleanup_checkpoint()

    loaded >> failure_paths
    failure_loaded >> p_join >> p_cleanup


maude_llm()

from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
import pendulum
import structlog
from structlog.contextvars import bind_contextvars

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.config import get_config
from maude_early_alert.assets import MAUDE_LLM_ASSET, MAUDE_CLUSTERED_ASSET

configure_logging(level='INFO', log_file='cluster.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_cfg = get_config().silver
CLUSTER_PYTHON = _cfg.get_clustering_vllm_python()


@dag(
    dag_id='maude_cluster',
    schedule=[MAUDE_LLM_ASSET],
    start_date=pendulum.datetime(2026, 1, 1, tz='Asia/Seoul'),
    catchup=False,
    max_active_runs=1,
    tags=['maude', 'clustering'],
    default_args={
        'retries': 1,
        'retry_delay': pendulum.duration(minutes=5),
    },
)
def maude_cluster():
    """MAUDE Clustering 파이프라인
    1. EVENT_LLM_EXTRACTED에서 데이터 로드
    2. vocab filter → embed
    3. 저장된 모델로 전체 클러스터링 예측 (추론 전용)
    4. clustering 결과 JOIN → {category}_CLUSTERED 테이블 생성
    """

    @task.external_python(python=CLUSTER_PYTHON, expect_airflow=False, max_active_tis_per_dagrun=1)
    def run_clustering_external(logical_date: str) -> dict:
        """외부 venv에서 fetch -> embed -> predict -> join 전체 실행."""
        from contextlib import closing

        import pendulum
        import snowflake.connector

        from maude_early_alert.pipelines.cluster_pipeline import ClusterPipeline
        from maude_early_alert.utils.secrets import get_secret

        pipeline = ClusterPipeline(logical_date=pendulum.parse(logical_date))
        secret = get_secret('snowflake/de')
        model_dir = pipeline.cfg.get_clustering_inference_model_dir()

        with closing(
            snowflake.connector.connect(
                user=secret['user'],
                password=secret['password'],
                account=secret['account'],
                warehouse=secret['warehouse'],
            )
        ) as conn, closing(conn.cursor()) as cursor:
            df = pipeline.fetch_clustering_data(cursor)
            embeddings, df = pipeline.prepare_clustering_embeddings(df)
            labels, metadata = pipeline.run_clustering_prediction(df, embeddings)
            pipeline.join_clustering_results(cursor, df, labels)

        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        noise_ratio = float((labels == -1).mean())
        return {
            'n_clusters': n_clusters,
            'noise_ratio': noise_ratio,
            'model_dir': model_dir,
        }

    @task(outlets=[MAUDE_CLUSTERED_ASSET])
    def finalize(metrics: dict, run_id: str, dag: DAG) -> None:
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        try:
            logger.info(
                'clustering 완료',
                n_clusters=metrics['n_clusters'],
                noise_ratio=f"{metrics['noise_ratio']:.2%}",
                model_dir=metrics.get('model_dir'),
            )
        except Exception as e:
            logger.error('clustering 실패', error=str(e), exc_info=True)
            raise AirflowException(f'clustering 실패: {e}') from e

    logical_date = pendulum.now('Asia/Seoul').isoformat()
    metrics = run_clustering_external(logical_date=logical_date)
    finalize(metrics)


maude_cluster()

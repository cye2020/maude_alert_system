from airflow.sdk import dag, task, DAG
from airflow.exceptions import AirflowException
from airflow.providers.snowflake.hooks.snowflake import SnowflakeHook
import pendulum
import structlog
from contextlib import closing
from structlog.contextvars import bind_contextvars

from maude_early_alert.logging_config import configure_logging
from maude_early_alert.pipelines.silver import SilverPipeline
from maude_early_alert.assets import MAUDE_LLM_ASSET, MAUDE_CLUSTERED_ASSET

configure_logging(level='INFO', log_file='cluster.log')

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

SNOWFLAKE_CONN_ID = 'snowflake_default'


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
    2. vocab filter → embed → (train.enabled=true) Optuna 튜닝 + 베스트 모델 저장
    3. 저장된 모델로 전체 클러스터링 예측
    4. clustering 결과 JOIN → {category}_CLUSTERED 테이블 생성
    """

    @task(outlets=[MAUDE_CLUSTERED_ASSET])
    def clustering(logical_date: pendulum.DateTime, run_id: str, dag: DAG) -> None:
        """fetch → tune → predict → join_clustering_results"""
        bind_contextvars(dag_id=dag.dag_id, run_id=run_id)
        pipeline = SilverPipeline(stage={'event': 0, 'udi': 0}, logical_date=logical_date)
        hook = SnowflakeHook(snowflake_conn_id=SNOWFLAKE_CONN_ID)
        try:
            # 1단계: 데이터 로드 (EVENT_LLM_EXTRACTED → DataFrame)
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                df = pipeline.fetch_clustering_data(cursor)
            logger.info('clustering 데이터 로드 완료', rows=len(df))

            # 2-5단계: vocab filter → 임베딩 → (train.enabled=true) Optuna 튜닝 + 저장
            embeddings, df = pipeline.run_clustering_tuning(df)

            # 6단계: 저장된 베스트 모델로 전체 클러스터링 예측
            labels, metadata = pipeline.run_clustering_prediction(df, embeddings)
            n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
            logger.info(
                'clustering 예측 완료',
                n_clusters=n_clusters,
                noise_ratio=f'{float((labels == -1).mean()):.2%}',
            )

            # 7단계: 결과 JOIN → _CLUSTERED 테이블 생성
            with hook.get_conn() as conn, closing(conn.cursor()) as cursor:
                pipeline.join_clustering_results(cursor, df, labels)
            logger.info('clustering 완료', n_clusters=n_clusters)
        except Exception as e:
            logger.error('clustering 실패', error=str(e), exc_info=True)
            raise AirflowException(f'clustering 실패: {e}') from e

    clustering()


maude_cluster()

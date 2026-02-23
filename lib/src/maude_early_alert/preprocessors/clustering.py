# --------------------------------------------
# 표준 라이브러리
# --------------------------------------------
import ast
import json

# --------------------------------------------
# 서드파티 라이브러리
# --------------------------------------------
import numpy as np
import pandas as pd
import snowflake.connector
import structlog
from sentence_transformers import SentenceTransformer
from snowflake.connector.pandas_tools import write_pandas

# --------------------------------------------
# GPU / CPU 자동 전환
# --------------------------------------------
try:
    import cupy as cp
    from cuml import UMAP
    from cuml.cluster import HDBSCAN
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
    _GPU = True
except ImportError:
    cp = None
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.metrics import silhouette_score as cython_silhouette_score
    _GPU = False

# --------------------------------------------
# 내부 라이브러리
# --------------------------------------------
from maude_early_alert.logging_config import configure_logging
from maude_early_alert.utils.secrets import get_secret


logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)
logger.info("클러스터링 백엔드", gpu=_GPU)


def _to_numpy(arr) -> np.ndarray:
    """cupy / numpy 배열을 numpy로 통일"""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


class Clustering:

    def __init__(self):
        self.embedding_model = "pritamdeka/S-PubMedBert-MS-MARCO"
        # ----- UMAP 파라미터 ----------
        self.umap_params = dict(
            n_neighbors  = 50,
            min_dist     = 0.0,
            n_components = 15,
            metric       = "cosine",
            random_state = 42,
        )
        # ----- 임베딩 파라미터 --------
        self.batch_size = 32
        self.normalize  = True
        # ----- 그리드 서치 파라미터 ---
        self.mcs_grid   = (50, 80, 100, 150, 200, 300)
        self.ms_grid    = (3, 5, 7, 9)
        self.target_k   = 100   # 클러스터 개수 상한

    def build_mdr_sntc(self, df: pd.DataFrame) -> pd.DataFrame:
        """LLM추출 4개 컬럼을 임베딩용으로 재생성"""

        def parse_components(val) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "unknown"
            if isinstance(val, list):
                return ", ".join(val)
            try:
                parsed = ast.literal_eval(str(val))
                return ", ".join(parsed) if isinstance(parsed, list) else str(parsed)
            except (ValueError, SyntaxError):
                return str(val)

        def fill_unknown(val) -> str:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return "unknown"
            return str(val)

        df = df.copy()
        df["MDR_SNTC"] = (
            df["PATIENT_HARM"].apply(fill_unknown)             + ". "
            + df["PROBLEM_COMPONENTS"].apply(parse_components) + ". "
            + df["DEFECT_CONFIRMED"].apply(fill_unknown)       + ". "
            + df["DEFECT_TYPE"].apply(fill_unknown)
        )
        return df

    def embed_texts(self, texts: list) -> np.ndarray:
        """768D 임베딩 생성"""
        logger.info("임베딩 시작", model=self.embedding_model, n=len(texts))
        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(
            texts,
            batch_size           = self.batch_size,
            show_progress_bar    = True,
            convert_to_numpy     = True,
            normalize_embeddings = self.normalize,
        )
        logger.info("임베딩 완료", shape=str(embeddings.shape))
        return embeddings

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        """UMAP 차원 축소 (GPU 있으면 cuML, 없으면 umap-learn)"""
        logger.info("UMAP 시작", backend="cuML" if _GPU else "CPU", params=self.umap_params)
        umap_model = UMAP(**self.umap_params)
        X = umap_model.fit_transform(embeddings)
        X_np = _to_numpy(X).astype(np.float32)
        logger.info("UMAP 완료", input=str(embeddings.shape), output=str(X_np.shape))
        return X_np

    def _fit_hdbscan(self, X: np.ndarray, mcs: int, ms: int) -> np.ndarray:
        """단일 파라미터 조합으로 HDBSCAN 실행 후 numpy 레이블 반환"""
        if _GPU:
            X_in = cp.asarray(X, dtype=cp.float32)
        else:
            X_in = X

        clusterer = HDBSCAN(
            min_cluster_size         = mcs,
            min_samples              = ms,
            metric                   = "euclidean",
            cluster_selection_method = "eom",
        )
        labels = clusterer.fit_predict(X_in)
        return _to_numpy(labels)

    def run_clustering(self, X: np.ndarray) -> np.ndarray:
        """
        HDBSCAN 그리드 서치로 최적 파라미터 자동 선택.

        - target_k 이하의 클러스터 수 중 Silhouette Score 최고인 조합 선택
        - mcs_grid / ms_grid / target_k 는 __init__ 에서 조정 가능
        """
        logger.info("HDBSCAN 그리드 서치 시작",
                    mcs_grid=self.mcs_grid,
                    ms_grid=self.ms_grid,
                    target_k=self.target_k)

        best: dict | None = None

        for mcs in self.mcs_grid:
            for ms in self.ms_grid:
                labels = self._fit_hdbscan(X, mcs, ms)

                k           = len(set(labels)) - (1 if -1 in labels else 0)
                noise_ratio = float((labels == -1).mean())

                if k < 2:
                    logger.debug("스킵 (클러스터 부족)",
                                 mcs=mcs, ms=ms, k=k, noise=round(noise_ratio, 3))
                    continue

                mask = labels != -1
                sil  = float(cython_silhouette_score(X[mask], labels[mask]))

                logger.info("후보",
                            mcs=mcs, ms=ms,
                            k=k, noise=round(noise_ratio, 3),
                            silhouette=round(sil, 3))

                if k <= self.target_k:
                    if best is None or sil > best["silhouette"]:
                        best = dict(mcs=mcs, ms=ms, k=k,
                                    noise=noise_ratio, silhouette=sil,
                                    labels=labels)

        if best is None:
            raise RuntimeError("target_k 이하의 유효한 클러스터 조합을 찾지 못했습니다. "
                               "mcs_grid 또는 target_k 를 조정하세요.")

        logger.info("최적 파라미터 선택",
                    mcs=best["mcs"], ms=best["ms"],
                    k=best["k"],
                    noise=round(best["noise"], 3),
                    silhouette=round(best["silhouette"], 3))
        return best["labels"]


if __name__ == "__main__":
    configure_logging(level="INFO")

    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret)
    cursor = conn.cursor()

    try:
        # ── 1. 데이터 로드 ─────────────────────────────────
        logger.info("Silver 데이터 로드 시작")
        cursor.execute("""
            SELECT
                MDR_REPORT_KEY,
                PRODUCT_CODE,
                EVENT_TYPE,
                PATIENT_HARM,
                PROBLEM_COMPONENTS,
                DEFECT_CONFIRMED,
                DEFECT_TYPE
            FROM MAUDE.SILVER.EVENT_STAGE_12_COMBINED
        """)
        df = cursor.fetch_pandas_all()
        logger.info("Silver 데이터 로드 완료", rows=len(df))

        # ── 2~5. ML 처리 ───────────────────────────────────
        c          = Clustering()
        df         = c.build_mdr_sntc(df)
        embeddings = c.embed_texts(df["MDR_SNTC"].tolist())
        X          = c.reduce_dimensions(embeddings)
        labels     = c.run_clustering(X)

        # ── 6. 결과 DataFrame 구성 ─────────────────────────
        df_result = df[[
            "MDR_REPORT_KEY", "PRODUCT_CODE", "EVENT_TYPE",
            "PATIENT_HARM", "PROBLEM_COMPONENTS",
            "DEFECT_CONFIRMED", "DEFECT_TYPE", "MDR_SNTC",
        ]].copy()
        df_result["CLUSTER_ID"] = labels
        df_result["PROBLEM_COMPONENTS"] = df_result["PROBLEM_COMPONENTS"].apply(
            lambda v: json.dumps(v) if isinstance(v, list) else str(v) if v else None
        )

        # ── 7. 시각화 (Plotly 인터랙티브) ─────────────────────
        import plotly.express as px

        logger.info("시각화용 2D UMAP 시작")
        umap_2d   = UMAP(
            n_neighbors  = 50,
            min_dist     = 0.3,
            n_components = 2,
            metric       = "cosine",
            random_state = 42,
        )
        coords_2d = _to_numpy(umap_2d.fit_transform(embeddings)).astype(np.float32)

        df_plot = pd.DataFrame({
            "x":       coords_2d[:, 0],
            "y":       coords_2d[:, 1],
            "cluster": labels.astype(str),
            "product": df["PRODUCT_CODE"].values,
            "harm":    df["PATIENT_HARM"].values,
            "text":    df["MDR_SNTC"].str[:100].values,
        })

        fig = px.scatter(
            df_plot,
            x          = "x",
            y          = "y",
            color      = "cluster",
            hover_data = ["product", "harm", "text"],
            title      = "MAUDE MDR 클러스터 분포 (UMAP 2D)",
            width      = 1200,
            height     = 800,
        )
        fig.write_html("cluster_plot.html")
        logger.info("시각화 저장 완료", path="cluster_plot.html")

        # ── 8. Snowflake 적재 ──────────────────────────────
        # logger.info("결과 적재 시작")
        # cursor.execute("""
        #     CREATE TABLE IF NOT EXISTS MAUDE.SILVER.MAUDE_CLUSTERED (
        #         MDR_REPORT_KEY     VARCHAR,
        #         PRODUCT_CODE       VARCHAR,
        #         EVENT_TYPE         VARCHAR,
        #         PATIENT_HARM       VARCHAR,
        #         PROBLEM_COMPONENTS VARCHAR,
        #         DEFECT_CONFIRMED   VARCHAR,
        #         DEFECT_TYPE        VARCHAR,
        #         MDR_SNTC           VARCHAR,
        #         CLUSTER_ID         NUMBER
        #     )
        # """)
        # cursor.execute("TRUNCATE TABLE MAUDE.SILVER.MAUDE_CLUSTERED")
        # write_pandas(
        #     conn       = conn,
        #     df         = df_result,
        #     table_name = "MAUDE_CLUSTERED",
        #     database   = "MAUDE",
        #     schema     = "SILVER",
        # )
        # logger.info("결과 적재 완료")

    finally:
        cursor.close()
        conn.close()

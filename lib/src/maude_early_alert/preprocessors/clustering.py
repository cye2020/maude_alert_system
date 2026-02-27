# --------------------------------------------
# 표준 라이브러리
# --------------------------------------------
import ast
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------
# 서드파티 라이브러리
# --------------------------------------------
import joblib
import numpy as np
import pandas as pd
import snowflake.connector
import structlog
from sentence_transformers import SentenceTransformer
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score

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


def _to_numpy(arr) -> np.ndarray:
    """cupy / numpy 배열을 numpy로 통일"""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# ------------------------------------------------------------------
# 1. 임베딩용 텍스트 생성
# ------------------------------------------------------------------

def build_mdr_sntc(df: pd.DataFrame) -> pd.DataFrame:
    """
    LLM 추출 4개 컬럼을 임베딩용 텍스트(MDR_SNTC)로 결합.

    - PATIENT_HARM, DEFECT_CONFIRMED, DEFECT_TYPE : null → "unknown"
    - PROBLEM_COMPONENTS                          : list / str 파싱 후 ", " 결합
    """
    def _parse_components(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "unknown"
        if isinstance(val, list):
            return ", ".join(val)
        try:
            parsed = ast.literal_eval(str(val))
            return ", ".join(parsed) if isinstance(parsed, list) else str(parsed)
        except (ValueError, SyntaxError):
            return str(val)

    def _fill_unknown(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "unknown"
        return str(val)

    df = df.copy()
    df["MDR_SNTC"] = (
        df["PATIENT_HARM"].apply(_fill_unknown)             + ". "
        + df["PROBLEM_COMPONENTS"].apply(_parse_components) + ". "
        + df["DEFECT_CONFIRMED"].apply(_fill_unknown)       + ". "
        + df["DEFECT_TYPE"].apply(_fill_unknown)
    )
    logger.info("MDR_SNTC 생성 완료", rows=len(df))
    return df


# ------------------------------------------------------------------
# 2. 텍스트 임베딩
# ------------------------------------------------------------------

def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """S-PubMedBert 768D 임베딩 생성."""
    logger.info("임베딩 시작", n=len(texts))
    embeddings = model.encode(
        texts,
        batch_size           = batch_size,
        show_progress_bar    = True,
        convert_to_numpy     = True,
        normalize_embeddings = normalize,
    )
    logger.info("임베딩 완료", shape=str(embeddings.shape))
    return embeddings


# ------------------------------------------------------------------
# 3. 차원 축소
# ------------------------------------------------------------------

def reduce_dimensions(
    embeddings : np.ndarray,
    umap_params: Optional[dict] = None,
) -> np.ndarray:
    """
    UMAP 차원 축소 (GPU 있으면 cuML, 없으면 umap-learn).

    umap_params 기본값: n_neighbors=50, min_dist=0.0, n_components=15, metric='cosine'
    """
    params = umap_params or dict(
        n_neighbors  = 50,
        min_dist     = 0.0,
        n_components = 15,
        metric       = "cosine",
        random_state = 42,
    )
    logger.info("UMAP 시작", backend="cuML" if _GPU else "CPU", params=params)
    umap_model = UMAP(**params)
    X_np = _to_numpy(umap_model.fit_transform(embeddings)).astype(np.float32)
    logger.info("UMAP 완료", input=str(embeddings.shape), output=str(X_np.shape))
    return X_np


# ------------------------------------------------------------------
# 4. 클러스터링
# ------------------------------------------------------------------

def _fit_hdbscan(X: np.ndarray, mcs: int, ms: int):
    """단일 파라미터 조합으로 HDBSCAN 실행 후 (labels_np, model) 반환"""
    X_in = cp.asarray(X, dtype=cp.float32) if _GPU else X
    clusterer = HDBSCAN(
        min_cluster_size         = mcs,
        min_samples              = ms,
        metric                   = "euclidean",
        cluster_selection_method = "eom",
    )
    labels = clusterer.fit_predict(X_in)
    return _to_numpy(labels), clusterer


def run_clustering(
    X        : np.ndarray,
    mcs_grid : tuple = (50, 80, 100, 150, 200, 300),
    ms_grid  : tuple = (3, 5, 7, 9),
    target_k : Optional[int] = None,
) -> np.ndarray:
    """
    HDBSCAN 그리드 서치로 최적 파라미터 자동 선택.

    target_k 이하의 클러스터 수 중 Silhouette Score 최고인 조합을 반환합니다.
    """
    n        = len(X)
    target_k = target_k if target_k is not None else max(2, n // min(mcs_grid))

    oversized = [mcs for mcs in mcs_grid if mcs > n // 2]
    if oversized:
        logger.warning(
            "mcs_grid 일부 값이 데이터의 절반을 초과합니다 — 해당 값은 클러스터를 생성하지 못할 수 있습니다",
            n=n, oversized=oversized,
        )

    logger.info(
        "HDBSCAN 그리드 서치 시작",
        n        = n,
        mcs_grid = mcs_grid,
        ms_grid  = ms_grid,
        target_k = target_k,
    )

    best: Optional[dict] = None

    for mcs in mcs_grid:
        for ms in ms_grid:
            labels, model = _fit_hdbscan(X, mcs, ms)

            k           = len(set(labels)) - (1 if -1 in labels else 0)
            noise_ratio = float((labels == -1).mean())

            if k < 2:
                logger.debug("스킵 (클러스터 부족)", mcs=mcs, ms=ms, k=k, noise=round(noise_ratio, 3))
                continue

            mask    = labels != -1
            n_valid = int(mask.sum())

            if n_valid < 2:
                logger.debug("스킵 (유효 샘플 부족)", mcs=mcs, ms=ms, k=k, n_valid=n_valid)
                continue

            sil = float(cython_silhouette_score(X[mask], labels[mask]))

            logger.info("후보", mcs=mcs, ms=ms, k=k, noise=round(noise_ratio, 3), silhouette=round(sil, 3))

            if k <= target_k and (best is None or sil > best["silhouette"]):
                best = dict(mcs=mcs, ms=ms, k=k, noise=noise_ratio, silhouette=sil, labels=labels)

    if best is None:
        raise RuntimeError(
            "target_k 이하의 유효한 클러스터 조합을 찾지 못했습니다. "
            "mcs_grid 또는 target_k 를 조정하세요."
        )

    logger.info(
        "최적 파라미터 선택",
        mcs       = best["mcs"],
        ms        = best["ms"],
        k         = best["k"],
        noise     = round(best["noise"], 3),
        silhouette= round(best["silhouette"], 3),
    )
    return best["labels"]


# ------------------------------------------------------------------
# 5. 평가
# ------------------------------------------------------------------

def evaluate(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    클러스터링 품질 평가.

    Returns
    -------
    dict
        silhouette, davies_bouldin, calinski_harabasz 지표.
        유효 클러스터가 없으면 빈 dict 반환.
    """
    mask    = labels != -1
    n_valid = int(mask.sum())

    if len(set(labels[mask])) <= 1 or n_valid < 2:
        logger.warning("클러스터 평가 불가 - 유효 클러스터 부족",
                       n_clusters=len(set(labels[mask])), n_valid=n_valid)
        return {}

    X_valid      = X[mask]
    labels_valid = labels[mask]

    metrics = {
        "silhouette"       : round(float(cython_silhouette_score(X_valid, labels_valid)), 4),
        "davies_bouldin"   : round(float(davies_bouldin_score(X_valid, labels_valid)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X_valid, labels_valid)), 2),
    }
    logger.info("클러스터링 평가 완료", **metrics)
    return metrics


# ------------------------------------------------------------------
# 6. 대표 샘플 추출
# ------------------------------------------------------------------

def get_representatives(
    df    : pd.DataFrame,
    X     : np.ndarray,
    labels: np.ndarray,
    n     : int = 5,
) -> Dict[int, pd.DataFrame]:
    """
    Centroid 기반 클러스터별 대표 샘플 추출.
    Noise(-1) 클러스터는 제외합니다.

    Returns
    -------
    dict
        {cluster_id: pd.DataFrame} 형태.
    """
    reps   = {}
    df_idx = df.reset_index(drop=True)

    for cid in np.unique(labels):
        if cid == -1:
            continue

        idx      = np.where(labels == cid)[0]
        centroid = X[idx].mean(axis=0)
        dists    = np.linalg.norm(X[idx] - centroid, axis=1)
        top_idx  = idx[np.argsort(dists)[:n]]

        reps[int(cid)] = df_idx.iloc[top_idx].copy()

    logger.info("대표 샘플 추출 완료", n_clusters=len(reps), n_per_cluster=n)
    return reps


# ------------------------------------------------------------------
# 7. 클러스터별 범주형 분석
# ------------------------------------------------------------------

def analyze_by_column(
    df         : pd.DataFrame,
    col        : str,
    cluster_col: str = "CLUSTER_ID",
    top_n      : Optional[int] = None,
) -> pd.DataFrame:
    """
    클러스터별 범주형 컬럼 빈도 교차표 반환.

    Returns
    -------
    pd.DataFrame
        행=클러스터, 열=범주 의 빈도 교차표
    """
    if top_n is not None:
        top_cats = df[col].value_counts().head(top_n).index
        df       = df[df[col].isin(top_cats)]

    crosstab = pd.crosstab(df[cluster_col], df[col])
    logger.info("범주별 분석 완료", col=col, n_clusters=len(crosstab), n_categories=len(crosstab.columns))
    return crosstab


def analyze_keywords(
    df         : pd.DataFrame,
    col        : str = "PROBLEM_COMPONENTS",
    cluster_col: str = "CLUSTER_ID",
    top_n      : int = 10,
) -> Dict[int, Counter]:
    """
    클러스터별 리스트형 컬럼의 키워드 빈도 분석.
    Noise(-1) 클러스터는 제외합니다.

    Returns
    -------
    dict
        {cluster_id: Counter} 형태.
    """
    result: Dict[int, Counter] = {}

    for cid, group in df.groupby(cluster_col):
        if cid == -1:
            continue

        keyword_list: List[str] = []
        for val in group[col]:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if isinstance(val, list):
                keyword_list.extend(val)
            else:
                try:
                    parsed = ast.literal_eval(str(val))
                    if isinstance(parsed, list):
                        keyword_list.extend(parsed)
                    else:
                        keyword_list.append(str(parsed))
                except (ValueError, SyntaxError):
                    keyword_list.append(str(val))

        counts = Counter(kw.strip().lower() for kw in keyword_list if kw.strip())
        result[int(cid)] = counts
        logger.debug("클러스터 키워드 Top", cluster=cid, top=counts.most_common(top_n))

    logger.info("키워드 분석 완료", n_clusters=len(result))
    return result


# ============================================================================
# 실행 진입점
# ============================================================================

if __name__ == "__main__":
    import plotly.express as px

    configure_logging(level="INFO")

    # ── 0. Snowflake 연결 ──────────────────────────────────────────────────
    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret, ocsp_fail_open=True)
    cursor = conn.cursor()

    try:
        # ── 1. 데이터 로드 ─────────────────────────────────────────────────
        logger.info("Silver 데이터 로드 시작")
        cursor.execute("""
            SELECT
                MDR_REPORT_KEY,
                MANUFACTURER_NAME,
                PRODUCT_CODE,
                EVENT_TYPE,
                DATE_RECEIVED,
                PATIENT_HARM,
                PROBLEM_COMPONENTS,
                DEFECT_CONFIRMED,
                DEFECT_TYPE
            FROM MAUDE.SILVER.EVENT_STAGE_12_COMBINED
        """)
        df = cursor.fetch_pandas_all()
        logger.info("Silver 데이터 로드 완료", rows=len(df))

        # ── 2. mcs_grid 자동 계산 ──────────────────────────────────────────
        # mcs 범위: 데이터의 약 0.5% ~ 5% (HDBSCAN 권장 경험치)
        n_rows   = len(df)
        mcs_min  = max(50, n_rows // 200)   # ~0.5%
        mcs_step = max(50, n_rows // 100)   # ~1% 간격
        mcs_grid = tuple(mcs_min + i * mcs_step for i in range(6))
        logger.info("mcs_grid 자동 계산", n_rows=n_rows, mcs_grid=mcs_grid)

        # ── 3. 클러스터링 파이프라인 실행 ──────────────────────────────────
        logger.info("SentenceTransformer 로딩 중", gpu=_GPU)
        model = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")

        df         = build_mdr_sntc(df)
        embeddings = embed_texts(model, df["MDR_SNTC"].tolist())
        X          = reduce_dimensions(embeddings)
        labels     = run_clustering(X, mcs_grid=mcs_grid, ms_grid=(3, 5, 7, 9), target_k=10)

        # ── 4. 평가 ────────────────────────────────────────────────────────
        metrics = evaluate(X, labels)
        logger.info("평가 결과", **metrics)

        # ── 5. 결과 DataFrame 구성 ─────────────────────────────────────────
        df_result = df[[
            "MDR_REPORT_KEY", "MANUFACTURER_NAME", "PRODUCT_CODE", "EVENT_TYPE",
            "DATE_RECEIVED",
            "PATIENT_HARM", "PROBLEM_COMPONENTS",
            "DEFECT_CONFIRMED", "DEFECT_TYPE", "MDR_SNTC",
        ]].copy()
        df_result["CLUSTER_ID"] = labels
        df_result["PROBLEM_COMPONENTS"] = df_result["PROBLEM_COMPONENTS"].apply(
            lambda v: json.dumps(v) if isinstance(v, list) else str(v) if v else None
        )

        # ── 6. 중간 결과 저장 (필요 시) ────────────────────────────────────
        # output_dir = Path("./output")
        # output_dir.mkdir(parents=True, exist_ok=True)
        # np.save(output_dir / "embeddings.npy", embeddings)
        # np.save(output_dir / "umap_X.npy", X)
        # np.save(output_dir / "cluster_labels.npy", labels)

        # ── 7. 대표 샘플 확인 ──────────────────────────────────────────────
        reps = get_representatives(df_result, X, labels, n=5)
        for cid, rep_df in reps.items():
            logger.info("대표 샘플", cluster=cid, rows=len(rep_df))

        # ── 8. 클러스터별 분석 ─────────────────────────────────────────────
        event_type_ct = analyze_by_column(df_result, "EVENT_TYPE")
        logger.info("이벤트 유형 교차표", shape=str(event_type_ct.shape))

        keywords = analyze_keywords(df_result, col="PROBLEM_COMPONENTS")

        # ── 9. 시각화 (UMAP 2D) ────────────────────────────────────────────
        logger.info("시각화용 UMAP 2D 시작")
        coords_2d = _to_numpy(
            UMAP(n_neighbors=50, min_dist=0.3, n_components=2, metric="cosine", random_state=42)
            .fit_transform(embeddings)
        ).astype(np.float32)

        fig = px.scatter(
            pd.DataFrame({
                "x"      : coords_2d[:, 0],
                "y"      : coords_2d[:, 1],
                "cluster": labels.astype(str),
                "product": df["PRODUCT_CODE"].values,
                "harm"   : df["PATIENT_HARM"].values,
                "text"   : df["MDR_SNTC"].str[:100].values,
            }),
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
        # cluster 확인 → 터미널에 start cluster_plot.html

        # ── 10. Snowflake 적재 ─────────────────────────────────────────────
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MAUDE.SILVER.EVENT_CLUSTERED (
                MDR_REPORT_KEY     VARCHAR,
                MANUFACTURER_NAME  VARCHAR,
                PRODUCT_CODE       VARCHAR,
                EVENT_TYPE         VARCHAR,
                DATE_RECEIVED      DATE,
                PATIENT_HARM       VARCHAR,
                PROBLEM_COMPONENTS VARCHAR,
                DEFECT_CONFIRMED   VARCHAR,
                DEFECT_TYPE        VARCHAR,
                MDR_SNTC           VARCHAR,
                CLUSTER_ID         NUMBER
            )
        """)
        cursor.execute("TRUNCATE TABLE MAUDE.SILVER.EVENT_CLUSTERED")
        rows = [tuple(r) for r in df_result.itertuples(index=False, name=None)]
        cursor.executemany(
            """
            INSERT INTO MAUDE.SILVER.EVENT_CLUSTERED (
                MDR_REPORT_KEY, MANUFACTURER_NAME, PRODUCT_CODE, EVENT_TYPE,
                DATE_RECEIVED, PATIENT_HARM, PROBLEM_COMPONENTS, DEFECT_CONFIRMED,
                DEFECT_TYPE, MDR_SNTC, CLUSTER_ID
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            rows,
        )
        logger.info("결과 적재 완료", rows=len(df_result))

    finally:
        cursor.close()
        conn.close()

# --------------------------------------------
# 표준 라이브러리
# --------------------------------------------
import ast
import json
import joblib
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# --------------------------------------------
# 서드파티 라이브러리
# --------------------------------------------
import numpy as np
import pandas as pd
import snowflake.connector
import structlog
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sentence_transformers import SentenceTransformer

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
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


# =====================
# Step 2. 텍스트 전처리
# =====================

def analyze_keywords(
    df: pd.DataFrame,
    text_col: str,
    min_freq: int,
) -> Set[str]:
    """
    list 형태 텍스트 컬럼의 키워드 빈도를 집계하고 min_freq 이상인 집합 반환.
    반환값을 prepare_text_col()의 vocab 인자로 전달.
    """
    counts: Counter = Counter()
    for val in df[text_col]:
        if isinstance(val, list):
            counts.update(val)
        else:
            try:
                items = ast.literal_eval(str(val))
                if isinstance(items, list):
                    counts.update(items)
            except (ValueError, SyntaxError):
                pass

    vocab = {k for k, v in counts.items() if v >= min_freq}
    logger.info("어휘 필터링", total_keywords=len(counts), kept=len(vocab), min_freq=min_freq)
    return vocab


def prepare_text_col(
    df: pd.DataFrame,
    text_col: str,
    output_col: str,
    vocab: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    list 형태 텍스트 컬럼을 필터링 후 쉼표 구분 문자열로 변환.

    Args:
        vocab: analyze_keywords() 반환값. None이면 필터링 없음.
    """
    def to_str(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "unknown"
        if isinstance(val, list):
            items = val
        else:
            try:
                items = ast.literal_eval(str(val))
                if not isinstance(items, list):
                    return str(val)
            except (ValueError, SyntaxError):
                return str(val)
        if vocab is not None:
            items = [item for item in items if str(item).lower() in vocab]
        return ", ".join(str(item) for item in items) if items else "unknown"

    df = df.copy()
    df[output_col] = df[text_col].apply(to_str)
    return df


# =====================
# Step 3. 임베딩
# =====================

def embed_texts(
    texts: List[str],
    model: str,
    batch_size: int,
    normalize: bool,
) -> np.ndarray:
    """SentenceTransformer로 임베딩 생성."""
    logger.info("임베딩 시작", model=model, n=len(texts))
    st_model = SentenceTransformer(model)
    embeddings = st_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    logger.info("임베딩 완료", shape=str(embeddings.shape))
    return embeddings


# =====================
# Step 4. 피처 준비
# =====================

def prepare_features(
    embeddings: np.ndarray,
    categorical_cols: Optional[List[str]],
    onehot_weight: float,
    df: Optional[pd.DataFrame] = None,
    scaler: Optional[StandardScaler] = None,
    encoder: Optional[OneHotEncoder] = None,
    fit: bool = True,
) -> Tuple[np.ndarray, StandardScaler, Optional[OneHotEncoder]]:
    """
    임베딩 스케일링 + 범주형 원핫 인코딩 결합.

    Args:
        fit: True면 scaler/encoder 학습, False면 transform만 (추론 시)

    Returns:
        (features, scaler, encoder)
    """
    if scaler is None:
        scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings) if fit else scaler.transform(embeddings)

    if not categorical_cols or df is None:
        return emb_scaled, scaler, None

    if encoder is None:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = encoder.fit_transform(df[categorical_cols]) if fit else encoder.transform(df[categorical_cols])
    features = np.hstack([emb_scaled, cat_encoded * onehot_weight])
    return features, scaler, encoder


# =====================
# Step 5. 평가 지표
# =====================

def compute_all_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False,
) -> Optional[Dict]:
    """
    모든 평가 지표를 한 번에 계산.

    Returns:
        지표 딕셔너리 또는 None (유효 클러스터 1개 이하)
    """
    labels_np = _to_numpy(labels)
    X_np = _to_numpy(X)

    mask = labels_np != -1
    n_total = len(labels_np)
    n_noise = int((~mask).sum())

    if len(set(labels_np[mask])) <= 1:
        return None

    if _GPU:
        X_gpu = cp.asarray(X_np[mask], dtype=cp.float32)
        labels_gpu = cp.asarray(labels_np[mask])
        sil = float(cython_silhouette_score(X_gpu, labels_gpu))
    else:
        sil = float(cython_silhouette_score(X_np[mask], labels_np[mask]))

    dbi = davies_bouldin_score(X_np[mask], labels_np[mask])
    chi = calinski_harabasz_score(X_np[mask], labels_np[mask])

    valid_labels = labels_np[mask]
    unique_labels = np.unique(valid_labels)
    n_clusters = len(unique_labels)
    n_valid = len(valid_labels)

    cluster_sizes = {int(l): int((valid_labels == l).sum()) for l in unique_labels}

    probs = np.array([cluster_sizes[l] / n_valid for l in unique_labels])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    normalized_entropy = entropy / np.log(n_clusters) if n_clusters > 1 else 0.0

    if n_clusters > 1:
        sizes = np.sort(np.array(list(cluster_sizes.values())))
        n = len(sizes)
        gini = (2 * np.sum(np.arange(1, n + 1) * sizes) - (n + 1) * sizes.sum()) / (n * sizes.sum())
    else:
        gini = 1.0

    metrics = {
        "sil": round(sil, 4),
        "dbi": round(dbi, 4),
        "chi": round(chi, 1),
        "k": n_clusters,
        "noise_ratio": round(n_noise / n_total, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "gini": round(gini, 4),
        "max_cluster_ratio": round(max(cluster_sizes.values()) / n_valid, 4),
        "n_total": n_total,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes,
    }

    if verbose:
        logger.info("지표 계산 완료",
            sil=round(sil, 3), k=n_clusters,
            noise=round(n_noise / n_total, 3),
            entropy=round(normalized_entropy, 3), gini=round(gini, 3))

    return metrics


def is_valid(metrics: Dict, validity_params: Dict) -> bool:
    """합격 여부 판단 (k 범위 + 노이즈 비율).

    Args:
        validity_params: {"k_min": int, "k_max": int, "noise_max": float}
    """
    return (
        validity_params["k_min"] <= metrics["k"] <= validity_params["k_max"]
        and metrics["noise_ratio"] <= validity_params["noise_max"]
    )


def compute_score(metrics: Dict, scoring_params: Dict) -> float:
    """
    품질 점수 계산 (0~1, 높을수록 좋음).
    DBI·CHI는 리포팅 전용 — composite score에 포함하지 않음.

    Args:
        scoring_params: {
            "weights": {"silhouette": float, "entropy": float, "gini": float},
            "silhouette_thresholds": {"low": float, "mid": float},
            "entropy_thresholds": {"good": float, "acceptable": float},
        }
    """
    weights = scoring_params["weights"]
    sil_thresh = scoring_params["silhouette_thresholds"]
    entropy_thresh = scoring_params["entropy_thresholds"]

    sil = metrics["sil"]
    if sil < sil_thresh["low"]:
        score_sil = max(0.0, sil / sil_thresh["low"] * 0.5)
    elif sil < sil_thresh["mid"]:
        score_sil = 0.5 + (sil - sil_thresh["low"]) * 0.3 / (sil_thresh["mid"] - sil_thresh["low"])
    else:
        score_sil = min(1.0, 0.8 + (sil - sil_thresh["mid"]) * 0.4)

    entropy = metrics["normalized_entropy"]
    if entropy >= entropy_thresh["good"]:
        score_entropy = 1.0
    elif entropy >= entropy_thresh["acceptable"]:
        score_entropy = 0.5 + (entropy - entropy_thresh["acceptable"]) * 2.5
    else:
        score_entropy = max(0.0, entropy / entropy_thresh["acceptable"] * 0.5)

    score_gini = max(0.0, 1.0 - metrics["gini"])

    total = (
        score_sil       * weights.get("silhouette", 0)
        + score_entropy * weights.get("entropy", 0)
        + score_gini    * weights.get("gini", 0)
    )
    return round(total, 4)


# =====================
# Step 5. UMAP + HDBSCAN
# =====================

def reduce_dimensions(
    features: np.ndarray,
    umap_params: Dict,
) -> Tuple[np.ndarray, object]:
    """UMAP 차원 축소. Returns (X_reduced, umap_model)."""
    logger.info("UMAP 시작", backend="cuML" if _GPU else "CPU", params=umap_params)
    umap_model = UMAP(**umap_params)
    X = _to_numpy(umap_model.fit_transform(features)).astype(np.float32)
    logger.info("UMAP 완료", input=str(features.shape), output=str(X.shape))
    return X, umap_model


def fit_hdbscan(
    X: np.ndarray,
    mcs: int,
    ms: int,
    hdbscan_params: Dict,
) -> Tuple[np.ndarray, object]:
    """단일 파라미터 조합으로 HDBSCAN 실행. Returns (labels, model).

    Args:
        hdbscan_params: {"metric": str, "cluster_selection_method": str, "gen_min_span_tree": bool}
    """
    X_in = cp.asarray(X, dtype=cp.float32) if _GPU else X
    clusterer = HDBSCAN(
        min_cluster_size=mcs,
        min_samples=ms,
        **hdbscan_params,
    )
    labels = clusterer.fit_predict(X_in)
    return _to_numpy(labels), clusterer


# =====================
# Step 5. Optuna 최적화
# =====================

def optuna_search(
    features: np.ndarray,
    umap_fixed_params: Dict,
    hdbscan_params: Dict,
    validity_params: Dict,
    scoring_params: Dict,
    n_trials: int,
    sampler_seed: int,
    log_file: str,
    study_name: str,
    ranges: Dict,
    timeout: Optional[int] = None,
    storage: Optional[str] = None,
) -> Optional[Dict]:
    """
    Optuna TPE 샘플러로 UMAP + HDBSCAN 하이퍼파라미터 최적화.

    Args:
        umap_fixed_params: Optuna가 튜닝하지 않는 UMAP 고정값 (metric, random_state)
        hdbscan_params: HDBSCAN 고정값 (metric, cluster_selection_method, gen_min_span_tree)
        validity_params: is_valid() 기준 (k_min, k_max, noise_max)
        scoring_params: compute_score() 가중치/임계값
        ranges: Optuna 탐색 범위 dict (config optuna.ranges 구조)

    Returns:
        {"score", "metrics", "labels", "model", "umap_model", "hyperparams"} 또는 None
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        raise ImportError("optuna를 설치하세요: pip install optuna")

    best_result: Optional[Dict] = None

    def objective(trial):
        nonlocal best_result

        r = ranges["min_cluster_size"]
        mcs = trial.suggest_int("mcs", r["min"], r["max"], step=r.get("step", 1))

        r = ranges["min_samples"]
        ms = trial.suggest_int("ms", r["min"], r["max"], step=r.get("step", 1))

        r = ranges["umap_n_components"]
        n_comp = trial.suggest_int("n_comp", r["min"], r["max"])

        r = ranges["umap_n_neighbors"]
        n_neigh = trial.suggest_int("n_neigh", r["min"], r["max"], step=r.get("step", 1))

        r = ranges["umap_min_dist"]
        min_dist = trial.suggest_float("min_dist", r["min"], r["max"])

        umap_params = {
            **umap_fixed_params,
            "n_components": n_comp,
            "n_neighbors": n_neigh,
            "min_dist": min_dist,
        }

        try:
            X_umap, umap_model = reduce_dimensions(features, umap_params)
            labels, hdb = fit_hdbscan(X_umap, mcs, ms, hdbscan_params)
        except Exception as e:
            logger.warning("trial 실패", trial=trial.number, error=str(e))
            return -1.0

        metrics = compute_all_metrics(X_umap, labels)
        if metrics is None or not is_valid(metrics, validity_params):
            return -1.0

        score = compute_score(metrics, scoring_params)
        hyperparams = dict(mcs=mcs, ms=ms, n_comp=n_comp, n_neigh=n_neigh, min_dist=min_dist)

        log_path = Path(log_file)
        logs = json.loads(log_path.read_text()) if log_path.exists() else []
        logs.append({
            "timestamp": datetime.now().isoformat(),
            "trial": trial.number,
            "hyperparams": hyperparams,
            "metrics": {k: v for k, v in metrics.items() if k != "cluster_sizes"},
            "score": round(score, 4),
        })
        log_path.write_text(json.dumps(logs, indent=2))

        if best_result is None or score > best_result["score"]:
            best_result = dict(
                score=score,
                metrics=metrics,
                labels=_to_numpy(labels).copy(),
                model=hdb,
                umap_model=umap_model,
                hyperparams=hyperparams,
            )
        return score

    try:
        optuna.delete_study(study_name=study_name, storage=storage)
    except Exception:
        pass

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=sampler_seed),
        load_if_exists=True,
    )

    if storage:
        logger.info("Optuna 대시보드", cmd=f"optuna-dashboard {storage}")

    logger.info("Optuna 시작", n_trials=n_trials, timeout=timeout)
    study.optimize(objective, n_trials=n_trials, timeout=timeout,
                   show_progress_bar=True, gc_after_trial=True)

    if best_result:
        logger.info("Optuna 완료",
            score=round(best_result["score"], 3),
            k=best_result["metrics"]["k"],
            sil=round(best_result["metrics"]["sil"], 3),
            noise=round(best_result["metrics"]["noise_ratio"], 3))
    else:
        logger.warning("Optuna 완료 — 합격 trial 없음")

    return best_result


# =====================
# Step 5-6. 모델 저장 / 로드
# =====================

def train_and_save(
    embeddings: np.ndarray,
    save_dir: str,
    umap_params: Dict,
    hdbscan_params: Dict,
    validity_params: Dict,
    scoring_params: Dict,
    optuna_params: Dict,
    categorical_cols: Optional[List[str]],
    onehot_weight: float,
    df: Optional[pd.DataFrame] = None,
) -> Dict:
    """
    Optuna 최적화 후 모델 일체를 저장.

    저장 구조:
        {save_dir}/
            scaler.joblib
            encoder.joblib  (범주형 사용 시)
            umap_model.joblib
            hdbscan_model.joblib
            labels.npy
            metadata.json

    Args:
        umap_params: yaml의 umap: (metric, random_state)
        hdbscan_params: yaml의 hdbscan:
        validity_params: yaml의 validity:
        scoring_params: yaml의 scoring:
        optuna_params: yaml의 optuna: (n_trials, sampler_seed, log_file, study_name,
                                       storage, timeout, ranges)

    Returns:
        {"score", "metrics", "labels", "model", "umap_model", "hyperparams"}
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    features, scaler, encoder = prepare_features(
        embeddings, categorical_cols, onehot_weight, df=df, fit=True
    )
    joblib.dump(scaler, save_path / "scaler.joblib")
    if encoder is not None:
        joblib.dump(encoder, save_path / "encoder.joblib")

    result = optuna_search(
        features=features,
        umap_fixed_params=umap_params,
        hdbscan_params=hdbscan_params,
        validity_params=validity_params,
        scoring_params=scoring_params,
        n_trials=optuna_params["n_trials"],
        sampler_seed=optuna_params["sampler_seed"],
        log_file=optuna_params["log_file"],
        study_name=optuna_params["study_name"],
        ranges=optuna_params["ranges"],
        timeout=optuna_params.get("timeout"),
        storage=optuna_params.get("storage"),
    )
    if result is None:
        raise RuntimeError("Optuna 최적화 실패 — 합격 trial이 없습니다.")

    joblib.dump(result["umap_model"], save_path / "umap_model.joblib")
    joblib.dump(result["model"], save_path / "hdbscan_model.joblib")
    np.save(save_path / "labels.npy", result["labels"])

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "hyperparams": result["hyperparams"],
        "metrics": {k: v for k, v in result["metrics"].items() if k != "cluster_sizes"},
        "cluster_sizes": {str(k): v for k, v in result["metrics"].get("cluster_sizes", {}).items()},
        "categorical_cols": categorical_cols or [],
        "onehot_weight": onehot_weight,
    }
    (save_path / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))

    logger.info("모델 저장 완료", path=str(save_path))
    return result


def load_and_predict(
    embeddings: np.ndarray,
    model_dir: str,
    df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Dict]:
    """
    저장된 모델로 새 데이터에 클러스터 레이블 부여.

    Args:
        embeddings: (N, D) 임베딩 배열
        model_dir: train_and_save() 저장 경로
        df: 범주형 컬럼 소스 (학습 시 범주형 사용했으면 필요)

    Returns:
        (labels, metadata)
    """
    model_path = Path(model_dir)
    metadata = json.loads((model_path / "metadata.json").read_text())
    categorical_cols = metadata.get("categorical_cols") or []
    onehot_weight = metadata["onehot_weight"]

    scaler: StandardScaler = joblib.load(model_path / "scaler.joblib")
    umap_model = joblib.load(model_path / "umap_model.joblib")
    hdb_model = joblib.load(model_path / "hdbscan_model.joblib")

    _, _, encoder = prepare_features(
        embeddings, categorical_cols, onehot_weight, df=df,
        scaler=scaler, fit=False
    )
    # encoder가 저장되어 있으면 로드해서 다시 변환
    enc_path = model_path / "encoder.joblib"
    if enc_path.exists() and categorical_cols and df is not None:
        encoder = joblib.load(enc_path)
        emb_scaled = scaler.transform(embeddings)
        cat_encoded = encoder.transform(df[categorical_cols])
        features = np.hstack([emb_scaled, cat_encoded * onehot_weight])
    else:
        features = scaler.transform(embeddings)

    X_umap = _to_numpy(umap_model.transform(features)).astype(np.float32)

    try:
        from hdbscan import approximate_predict as _approx_predict
        labels, _ = _approx_predict(hdb_model, X_umap)
        labels = _to_numpy(labels)
    except (ImportError, AttributeError, TypeError):
        logger.warning("approximate_predict 미지원 → fit_predict 사용")
        X_in = cp.asarray(X_umap) if _GPU else X_umap
        labels = _to_numpy(hdb_model.fit_predict(X_in))

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    logger.info("예측 완료", n=len(labels), k=n_clusters)
    return labels, metadata


# =====================
# Step 7. 시각화 & 성능 평가
# =====================

def plot_clusters_2d(
    embeddings: np.ndarray,
    labels: np.ndarray,
    umap_params: Dict,
    output_path: str,
    df: Optional[pd.DataFrame] = None,
    hover_cols: Optional[List[str]] = None,
    title: str = "MAUDE MDR 클러스터 분포 (UMAP 2D)",
):
    """UMAP 2D 투영 후 클러스터 인터랙티브 산점도 저장 (.html)."""
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("pip install plotly")

    umap_2d = UMAP(**umap_params)
    coords = _to_numpy(umap_2d.fit_transform(embeddings)).astype(np.float32)
    labels_np = _to_numpy(labels)

    plot_data: Dict = {"x": coords[:, 0], "y": coords[:, 1], "cluster": labels_np.astype(str)}
    if df is not None and hover_cols:
        for col in hover_cols:
            if col in df.columns:
                plot_data[col] = df[col].values

    df_plot = pd.DataFrame(plot_data)
    hover_data = [c for c in (hover_cols or []) if c in df_plot.columns]

    fig = px.scatter(df_plot, x="x", y="y", color="cluster", hover_data=hover_data,
                     title=title, width=1200, height=800,
                     color_discrete_sequence=px.colors.qualitative.Safe)
    fig.update_traces(marker=dict(size=3, opacity=0.6))
    fig.update_layout(legend_title_text="Cluster ID")
    fig.write_html(output_path)
    logger.info("산점도 저장 완료", path=output_path)
    return fig


def plot_cluster_distribution(
    labels: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
):
    """클러스터별 데이터 수 막대 그래프 저장 (.html)."""
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("pip install plotly")

    labels_np = _to_numpy(labels)
    noise_count = int((labels_np == -1).sum())
    noise_ratio = noise_count / len(labels_np)

    unique, counts = np.unique(labels_np[labels_np != -1], return_counts=True)
    df_dist = pd.DataFrame({"cluster": unique.astype(str), "count": counts})
    df_dist = df_dist.sort_values("count", ascending=False)

    if title is None:
        title = f"클러스터 크기 분포 (k={len(unique)}, 노이즈={noise_count:,}개 / {noise_ratio:.1%})"

    fig = px.bar(df_dist, x="cluster", y="count", title=title,
                 labels={"count": "데이터 수", "cluster": "클러스터 ID"},
                 color="count", color_continuous_scale="Blues")
    fig.update_layout(coloraxis_showscale=False)
    fig.write_html(output_path)
    logger.info("분포 그래프 저장 완료", path=output_path)
    return fig


def print_metrics_report(
    metrics: Dict,
    hyperparams: Optional[Dict] = None,
    score: Optional[float] = None,
):
    """클러스터링 성능 지표 요약 출력."""
    sep = "=" * 55
    print(sep)
    print("  클러스터링 성능 요약")
    print(sep)

    if hyperparams:
        print("[하이퍼파라미터]")
        for k, v in hyperparams.items():
            print(f"  {k:20s}: {v}")
        print()

    print("[클러스터 정보]")
    print(f"  {'클러스터 수 (k)':<22}: {metrics['k']}")
    print(f"  {'전체 데이터':<22}: {metrics['n_total']:,}")
    print(f"  {'노이즈 포인트':<22}: {metrics['n_noise']:,}  ({metrics['noise_ratio']:.1%})")
    print()

    print("[품질 지표]")
    print(f"  {'Silhouette':<22}: {metrics['sil']:.4f}   ↑ 높을수록 좋음")
    print(f"  {'Davies-Bouldin':<22}: {metrics['dbi']:.4f}   ↓ 낮을수록 좋음")
    print(f"  {'Calinski-Harabasz':<22}: {metrics['chi']:.1f}   ↑ 높을수록 좋음")
    print()

    print("[분포 균등성]")
    print(f"  {'Normalized Entropy':<22}: {metrics['normalized_entropy']:.4f}   ↑ 균등")
    print(f"  {'Gini':<22}: {metrics['gini']:.4f}   ↓ 균등")
    print(f"  {'Max Cluster Ratio':<22}: {metrics['max_cluster_ratio']:.1%}   ↓ 균등")
    print()

    if score is not None:
        print(f"  {'종합 점수':<22}: {score:.4f}   (0~1, 높을수록 좋음)")
        print()

    if metrics.get("cluster_sizes"):
        sizes = sorted(metrics["cluster_sizes"].values(), reverse=True)
        print(f"[클러스터 크기 Top-5]  {sizes[:5]}")
        print(f"  최대: {sizes[0]:,}  /  최소: {sizes[-1]:,}")

    print(sep)


# =====================
# 실행 예시 (7-step 파이프라인)
# =====================
if __name__ == "__main__":
    MODEL_DIR   = "models/clustering/v1"
    CATEGORICAL = ["PATIENT_HARM", "DEFECT_TYPE", "DEFECT_CONFIRMED"]
    TEXT_COL    = "PROBLEM_COMPONENTS"
    SNTC_COL    = "MDR_SNTC"

    configure_logging(level="INFO")

    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret, ocsp_fail_open=True)
    cursor = conn.cursor()

    try:
        # ── Step 1. 데이터 로드 ────────────────────────────────────────────
        cursor.execute(f"""
            SELECT MDR_REPORT_KEY, PRODUCT_CODE,
                   {", ".join(CATEGORICAL)}, {TEXT_COL}
            FROM MAUDE.SILVER.EVENT_LLM_EXTRACTED
        """)
        df = cursor.fetch_pandas_all()
        logger.info("데이터 로드 완료", rows=len(df))

        # ── Step 2. 어휘 필터링 ────────────────────────────────────────────
        vocab = analyze_keywords(df, text_col=TEXT_COL, min_freq=10)

        # ── Step 3. 텍스트 전처리 + 임베딩 ───────────────────────────────
        df = prepare_text_col(df, text_col=TEXT_COL, output_col=SNTC_COL, vocab=vocab)
        embeddings = embed_texts(df[SNTC_COL].tolist(),
                                 model="pritamdeka/S-PubMedBert-MS-MARCO",
                                 batch_size=32, normalize=True)

        # ── Step 4-5. Optuna 튜닝 + 베스트 모델 저장 ─────────────────────
        train_and_save(
            embeddings      = embeddings,
            save_dir        = MODEL_DIR,
            df              = df,
            categorical_cols = CATEGORICAL,
            onehot_weight   = 5.0,
            umap_params     = {"metric": "cosine", "random_state": 42},
            hdbscan_params  = {"metric": "euclidean",
                               "cluster_selection_method": "eom",
                               "gen_min_span_tree": True},
            validity_params = {"k_min": 3, "k_max": 100, "noise_max": 0.35},
            scoring_params  = {
                "weights": {"silhouette": 0.60, "entropy": 0.20, "gini": 0.20},
                "silhouette_thresholds": {"low": 0.4, "mid": 0.5},
                "entropy_thresholds": {"good": 0.7, "acceptable": 0.5},
            },
            optuna_params   = {
                "n_trials": 100, "sampler_seed": 42,
                "log_file": "results_optuna.json",
                "study_name": "maude_clustering",
                "storage": "sqlite:///optuna_clustering.db",
                "timeout": None,
                "ranges": {
                    "min_cluster_size":  {"min": 50,  "max": 500, "step": 10},
                    "min_samples":       {"min": 3,   "max": 20,  "step": 1},
                    "umap_n_components": {"min": 10,  "max": 20},
                    "umap_n_neighbors":  {"min": 30,  "max": 80,  "step": 5},
                    "umap_min_dist":     {"min": 0.0, "max": 0.3},
                },
            },
        )

        # ── Step 6. 저장된 베스트 모델로 전체 클러스터링 ─────────────────
        labels, metadata = load_and_predict(
            embeddings = embeddings,
            model_dir  = MODEL_DIR,
            df         = df,
        )
        metrics = compute_all_metrics(embeddings, labels)
        print_metrics_report(metrics=metrics, hyperparams=metadata.get("hyperparams"))

        # ── Step 7. 시각화 ─────────────────────────────────────────────────
        plot_clusters_2d(
            embeddings  = embeddings,
            labels      = labels,
            umap_params = {"n_neighbors": 50, "min_dist": 0.3,
                           "n_components": 2, "metric": "cosine", "random_state": 42},
            output_path = "cluster_plot.html",
            df          = df,
            hover_cols  = ["PRODUCT_CODE"] + CATEGORICAL,
        )
        plot_cluster_distribution(labels=labels, output_path="cluster_distribution.html")

    finally:
        cursor.close()
        conn.close()

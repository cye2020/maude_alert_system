# --------------------------------------------
# 표준 라이브러리
# --------------------------------------------
import ast
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# --------------------------------------------
# 서드파티 라이브러리
# --------------------------------------------
import joblib
import numpy as np
import pandas as pd
import snowflake.connector
import structlog
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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


# =====================
# 평가 지표 (독립 함수)
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

    지표 구성:
        - 기본: silhouette, DBI, CHI
        - 균등성: 정규화 엔트로피, 지니 계수, 최대 클러스터 점유율
        - 요약: k, noise_ratio, n_total, n_noise, cluster_sizes
    """
    labels_np = _to_numpy(labels)
    X_np = _to_numpy(X)

    mask = labels_np != -1
    n_total = len(labels_np)
    n_noise = int((~mask).sum())

    if len(set(labels_np[mask])) <= 1:
        return None

    # 1. 기본 지표
    if _GPU:
        X_gpu = cp.asarray(X_np[mask], dtype=cp.float32)
        labels_gpu = cp.asarray(labels_np[mask])
        sil = float(cython_silhouette_score(X_gpu, labels_gpu))
    else:
        sil = float(cython_silhouette_score(X_np[mask], labels_np[mask]))

    dbi = davies_bouldin_score(X_np[mask], labels_np[mask])
    chi = calinski_harabasz_score(X_np[mask], labels_np[mask])

    # 2. 균등성 지표
    valid_labels = labels_np[mask]
    unique_labels = np.unique(valid_labels)
    n_clusters = len(unique_labels)
    n_valid = len(valid_labels)

    cluster_sizes = {int(l): int((valid_labels == l).sum()) for l in unique_labels}

    # 정규화 엔트로피 (0~1, 높을수록 균등)
    probs = np.array([cluster_sizes[l] / n_valid for l in unique_labels])
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    normalized_entropy = entropy / np.log(n_clusters) if n_clusters > 1 else 0.0

    # 지니 계수 (0~1, 낮을수록 균등)
    if n_clusters > 1:
        sizes = np.sort(np.array(list(cluster_sizes.values())))
        n = len(sizes)
        gini = (2 * np.sum(np.arange(1, n + 1) * sizes) - (n + 1) * sizes.sum()) / (n * sizes.sum())
    else:
        gini = 1.0

    max_cluster_ratio = max(cluster_sizes.values()) / n_valid

    metrics = {
        "sil": round(sil, 4),
        "dbi": round(dbi, 4),
        "chi": round(chi, 1),
        "k": n_clusters,
        "noise_ratio": round(n_noise / n_total, 4),
        "normalized_entropy": round(normalized_entropy, 4),
        "gini": round(gini, 4),
        "max_cluster_ratio": round(max_cluster_ratio, 4),
        "n_total": n_total,
        "n_noise": n_noise,
        "cluster_sizes": cluster_sizes,
    }

    if verbose:
        logger.info(
            "지표 계산 완료",
            sil=round(sil, 3),
            k=n_clusters,
            noise=round(n_noise / n_total, 3),
            entropy=round(normalized_entropy, 3),
            gini=round(gini, 3),
        )

    return metrics


def is_valid(
    metrics: Dict,
    k_min: int,
    k_max: int,
    noise_max: float,
) -> bool:
    """합격 여부 판단 (k 범위 + 노이즈 비율)"""
    return k_min <= metrics["k"] <= k_max and metrics["noise_ratio"] <= noise_max


def compute_score(
    metrics: Dict,
    weights: Dict[str, float],
    sil_thresh: Dict[str, float],
    entropy_thresh: Dict[str, float],
    dbi_ref: float,
    chi_ref: float,
) -> float:
    """
    품질 점수 계산 (0~1, 높을수록 좋음).

    is_valid() 통과 후 순위 결정용.
    각 지표를 0~1로 정규화 후 가중 합산.

    Args:
        metrics: compute_all_metrics() 결과
        weights: 지표별 가중치 딕셔너리 (합계 = 1.0)
        sil_thresh: Silhouette 구간 임계값 {'low': float, 'mid': float}
        entropy_thresh: Entropy 구간 임계값 {'good': float, 'acceptable': float}
        dbi_ref: DBI 기준값 (이 값 이상이면 0점)
        chi_ref: CHI 기준값 (이 값 이상이면 만점)
    """
    # Silhouette (0~1)
    sil = metrics["sil"]
    if sil < sil_thresh["low"]:
        score_sil = max(0.0, sil / sil_thresh["low"] * 0.5)
    elif sil < sil_thresh["mid"]:
        scale = 0.3 / (sil_thresh["mid"] - sil_thresh["low"])
        score_sil = 0.5 + (sil - sil_thresh["low"]) * scale
    else:
        score_sil = min(1.0, 0.8 + (sil - sil_thresh["mid"]) * 0.4)

    # 정규화 엔트로피 (0~1)
    entropy = metrics["normalized_entropy"]
    if entropy >= entropy_thresh["good"]:
        score_entropy = 1.0
    elif entropy >= entropy_thresh["acceptable"]:
        score_entropy = 0.5 + (entropy - entropy_thresh["acceptable"]) * 2.5
    else:
        score_entropy = max(0.0, entropy / entropy_thresh["acceptable"] * 0.5)

    # 지니 계수 (낮을수록 좋음 → 역산)
    score_gini = max(0.0, 1.0 - metrics["gini"])

    # DBI (낮을수록 좋음 → 역산)
    score_dbi = max(0.0, 1.0 - metrics["dbi"] / dbi_ref)

    # CHI (높을수록 좋음)
    score_chi = min(1.0, metrics["chi"] / chi_ref)

    total = (
        score_sil * weights.get("silhouette", 0)
        + score_entropy * weights.get("entropy", 0)
        + score_gini * weights.get("gini", 0)
        + score_dbi * weights.get("dbi", 0)
        + score_chi * weights.get("chi", 0)
    )

    return round(total, 4)


# =====================
# 메인 클래스
# =====================

class Clustering:
    """
    MAUDE MDR 클러스터링 파이프라인.

    구성:
        1. 텍스트 전처리 (build_mdr_sntc)
        2. 임베딩 생성 (embed_texts)
        3. 피처 준비 [임베딩 + 범주형] (_prepare_features)
        4. UMAP 차원 축소 (reduce_dimensions)
        5. HDBSCAN 클러스터링
           - 그리드 서치 (run_grid_search)
           - Optuna 최적화 (optuna_search)
        6. 모델 저장 / 로드 (train_and_save / load_and_predict)
    """

    def __init__(
        self,
        # 임베딩
        embedding_model: str,
        batch_size: int,
        normalize: bool,
        # UMAP
        umap_params: Dict,
        # HDBSCAN
        hdbscan_metric: str,
        hdbscan_cluster_selection_method: str,
        hdbscan_gen_min_span_tree: bool,
        # 합격 조건
        k_min: int,
        k_max: int,
        noise_max: float,
        # 범주형 피처
        onehot_weight: float,
        # 그리드 서치
        mcs_grid: List[int],
        ms_grid: List[int],
        # 스코어링
        score_weights: Dict[str, float],
        sil_thresh: Dict[str, float],
        entropy_thresh: Dict[str, float],
        dbi_ref: float,
        chi_ref: float,
        # 범주형 컬럼 (없으면 None)
        categorical_cols: Optional[List[str]] = None,
    ):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.normalize = normalize

        self.umap_params = umap_params

        self.hdbscan_metric = hdbscan_metric
        self.hdbscan_cluster_selection_method = hdbscan_cluster_selection_method
        self.hdbscan_gen_min_span_tree = hdbscan_gen_min_span_tree

        self.k_min = k_min
        self.k_max = k_max
        self.noise_max = noise_max

        self.categorical_cols = categorical_cols or []
        self.onehot_weight = onehot_weight

        self.mcs_grid = mcs_grid
        self.ms_grid = ms_grid

        self.score_weights = score_weights
        self.sil_thresh = sil_thresh
        self.entropy_thresh = entropy_thresh
        self.dbi_ref = dbi_ref
        self.chi_ref = chi_ref

    # --------------------------------------------------
    # 1. 텍스트 전처리
    # --------------------------------------------------

    def build_mdr_sntc(self, df: pd.DataFrame) -> pd.DataFrame:
        """LLM 추출 4개 컬럼을 임베딩용 문장(MDR_SNTC)으로 재생성"""

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

    def _fill_unknown(val) -> str:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "unknown"
        return str(val)

        df = df.copy()
        df["MDR_SNTC"] = (
            df["PATIENT_HARM"].apply(fill_unknown) + ". "
            + df["PROBLEM_COMPONENTS"].apply(parse_components) + ". "
            + df["DEFECT_CONFIRMED"].apply(fill_unknown) + ". "
            + df["DEFECT_TYPE"].apply(fill_unknown)
        )
        return df

    # --------------------------------------------------
    # 2. 임베딩 생성
    # --------------------------------------------------

    def embed_texts(self, texts: list) -> np.ndarray:
        """SentenceTransformer로 768D 임베딩 생성"""
        logger.info("임베딩 시작", model=self.embedding_model, n=len(texts))
        model = SentenceTransformer(self.embedding_model)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        logger.info("임베딩 완료", shape=str(embeddings.shape))
        return embeddings

    # --------------------------------------------------
    # 3. 피처 준비 (임베딩 + 범주형)
    # --------------------------------------------------

    def _prepare_features(
        self,
        embeddings: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        scaler: Optional[StandardScaler] = None,
        encoder: Optional[OneHotEncoder] = None,
        fit: bool = True,
    ) -> Tuple[np.ndarray, StandardScaler, Optional[OneHotEncoder]]:
        """
        임베딩 스케일링 + 범주형 원핫 인코딩 결합.

        Args:
            embeddings: (N, D) 임베딩 배열
            df: 범주형 컬럼 소스
            scaler: 기존 스케일러 (fit=False 시 사용)
            encoder: 기존 인코더 (fit=False 시 사용)
            fit: True면 스케일러/인코더를 학습, False면 transform만

        Returns:
            (features, scaler, encoder)
        """
        if scaler is None:
            scaler = StandardScaler()

        emb_scaled = scaler.fit_transform(embeddings) if fit else scaler.transform(embeddings)

        if not self.categorical_cols or df is None:
            return emb_scaled, scaler, None

        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        cat_data = df[self.categorical_cols]
        cat_encoded = encoder.fit_transform(cat_data) if fit else encoder.transform(cat_data)
        features = np.hstack([emb_scaled, cat_encoded * self.onehot_weight])
        return features, scaler, encoder

    # --------------------------------------------------
    # 4. UMAP 차원 축소
    # --------------------------------------------------

    def reduce_dimensions(
        self,
        features: np.ndarray,
        umap_params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, object]:
        """
        UMAP 차원 축소 (GPU 있으면 cuML, 없으면 umap-learn).

        Returns:
            (X_reduced, umap_model)  — umap_model은 train_and_save에서 저장
        """
        params = umap_params if umap_params is not None else self.umap_params
        logger.info("UMAP 시작", backend="cuML" if _GPU else "CPU", params=params)
        umap_model = UMAP(**params)
        X = umap_model.fit_transform(features)
        X_np = _to_numpy(X).astype(np.float32)
        logger.info("UMAP 완료", input=str(features.shape), output=str(X_np.shape))
        return X_np, umap_model

    # --------------------------------------------------
    # 5-a. 단일 HDBSCAN 실행
    # --------------------------------------------------

    def _fit_hdbscan(
        self,
        X: np.ndarray,
        mcs: int,
        ms: int,
    ) -> Tuple[np.ndarray, object]:
        """
        단일 파라미터 조합으로 HDBSCAN 실행.

        Returns:
            (labels_np, hdb_model)
        """
        X_in = cp.asarray(X, dtype=cp.float32) if _GPU else X
        clusterer = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=self.hdbscan_metric,
            cluster_selection_method=self.hdbscan_cluster_selection_method,
            gen_min_span_tree=self.hdbscan_gen_min_span_tree,
        )
        labels = clusterer.fit_predict(X_in)
        return _to_numpy(labels), clusterer

    # --------------------------------------------------
    # 5-b. 그리드 서치
    # --------------------------------------------------

    def run_grid_search(self, X: np.ndarray) -> Dict:
        """
        HDBSCAN 그리드 서치로 최적 파라미터 자동 선택.

        합격 조건 (is_valid) 충족 + 종합 점수(compute_score) 최고인 조합 선택.

        Returns:
            {
                'mcs', 'ms', 'score', 'metrics', 'labels', 'model'
            }
        """
        logger.info(
            "HDBSCAN 그리드 서치 시작",
            mcs_grid=self.mcs_grid,
            ms_grid=self.ms_grid,
            k_range=f"{self.k_min}~{self.k_max}",
            noise_max=self.noise_max,
        )

        best: Optional[Dict] = None

        for mcs in self.mcs_grid:
            for ms in self.ms_grid:
                labels, hdb = self._fit_hdbscan(X, mcs, ms)
                metrics = compute_all_metrics(X, labels)

                if metrics is None:
                    logger.debug("스킵 (클러스터 부족)", mcs=mcs, ms=ms)
                    continue

                if not is_valid(metrics, self.k_min, self.k_max, self.noise_max):
                    logger.debug(
                        "스킵 (합격 조건 미달)",
                        mcs=mcs, ms=ms,
                        k=metrics["k"],
                        noise=round(metrics["noise_ratio"], 3),
                    )
                    continue

                score = compute_score(
                    metrics,
                    weights=self.score_weights,
                    sil_thresh=self.sil_thresh,
                    entropy_thresh=self.entropy_thresh,
                    dbi_ref=self.dbi_ref,
                    chi_ref=self.chi_ref,
                )
                logger.info(
                    "후보",
                    mcs=mcs, ms=ms,
                    k=metrics["k"],
                    noise=round(metrics["noise_ratio"], 3),
                    sil=round(metrics["sil"], 3),
                    score=round(score, 3),
                )

                if best is None or score > best["score"]:
                    best = dict(
                        mcs=mcs, ms=ms, score=score,
                        metrics=metrics, labels=labels, model=hdb,
                    )

        if best is None:
            raise RuntimeError(
                "합격 조건을 만족하는 클러스터 조합이 없습니다. "
                "mcs_grid / k_max / noise_max를 조정하세요."
            )

        logger.info(
            "그리드 서치 완료",
            mcs=best["mcs"], ms=best["ms"],
            k=best["metrics"]["k"],
            score=round(best["score"], 3),
        )
        return best

    def run_clustering(self, X: np.ndarray) -> np.ndarray:
        """하위 호환성용 — run_grid_search()의 labels 반환"""
        return self.run_grid_search(X)["labels"]

    # --------------------------------------------------
    # 5-c. Optuna 최적화
    # --------------------------------------------------

    def optuna_search(
        self,
        features: np.ndarray,
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

        합격 조건 불만족 시 페널티(-1.0) 반환 → Optuna가 탐색 공간 제한.

        Args:
            features: _prepare_features() 결과 (임베딩 + 범주형 결합)
            n_trials: 시도 횟수
            sampler_seed: TPE 샘플러 시드
            log_file: 합격 trial 결과 저장 경로
            study_name: 스터디 이름
            ranges: 하이퍼파라미터 탐색 범위 딕셔너리 (config의 optuna.ranges 구조)
                {
                    'min_cluster_size': {'min': int, 'max': int, 'step': int},
                    'min_samples':      {'min': int, 'max': int, 'step': int},
                    'umap_n_components':{'min': int, 'max': int},
                    'umap_n_neighbors': {'min': int, 'max': int, 'step': int},
                    'umap_min_dist':    {'min': float, 'max': float},
                }
            timeout: 최대 실행 시간 (초), None이면 제한 없음
            storage: Optuna 스토리지 URL (예: "sqlite:///optuna.db"), None이면 인메모리

        Returns:
            {
                'score', 'metrics', 'labels', 'model', 'umap_model', 'hyperparams'
            }
            또는 None (합격 trial 없음)
        """
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            raise ImportError("optuna를 설치하세요: pip install optuna")

        best_result: Optional[Dict] = None

        def objective(trial):
            nonlocal best_result

            r_mcs = ranges["min_cluster_size"]
            mcs = trial.suggest_int("mcs", r_mcs["min"], r_mcs["max"], step=r_mcs.get("step", 1))

            r_ms = ranges["min_samples"]
            ms = trial.suggest_int("ms", r_ms["min"], r_ms["max"], step=r_ms.get("step", 1))

            r_comp = ranges["umap_n_components"]
            n_comp = trial.suggest_int("n_comp", r_comp["min"], r_comp["max"])

            r_neigh = ranges["umap_n_neighbors"]
            n_neigh = trial.suggest_int("n_neigh", r_neigh["min"], r_neigh["max"], step=r_neigh.get("step", 1))

            r_dist = ranges["umap_min_dist"]
            min_dist = trial.suggest_float("min_dist", r_dist["min"], r_dist["max"])

            umap_params = {
                "n_components": n_comp,
                "n_neighbors": n_neigh,
                "min_dist": min_dist,
                "metric": self.umap_params.get("metric", "cosine"),
                "random_state": self.umap_params.get("random_state", 42),
            }

            try:
                X_umap, umap_model = self.reduce_dimensions(features, umap_params)
                labels, hdb = self._fit_hdbscan(X_umap, mcs, ms)
            except Exception as e:
                logger.warning("trial 실패", trial=trial.number, error=str(e))
                return -1.0

            metrics = compute_all_metrics(X_umap, labels)
            if metrics is None:
                return -1.0

            if not is_valid(metrics, self.k_min, self.k_max, self.noise_max):
                return -1.0

            score = compute_score(
                metrics,
                weights=self.score_weights,
                sil_thresh=self.sil_thresh,
                entropy_thresh=self.entropy_thresh,
                dbi_ref=self.dbi_ref,
                chi_ref=self.chi_ref,
            )
            hyperparams = dict(
                mcs=mcs, ms=ms, n_comp=n_comp,
                n_neigh=n_neigh, min_dist=min_dist,
            )

            # 합격 trial 로깅
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

        # 기존 스터디 초기화 후 재생성
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
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        if best_result:
            logger.info(
                "Optuna 완료",
                score=round(best_result["score"], 3),
                k=best_result["metrics"]["k"],
                sil=round(best_result["metrics"]["sil"], 3),
                noise=round(best_result["metrics"]["noise_ratio"], 3),
            )
        else:
            logger.warning("Optuna 완료 — 합격 trial 없음")

        return best_result

    # --------------------------------------------------
    # 6. 모델 저장
    # --------------------------------------------------

    def train_and_save(
        self,
        embeddings: np.ndarray,
        save_dir: str,
        df: Optional[pd.DataFrame] = None,
        use_optuna: bool = False,
        **optuna_kwargs,
    ) -> Dict:
        """
        클러스터링 실행 후 모델 일체를 저장.

        저장 구조:
            {save_dir}/
                scaler.joblib         — 임베딩 StandardScaler
                encoder.joblib        — 범주형 OneHotEncoder (범주형 사용 시)
                umap_model.joblib     — 학습된 UMAP
                hdbscan_model.joblib  — 학습된 HDBSCAN
                labels.npy            — 학습 데이터 클러스터 레이블
                metadata.json         — 하이퍼파라미터 + 지표 + 설정

        Args:
            embeddings: (N, D) 임베딩 배열
            save_dir: 저장 경로
            df: 범주형 컬럼 소스 (categorical_cols 지정 시 필요)
            use_optuna: True면 Optuna, False면 그리드 서치
            **optuna_kwargs: optuna_search() 전달 인자

        Returns:
            {
                'score', 'metrics', 'labels', 'model', 'umap_model', 'hyperparams'
            }
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 피처 준비 + 스케일러/인코더 저장
        features, scaler, encoder = self._prepare_features(embeddings, df, fit=True)
        joblib.dump(scaler, save_path / "scaler.joblib")
        if encoder is not None:
            joblib.dump(encoder, save_path / "encoder.joblib")

        # 클러스터링
        if use_optuna:
            result = self.optuna_search(features, **optuna_kwargs)
            if result is None:
                raise RuntimeError("Optuna 최적화 실패 — 합격 trial이 없습니다.")
        else:
            X_umap, umap_model = self.reduce_dimensions(features)
            grid_result = self.run_grid_search(X_umap)
            result = dict(
                labels=grid_result["labels"],
                metrics=grid_result["metrics"],
                score=grid_result["score"],
                model=grid_result["model"],
                umap_model=umap_model,
                hyperparams={"mcs": grid_result["mcs"], "ms": grid_result["ms"]},
            )

        # 모델 저장
        joblib.dump(result["umap_model"], save_path / "umap_model.joblib")
        joblib.dump(result["model"], save_path / "hdbscan_model.joblib")
        np.save(save_path / "labels.npy", result["labels"])

        metadata = {
            "timestamp": datetime.now().isoformat(),
            "hyperparams": result["hyperparams"],
            "metrics": {k: v for k, v in result["metrics"].items() if k != "cluster_sizes"},
            "cluster_sizes": {
                str(k): v for k, v in result["metrics"].get("cluster_sizes", {}).items()
            },
            "categorical_cols": self.categorical_cols,
            "onehot_weight": self.onehot_weight,
        }
        (save_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )

        logger.info("모델 저장 완료", path=str(save_path))
        return result

    # --------------------------------------------------
    # 7. 모델 로드 + 추론
    # --------------------------------------------------

    @staticmethod
    def load_and_predict(
        embeddings: np.ndarray,
        model_dir: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        저장된 모델을 로드해 새 데이터에 클러스터 레이블 부여.

        학습 시 사용한 스케일러/인코더/UMAP을 그대로 적용하므로
        train_and_save()와 동일한 전처리 파이프라인이 보장됩니다.

        Args:
            embeddings: (N, D) 새 데이터 임베딩
            model_dir: train_and_save() 저장 경로
            df: 범주형 컬럼 소스 (학습 시 범주형 사용했으면 필요)

        Returns:
            (labels, metadata)
            labels: (N,) 클러스터 레이블 (-1은 노이즈)
            metadata: metadata.json 내용
        """
        model_path = Path(model_dir)

        metadata = json.loads((model_path / "metadata.json").read_text())
        categorical_cols = metadata.get("categorical_cols", [])
        onehot_weight = metadata.get("onehot_weight", 5.0)

        scaler: StandardScaler = joblib.load(model_path / "scaler.joblib")
        umap_model = joblib.load(model_path / "umap_model.joblib")
        hdb_model = joblib.load(model_path / "hdbscan_model.joblib")

        # 피처 생성 (학습과 동일한 전처리)
        emb_scaled = scaler.transform(embeddings)

        enc_path = model_path / "encoder.joblib"
        if enc_path.exists() and categorical_cols and df is not None:
            encoder: OneHotEncoder = joblib.load(enc_path)
            cat_encoded = encoder.transform(df[categorical_cols])
            features = np.hstack([emb_scaled, cat_encoded * onehot_weight])
        else:
            features = emb_scaled

        # UMAP 변환
        X_umap = _to_numpy(umap_model.transform(features)).astype(np.float32)

        # HDBSCAN 예측
        try:
            # hdbscan(CPU) 라이브러리의 approximate_predict 시도
            from hdbscan import approximate_predict as _approx_predict
            labels, _ = _approx_predict(hdb_model, X_umap)
            labels = _to_numpy(labels)
        except (ImportError, AttributeError, TypeError):
            # cuML HDBSCAN — approximate_predict 미지원 → fit_predict(재훈련)
            logger.warning("approximate_predict 미지원 → fit_predict 사용 (내부 재훈련)")
            X_in = cp.asarray(X_umap) if _GPU else X_umap
            labels = _to_numpy(hdb_model.fit_predict(X_in))

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info("예측 완료", n=len(labels), k=n_clusters)

        return labels, metadata


# =====================
# 시각화 & 성능 평가
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
    """
    UMAP 2D 투영 후 클러스터 인터랙티브 산점도 저장 (.html).

    Args:
        embeddings: (N, D) 원본 임베딩 (시각화 전용 2D UMAP 별도 학습)
        labels: (N,) 클러스터 레이블
        umap_params: 2D UMAP 파라미터 딕셔너리
        output_path: 저장 경로 (.html)
        df: hover 데이터 소스
        hover_cols: hover 시 표시할 컬럼명 목록
        title: 그래프 제목

    Returns:
        plotly Figure 객체
    """
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly를 설치하세요: pip install plotly")

    logger.info("2D UMAP 시작 (시각화)")
    umap_2d = UMAP(**umap_params)
    coords = _to_numpy(umap_2d.fit_transform(embeddings)).astype(np.float32)

    labels_np = _to_numpy(labels)
    plot_data: Dict = {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": labels_np.astype(str),
    }

    if df is not None and hover_cols:
        for col in hover_cols:
            if col in df.columns:
                plot_data[col] = df[col].values

    df_plot = pd.DataFrame(plot_data)
    hover_data = [c for c in (hover_cols or []) if c in df_plot.columns]

    fig = px.scatter(
        df_plot,
        x="x", y="y",
        color="cluster",
        hover_data=hover_data,
        title=title,
        width=1200, height=800,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
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
    """
    클러스터별 데이터 수 막대 그래프 저장 (.html).

    Args:
        labels: (N,) 클러스터 레이블 (-1 노이즈 제외 후 표시)
        output_path: 저장 경로
        title: 그래프 제목 (기본값: 자동 생성)

    Returns:
        plotly Figure 객체
    """
    try:
        import plotly.express as px
    except ImportError:
        raise ImportError("plotly를 설치하세요: pip install plotly")

    labels_np = _to_numpy(labels)
    noise_count = int((labels_np == -1).sum())
    noise_ratio = noise_count / len(labels_np)

    valid_mask = labels_np != -1
    unique, counts = np.unique(labels_np[valid_mask], return_counts=True)

    df_dist = pd.DataFrame({"cluster": unique.astype(str), "count": counts})
    df_dist = df_dist.sort_values("count", ascending=False)

    if title is None:
        title = (
            f"클러스터 크기 분포 "
            f"(k={len(unique)}, 노이즈={noise_count:,}개 / {noise_ratio:.1%})"
        )

    fig = px.bar(
        df_dist,
        x="cluster", y="count",
        title=title,
        labels={"count": "데이터 수", "cluster": "클러스터 ID"},
        color="count",
        color_continuous_scale="Blues",
    )
    fig.update_layout(coloraxis_showscale=False)
    fig.write_html(output_path)

    logger.info("분포 그래프 저장 완료", path=output_path)
    return fig


def print_metrics_report(
    metrics: Dict,
    hyperparams: Optional[Dict] = None,
    score: Optional[float] = None,
):
    """
    클러스터링 성능 지표 요약 출력.

    Args:
        metrics: compute_all_metrics() 결과
        hyperparams: 하이퍼파라미터 딕셔너리 (선택)
        score: compute_score() 결과 (선택)
    """
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
# 실행 예시
# =====================
if __name__ == "__main__":
    import plotly.express as px

    configure_logging(level="INFO")

    # ── 0. Snowflake 연결 ──────────────────────────────────────────────────
    secret = get_secret("snowflake/de", region_name="ap-northeast-2")
    conn   = snowflake.connector.connect(**secret, ocsp_fail_open=True)
    cursor = conn.cursor()

    try:
        # ── 1. 데이터 로드 ─────────────────────────────────────
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

        # ── 2. 클러스터링 객체 생성 ────────────────────────────
        c = Clustering(
            embedding_model="pritamdeka/S-PubMedBert-MS-MARCO",
            batch_size=32,
            normalize=True,
            umap_params=dict(n_neighbors=50, min_dist=0.0, n_components=15, metric="cosine", random_state=42),
            hdbscan_metric="euclidean",
            hdbscan_cluster_selection_method="eom",
            hdbscan_gen_min_span_tree=True,
            k_min=3,
            k_max=100,
            noise_max=0.35,
            onehot_weight=5.0,
            mcs_grid=[50, 80, 100, 150, 200, 300],
            ms_grid=[3, 5, 7, 9],
            score_weights={"silhouette": 0.05, "entropy": 0.40, "gini": 0.25, "dbi": 0.05, "chi": 0.25},
            sil_thresh={"low": 0.4, "mid": 0.5},
            entropy_thresh={"good": 0.7, "acceptable": 0.5},
            dbi_ref=1.5,
            chi_ref=25000.0,
            categorical_cols=["patient_harm", "defect_type", "defect_confirmed"],
        )

        # ── 3. 전처리 + 임베딩 ─────────────────────────────────
        df         = c.build_mdr_sntc(df)
        embeddings = c.embed_texts(df["MDR_SNTC"].tolist())

        # ── 4. 클러스터링 + 저장 (그리드 서치) ────────────────
        result = c.train_and_save(
            embeddings = embeddings,
            save_dir   = "models/clustering/v1",
            df         = df,
            use_optuna = False,
        )

        # ── 5. 성능 출력 ───────────────────────────────────────
        print_metrics_report(
            metrics     = result["metrics"],
            hyperparams = result["hyperparams"],
            score       = result["score"],
        )

        # ── 6. 시각화 ──────────────────────────────────────────
        plot_clusters_2d(
            embeddings  = embeddings,
            labels      = result["labels"],
            umap_params = dict(n_neighbors=50, min_dist=0.3, n_components=2, metric="cosine", random_state=42),
            output_path = "cluster_plot.html",
            df          = df,
            hover_cols  = ["PRODUCT_CODE", "PATIENT_HARM", "MDR_SNTC"],
        )
        plot_cluster_distribution(
            labels      = result["labels"],
            output_path = "cluster_distribution.html",
        )

        # ── 7. 로드 + 추론 예시 ────────────────────────────────
        # new_embeddings = c.embed_texts(df["MDR_SNTC"].tolist())
        # labels, meta = Clustering.load_and_predict(
        #     embeddings = new_embeddings,
        #     model_dir  = "models/clustering/v1",
        #     df         = df,
        # )

        # ── 8. Optuna 사용 예시 ────────────────────────────────
        # result = c.train_and_save(
        #     embeddings  = embeddings,
        #     save_dir    = "models/clustering/v2_optuna",
        #     df          = df,
        #     use_optuna  = True,
        #     n_trials    = 100,
        #     sampler_seed= 42,
        #     log_file    = "results_optuna.json",
        #     study_name  = "maude_clustering",
        #     storage     = "sqlite:///optuna_clustering.db",
        #     ranges      = {
        #         "min_cluster_size":  {"min": 50,  "max": 500, "step": 10},
        #         "min_samples":       {"min": 3,   "max": 20,  "step": 1},
        #         "umap_n_components": {"min": 10,  "max": 20},
        #         "umap_n_neighbors":  {"min": 30,  "max": 80,  "step": 5},
        #         "umap_min_dist":     {"min": 0.0, "max": 0.3},
        #     },
        # )

    finally:
        cursor.close()
        conn.close()

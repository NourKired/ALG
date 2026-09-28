"""General-purpose similarities between two finite point sets.

Every public score follows the convention "larger means more similar".  Raw
distances are therefore negated.  Expensive metrics use deterministic point
subsampling so that the same object pair has the same score on every shard.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import wasserstein_distance as wasserstein_1d
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class MetricConfig:
    manifold_k: int = 5
    prc_k: int = 3
    prc_c: int = 3
    point_fscore_threshold: float = 0.10
    max_points: int = 64
    sliced_projections: int = 32
    dcd_alpha: float = 100.0
    sinkhorn_reg: float = 0.10
    prd_clusters: int = 20
    prd_angles: int = 1001
    c2st_folds: int = 5
    alpha_beta_steps: int = 30
    geometry_landmarks: int = 32
    geometry_repeats: int = 16
    geometry_i_max: int = 10
    geometry_gamma: float = 0.125


CONFIG = MetricConfig()
_INTERNAL_DISTANCE_CACHE: dict[str, np.ndarray] = {}
_PERSISTENCE_CACHE: dict[tuple[str, int], list[np.ndarray]] = {}
_CACHE_LIMIT = 4096


def configure_metrics(**kwargs) -> None:
    for key, value in kwargs.items():
        if not hasattr(CONFIG, key):
            raise ValueError(f"Unknown point-set metric option: {key}")
        setattr(CONFIG, key, value)


def _as_cloud(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("A point cloud must be a non-empty 2D array")
    if not np.isfinite(x).all():
        raise ValueError("Point clouds must contain finite values")
    return x


def _subsample(x: np.ndarray, max_points: int | None = None) -> np.ndarray:
    cap = CONFIG.max_points if max_points is None else max_points
    if cap <= 0 or len(x) <= cap:
        return x
    # Deterministic and independent of process/shard RNG state.
    return x[np.linspace(0, len(x) - 1, cap, dtype=int)]


def _cloud_key(x: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(x)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).hexdigest()
    return f"{contiguous.shape}:{contiguous.dtype}:{digest}"


def _cache_put(cache: dict, key, value) -> None:
    if len(cache) >= _CACHE_LIMIT:
        cache.clear()
    cache[key] = value


def _internal_distances(x: np.ndarray) -> np.ndarray:
    x = _subsample(_as_cloud(x))
    key = _cloud_key(x)
    if key not in _INTERNAL_DISTANCE_CACHE:
        _cache_put(_INTERNAL_DISTANCE_CACHE, key, cdist(x, x))
    return _INTERNAL_DISTANCE_CACHE[key]


def _pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cdist(_subsample(_as_cloud(a)), _subsample(_as_cloud(b)), metric="euclidean")


def _internal_radii(x: np.ndarray, k: int) -> np.ndarray:
    x = _subsample(_as_cloud(x))
    if len(x) < 2:
        return np.zeros(len(x), dtype=float)
    k_eff = min(max(int(k), 1), len(x) - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="brute").fit(x)
    distances, _ = nn.kneighbors(x)
    return distances[:, k_eff]


def chamfer_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = _pairwise(a, b)
    return -float(0.5 * (d.min(axis=1).mean() + d.min(axis=0).mean()))


def hausdorff_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = _pairwise(a, b)
    return -float(max(d.min(axis=1).max(), d.min(axis=0).max()))


def modified_hausdorff_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = _pairwise(a, b)
    return -float(max(d.min(axis=1).mean(), d.min(axis=0).mean()))


def percentile_hausdorff_similarity(a: np.ndarray, b: np.ndarray, q: float = 95.0) -> float:
    d = _pairwise(a, b)
    return -float(max(np.percentile(d.min(axis=1), q), np.percentile(d.min(axis=0), q)))


def point_fscore_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = _pairwise(a, b)
    precision = float(np.mean(d.min(axis=0) <= CONFIG.point_fscore_threshold))
    recall = float(np.mean(d.min(axis=1) <= CONFIG.point_fscore_threshold))
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def best_buddies_similarity(a: np.ndarray, b: np.ndarray) -> float:
    d = _pairwise(a, b)
    a_to_b = np.argmin(d, axis=1)
    b_to_a = np.argmin(d, axis=0)
    mutual = sum(int(b_to_a[j] == i) for i, j in enumerate(a_to_b))
    return float(mutual / max(min(d.shape), 1))


def coverage_similarity(a: np.ndarray, b: np.ndarray, k: int | None = None) -> float:
    """Symmetric Naeem et al. coverage; k=1 equals the project's overlap."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    k = CONFIG.manifold_k if k is None else k
    d = cdist(a, b)
    cov_a = np.mean(d.min(axis=1) <= _internal_radii(a, k)) if len(a) > 1 else 0.0
    cov_b = np.mean(d.min(axis=0) <= _internal_radii(b, k)) if len(b) > 1 else 0.0
    return float(0.5 * (cov_a + cov_b))


def manifold_pr_f1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Kynkaanniemi-style manifold precision/recall, summarized by F1."""
    precision, recall = improved_precision_recall(a, b)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def improved_precision_recall(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Original directed improved precision and recall (Kynkaanniemi et al.).

    ``a`` is the reference/real cloud and ``b`` the evaluated/generated cloud.
    """
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    d = cdist(a, b)
    ra = _internal_radii(a, CONFIG.manifold_k)
    rb = _internal_radii(b, CONFIG.manifold_k)
    # A point is in the opposite manifold if it lies in at least one adaptive ball.
    precision = float(np.mean(np.any(d <= ra[:, None], axis=0)))
    recall = float(np.mean(np.any(d <= rb[None, :], axis=1)))
    return precision, recall


def improved_precision_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return improved_precision_recall(a, b)[0]


def improved_recall_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return improved_precision_recall(a, b)[1]


def prdc_directional(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Precision, recall, density and coverage in Naeem et al.'s orientation."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    k = min(CONFIG.manifold_k, max(1, len(a) - 1), max(1, len(b) - 1))
    d = cdist(a, b)
    ra = _internal_radii(a, k)
    rb = _internal_radii(b, k)
    precision = float(np.mean(np.any(d <= ra[:, None], axis=0)))
    recall = float(np.mean(np.any(d <= rb[None, :], axis=1)))
    density = float(np.mean(np.sum(d <= ra[:, None], axis=0) / k))
    coverage = float(np.mean(d.min(axis=1) <= ra))
    return precision, recall, density, coverage


def prdc_precision_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return prdc_directional(a, b)[0]


def prdc_recall_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return prdc_directional(a, b)[1]


def prdc_density_directional_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return prdc_directional(a, b)[2]


def prdc_coverage_directional_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return prdc_directional(a, b)[3]


def prdc_f1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    precision, recall, _, _ = prdc_directional(a, b)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def prdc_density_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Symmetric version of PRDC density (Naeem et al., 2020)."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    d = cdist(a, b)
    k = min(CONFIG.manifold_k, max(1, len(a) - 1), max(1, len(b) - 1))
    ra = _internal_radii(a, k)
    rb = _internal_radii(b, k)
    density_b_in_a = np.mean(np.sum(d <= ra[:, None], axis=0) / k)
    density_a_in_b = np.mean(np.sum(d <= rb[None, :], axis=1) / k)
    return float(0.5 * (density_a_in_b + density_b_in_a))


def precision_recall_cover(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Precision Cover and Recall Cover (Cheema & Urner, 2023).

    The paper recommends k'=3 and C=3.  A point is covered when its C*k'-NN
    same-set ball contains at least k' points from the opposite cloud.
    """
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    k = max(1, int(CONFIG.prc_k))
    c = max(1, int(CONFIG.prc_c))
    ka = min(c * k, max(1, len(a) - 1))
    kb = min(c * k, max(1, len(b) - 1))
    threshold_a = min(k, len(b))
    threshold_b = min(k, len(a))
    radii_a = _internal_radii(a, ka)
    radii_b = _internal_radii(b, kb)
    d = cdist(a, b)
    cover_precision = float(np.mean(np.sum(d <= radii_a[:, None], axis=1) >= threshold_a))
    cover_recall = float(np.mean(np.sum(d <= radii_b[None, :], axis=0) >= threshold_b))
    return cover_precision, cover_recall


def prc_precision_cover_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return precision_recall_cover(a, b)[0]


def prc_recall_cover_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return precision_recall_cover(a, b)[1]


def prc_f1_similarity(a: np.ndarray, b: np.ndarray) -> float:
    precision, recall = precision_recall_cover(a, b)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def clipped_density_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Clipped Density (Salvy, Talbot & Thirion, 2026)."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    k = min(CONFIG.manifold_k, max(1, len(a) - 1))
    raw_radii = _internal_radii(a, k)
    radii = np.minimum(raw_radii, np.median(raw_radii))
    d_ab = cdist(a, b)
    generated_contrib = np.minimum(np.sum(d_ab <= radii[:, None], axis=0) / k, 1.0)
    unnormalized = float(generated_contrib.mean())

    d_aa = cdist(a, a)
    membership = d_aa <= radii[:, None]
    np.fill_diagonal(membership, False)
    real_contrib = np.minimum(membership.sum(axis=0) / k, 1.0)
    real_score = float(real_contrib.mean())
    return float(min(unnormalized / real_score, 1.0)) if real_score > 0 else 0.0


def _expected_clipped_coverage(n_real: int, m_good: int, k: int) -> float:
    if m_good <= 0:
        return 0.0
    j = np.arange(1, m_good + 1, dtype=float)
    log_comb = (
        gammaln(m_good + 1)
        - gammaln(j + 1)
        - gammaln(m_good - j + 1)
    )
    log_prob = log_comb + betaln(k + j, m_good - j + n_real - k) - betaln(k, n_real - k)
    log_weight = np.log(np.minimum(j / k, 1.0))
    return float(np.exp(logsumexp(log_prob + log_weight)))


def clipped_coverage_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calibrated Clipped Coverage (Salvy, Talbot & Thirion, 2026)."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    k = min(CONFIG.manifold_k, max(1, len(a) - 1))
    if len(a) <= k:
        return 0.0
    radii = _internal_radii(a, k)
    counts = np.sum(cdist(a, b) <= radii[:, None], axis=1)
    unnormalized = float(np.minimum(counts / k, 1.0).mean())
    expected = np.array(
        [_expected_clipped_coverage(len(a), m, k) for m in range(len(b) + 1)],
        dtype=float,
    )
    expected = np.maximum.accumulate(np.nan_to_num(expected, nan=0.0, posinf=1.0))
    insertion = int(np.searchsorted(expected, unnormalized, side="left"))
    return float(np.clip(insertion / max(len(b), 1), 0.0, 1.0))


def local_scaling_affinity(a: np.ndarray, b: np.ndarray) -> float:
    """Nearest-cross-set affinity with self-tuning local bandwidths."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    d = cdist(a, b)
    ra = np.maximum(_internal_radii(a, CONFIG.manifold_k), np.finfo(float).eps)
    rb = np.maximum(_internal_radii(b, CONFIG.manifold_k), np.finfo(float).eps)
    j = np.argmin(d, axis=1)
    i = np.argmin(d, axis=0)
    aff_a = np.exp(-(d[np.arange(len(a)), j] ** 2) / (ra * rb[j]))
    aff_b = np.exp(-(d[i, np.arange(len(b))] ** 2) / (ra[i] * rb))
    return float(0.5 * (aff_a.mean() + aff_b.mean()))


def dcd_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Density-aware Chamfer similarity following Wu et al. (NeurIPS 2021)."""
    d = _pairwise(a, b)
    j = np.argmin(d, axis=1)
    i = np.argmin(d, axis=0)
    count_b = np.maximum(np.bincount(j, minlength=d.shape[1]), 1)
    count_a = np.maximum(np.bincount(i, minlength=d.shape[0]), 1)
    a_term = np.exp(-CONFIG.dcd_alpha * d[np.arange(d.shape[0]), j] ** 2) / count_b[j]
    b_term = np.exp(-CONFIG.dcd_alpha * d[i, np.arange(d.shape[1])] ** 2) / count_a[i]
    # 1-DCD: bounded similarity, with 1 indicating an ideal match.
    return float(0.5 * (a_term.mean() + b_term.mean()))


def energy_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    cross = cdist(a, b).mean()
    within_a = cdist(a, a).mean()
    within_b = cdist(b, b).mean()
    return -float(max(2.0 * cross - within_a - within_b, 0.0))


def mmd_rbf_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    z = np.vstack([a, b])
    dz = cdist(z, z, metric="sqeuclidean")
    positive = dz[dz > 0]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
    bandwidth2 = max(bandwidth2, np.finfo(float).eps)
    kaa = np.exp(-cdist(a, a, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
    kbb = np.exp(-cdist(b, b, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
    kab = np.exp(-cdist(a, b, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
    return -float(max(kaa + kbb - 2.0 * kab, 0.0))


def kid_polynomial_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Negative unbiased polynomial-kernel MMD^2 (the KID estimator)."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    if len(a) < 2 or len(b) < 2:
        return math.nan
    dim = max(a.shape[1], 1)
    kaa = (a @ a.T / dim + 1.0) ** 3
    kbb = (b @ b.T / dim + 1.0) ** 3
    kab = (a @ b.T / dim + 1.0) ** 3
    mmd2 = (
        (kaa.sum() - np.trace(kaa)) / (len(a) * (len(a) - 1))
        + (kbb.sum() - np.trace(kbb)) / (len(b) * (len(b) - 1))
        - 2.0 * kab.mean()
    )
    return -float(mmd2)


def prd_f1_max_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Maximum F1 along Sajjadi et al.'s PRD curve."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    z = np.vstack([a, b])
    clusters = min(CONFIG.prd_clusters, max(2, len(z) // 2), len(z))
    labels = MiniBatchKMeans(
        n_clusters=clusters,
        n_init=10,
        random_state=0,
        batch_size=max(32, len(z)),
    ).fit_predict(z)
    p = np.bincount(labels[: len(a)], minlength=clusters).astype(float)
    q = np.bincount(labels[len(a) :], minlength=clusters).astype(float)
    p /= max(p.sum(), 1.0)
    q /= max(q.sum(), 1.0)
    eps = 1e-10
    slopes = np.tan(np.linspace(eps, np.pi / 2.0 - eps, CONFIG.prd_angles))
    precision = np.minimum(p[None, :] * slopes[:, None], q[None, :]).sum(axis=1)
    recall = precision / slopes
    precision = np.clip(precision, 0.0, 1.0)
    recall = np.clip(recall, 0.0, 1.0)
    denom = precision + recall
    f1 = np.divide(2.0 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(f1.max(initial=0.0))


def c2st_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Classifier two-sample similarity: 1 at chance, 0 at perfect separation."""
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    folds = min(CONFIG.c2st_folds, len(a), len(b))
    if folds < 2:
        return math.nan
    x = np.vstack([a, b])
    y = np.r_[np.zeros(len(a), dtype=int), np.ones(len(b), dtype=int)]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, solver="liblinear", random_state=0),
    )
    accuracy = float(cross_val_score(model, x, y, cv=cv, scoring="accuracy").mean())
    return float(np.clip(1.0 - 2.0 * abs(accuracy - 0.5), 0.0, 1.0))


def _equal_length_clouds(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    n = min(len(a), len(b))
    return (
        a[np.linspace(0, len(a) - 1, n, dtype=int)],
        b[np.linspace(0, len(b) - 1, n, dtype=int)],
    )


def alpha_beta_authenticity(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Alaa et al.'s alpha-precision, beta-recall and authenticity summaries."""
    a, b = _equal_length_clouds(a, b)
    if len(a) < 2:
        return math.nan, math.nan, math.nan
    center_a = a.mean(axis=0)
    center_b = b.mean(axis=0)
    levels = np.linspace(0.0, 1.0, CONFIG.alpha_beta_steps)
    radii = np.quantile(np.linalg.norm(a - center_a, axis=1), levels)
    b_to_center_a = np.linalg.norm(b - center_a, axis=1)

    nn_a = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(a)
    a_to_a = nn_a.kneighbors(a, return_distance=True)[0][:, 1]
    nn_b = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(b)
    a_to_b, a_to_b_idx = nn_b.kneighbors(a, return_distance=True)
    a_to_b = a_to_b[:, 0]
    a_to_b_idx = a_to_b_idx[:, 0]
    closest_b_center_distance = np.linalg.norm(b[a_to_b_idx] - center_b, axis=1)
    closest_b_radii = np.quantile(closest_b_center_distance, levels)

    alpha_curve = np.array([np.mean(b_to_center_a <= radius) for radius in radii])
    beta_curve = np.array(
        [
            np.mean((a_to_b <= a_to_a) & (closest_b_center_distance <= radius))
            for radius in closest_b_radii
        ]
    )
    normalizer = max(float(levels.sum()), np.finfo(float).eps)
    alpha_score = 1.0 - float(np.abs(levels - alpha_curve).sum()) / normalizer
    beta_score = 1.0 - float(np.abs(levels - beta_curve).sum()) / normalizer
    authenticity = float(np.mean(a_to_a[a_to_b_idx] < a_to_b))
    return (
        float(np.clip(alpha_score, 0.0, 1.0)),
        float(np.clip(beta_score, 0.0, 1.0)),
        float(np.clip(authenticity, 0.0, 1.0)),
    )


def alpha_precision_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return alpha_beta_authenticity(a, b)[0]


def beta_recall_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return alpha_beta_authenticity(a, b)[1]


def authenticity_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return alpha_beta_authenticity(a, b)[2]


def gaussian_frechet_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    ca = np.atleast_2d(np.cov(a, rowvar=False))
    cb = np.atleast_2d(np.cov(b, rowvar=False))
    covmean = sqrtm(ca @ cb)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = np.sum((ma - mb) ** 2) + np.trace(ca + cb - 2.0 * covmean)
    return -float(max(value, 0.0))


def sliced_wasserstein_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(CONFIG.sliced_projections, a.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), np.finfo(float).eps)
    values = [wasserstein_1d(a @ u, b @ u) for u in directions]
    return -float(np.mean(values))


def emd_wasserstein_similarity(a: np.ndarray, b: np.ndarray) -> float:
    import ot

    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    cost = cdist(a, b)
    wa = np.full(len(a), 1.0 / len(a))
    wb = np.full(len(b), 1.0 / len(b))
    return -float(ot.emd2(wa, wb, cost, numItermax=100000))


def sinkhorn_divergence_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Negative debiased entropic OT (Sinkhorn divergence)."""
    import ot

    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    cost_ab = cdist(a, b)
    cost_aa = cdist(a, a)
    cost_bb = cdist(b, b)
    positive = np.concatenate(
        [cost_ab[cost_ab > 0], cost_aa[cost_aa > 0], cost_bb[cost_bb > 0]]
    )
    scale = float(np.median(positive)) if len(positive) else 1.0
    reg = max(CONFIG.sinkhorn_reg * scale, np.finfo(float).eps)
    wa = np.full(len(a), 1.0 / len(a))
    wb = np.full(len(b), 1.0 / len(b))

    def regularized_ot(xw, yw, cost):
        return float(
            ot.sinkhorn2(
                xw,
                yw,
                cost,
                reg,
                method="sinkhorn_log",
                numItermax=2000,
                stopThr=1e-8,
            )
        )

    value = regularized_ot(wa, wb, cost_ab)
    value -= 0.5 * regularized_ot(wa, wa, cost_aa)
    value -= 0.5 * regularized_ot(wb, wb, cost_bb)
    return -float(max(value, 0.0))


def gromov_wasserstein_similarity(a: np.ndarray, b: np.ndarray) -> float:
    import ot

    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    ca = _internal_distances(a)
    cb = _internal_distances(b)
    wa = np.full(len(a), 1.0 / len(a))
    wb = np.full(len(b), 1.0 / len(b))
    value = ot.gromov.gromov_wasserstein2(ca, cb, wa, wb, loss_fun="square_loss", log=False)
    return -float(value)


def persistence_bottleneck_similarity(a: np.ndarray, b: np.ndarray) -> float:
    from persim import bottleneck
    from ripser import ripser

    a = _subsample(_as_cloud(a))
    b = _subsample(_as_cloud(b))
    maxdim = 1 if min(len(a), len(b)) >= 4 else 0
    def diagrams(x: np.ndarray) -> list[np.ndarray]:
        key = (_cloud_key(x), maxdim)
        if key not in _PERSISTENCE_CACHE:
            value = ripser(_internal_distances(x), maxdim=maxdim, distance_matrix=True)["dgms"]
            _cache_put(_PERSISTENCE_CACHE, key, value)
        return _PERSISTENCE_CACHE[key]

    da = diagrams(a)
    db = diagrams(b)
    values = []
    for dim in range(maxdim + 1):
        xa = da[dim][np.isfinite(da[dim]).all(axis=1)]
        xb = db[dim][np.isfinite(db[dim]).all(axis=1)]
        values.append(float(bottleneck(xa, xb)))
    return -float(max(values, default=0.0))


def _relative_living_times(intervals: np.ndarray, alpha_max: float, i_max: int) -> np.ndarray:
    out = np.zeros(i_max, dtype=float)
    if alpha_max <= 0:
        out[0] = 1.0
        return out
    finite = []
    for birth, death in np.asarray(intervals, dtype=float):
        finite.append((float(birth), alpha_max if not np.isfinite(death) else min(float(death), alpha_max)))
    finite = [(birth, death) for birth, death in finite if death > birth]
    if not finite:
        out[0] = 1.0
        return out
    switches = np.unique(np.r_[0.0, alpha_max, np.asarray(finite).ravel()])
    switches = switches[(switches >= 0.0) & (switches <= alpha_max)]
    for left, right in zip(switches[:-1], switches[1:]):
        midpoint = 0.5 * (left + right)
        holes = sum(birth <= midpoint < death for birth, death in finite)
        if holes < i_max:
            out[holes] += right - left
    return out / alpha_max


def _geometry_mrlt(x: np.ndarray) -> np.ndarray:
    import gudhi

    x = _subsample(_as_cloud(x))
    landmark_count = min(CONFIG.geometry_landmarks, len(x))
    rng = np.random.default_rng(0)
    values = []
    for _ in range(CONFIG.geometry_repeats):
        indices = rng.choice(len(x), size=landmark_count, replace=False)
        landmarks = x[indices]
        distances = cdist(x, landmarks)
        order = np.argsort(distances, axis=1)
        sorted_distances = np.take_along_axis(distances, order, axis=1)
        table = np.dstack([order, sorted_distances])
        alpha_max = float(distances.max() * CONFIG.geometry_gamma)
        complex_ = gudhi.WitnessComplex(table)
        tree = complex_.create_simplex_tree(max_alpha_square=alpha_max, limit_dimension=2)
        tree.persistence(homology_coeff_field=2)
        intervals = tree.persistence_intervals_in_dimension(1)
        values.append(_relative_living_times(intervals, alpha_max, CONFIG.geometry_i_max))
    return np.mean(values, axis=0)


def geometry_score_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Negative Geometry Score using witness-complex MRLT distributions.

    This optional baseline requires GUDHI.  The configurable repeat/landmark
    budget is recorded by the main runner for reproducible efficiency studies.
    """
    mrlt_a = _geometry_mrlt(a)
    mrlt_b = _geometry_mrlt(b)
    return -float(np.sum((mrlt_a - mrlt_b) ** 2))


METRIC_FUNCTIONS = {
    "chamfer_similarity": chamfer_similarity,
    "hausdorff_similarity": hausdorff_similarity,
    "modified_hausdorff_similarity": modified_hausdorff_similarity,
    "hausdorff95_similarity": percentile_hausdorff_similarity,
    "point_fscore_global": point_fscore_similarity,
    "best_buddies_similarity": best_buddies_similarity,
    "coverage_sym_k5": coverage_similarity,
    "manifold_pr_f1": manifold_pr_f1_similarity,
    "improved_precision": improved_precision_similarity,
    "improved_recall": improved_recall_similarity,
    "prdc_precision": prdc_precision_similarity,
    "prdc_recall": prdc_recall_similarity,
    "prdc_density": prdc_density_directional_similarity,
    "prdc_coverage": prdc_coverage_directional_similarity,
    "prdc_f1": prdc_f1_similarity,
    "prdc_density_sym": prdc_density_similarity,
    "prc_precision_cover": prc_precision_cover_similarity,
    "prc_recall_cover": prc_recall_cover_similarity,
    "prc_f1": prc_f1_similarity,
    "clipped_density": clipped_density_similarity,
    "clipped_coverage": clipped_coverage_similarity,
    "local_scaling_affinity": local_scaling_affinity,
    "dcd_similarity": dcd_similarity,
    "energy_similarity": energy_similarity,
    "mmd_rbf_similarity": mmd_rbf_similarity,
    "kid_polynomial_similarity": kid_polynomial_similarity,
    "prd_f1_max": prd_f1_max_similarity,
    "c2st_similarity": c2st_similarity,
    "alpha_precision_similarity": alpha_precision_similarity,
    "beta_recall_similarity": beta_recall_similarity,
    "authenticity_similarity": authenticity_similarity,
    "gaussian_frechet_similarity": gaussian_frechet_similarity,
    "sliced_wasserstein_similarity": sliced_wasserstein_similarity,
    "emd_wasserstein_similarity": emd_wasserstein_similarity,
    "sinkhorn_divergence_similarity": sinkhorn_divergence_similarity,
    "gromov_wasserstein_similarity": gromov_wasserstein_similarity,
    "persistence_bottleneck_similarity": persistence_bottleneck_similarity,
    "geometry_score_similarity": geometry_score_similarity,
}


def compute_pointset_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, fn in METRIC_FUNCTIONS.items():
        try:
            value = float(fn(a, b))
            scores[name] = value if math.isfinite(value) else math.nan
        except Exception:
            scores[name] = math.nan
    return scores

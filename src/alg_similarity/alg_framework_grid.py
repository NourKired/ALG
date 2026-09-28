"""Composable Adaptive Local--Global (ALG) similarity framework.

The framework separates three choices:

1. a bounded local agreement ``L(A, B)``;
2. a bounded global agreement ``G(A, B)``;
3. a fusion weight ``lambda``.

The expanded research grid contains 64 local specifications, 240 global
specifications and 11 fusion weights, for 168,960 configurations. Expensive
local/global components are computed once and scalar fusions are derived from
the cached component values.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Iterable, Iterator

import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


GEOMETRIES = ("euclidean", "manhattan", "chebyshev", "cosine")
SCIPY_METRIC = {
    "euclidean": "euclidean",
    "manhattan": "cityblock",
    "chebyshev": "chebyshev",
    "cosine": "cosine",
}
LOCAL_KS = (1, 3, 5, 10)
# Keep k=1 in Manifold PR-F1.  It was inadvertently absent from the completed
# 158,400-configuration campaign even though k=1 is meaningful and is used by
# the initial nearest-neighbour formulation.  With this value the full grid is
# 64 x 240 x 11 = 168,960 configurations.
MANIFOLD_KS = (1, 3, 5, 10)
DCD_ALPHAS = (10.0, 50.0, 100.0)
PSI_BY_GEOMETRY = {
    "euclidean": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "manhattan": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "chebyshev": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "cosine": ("linear", "rational", "cauchy", "exponential", "gaussian", "logistic"),
}
SCALE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = tuple(round(float(value), 1) for value in np.linspace(0.0, 1.0, 11))


@dataclass(frozen=True)
class LocalSpec:
    family: str
    geometry: str
    k: int | None = None
    alpha: float | None = None

    @property
    def slug(self) -> str:
        fields = [self.family, self.geometry]
        if self.k is not None:
            fields.append(f"k{self.k}")
        if self.alpha is not None:
            fields.append(f"alpha{self.alpha:g}")
        return "_".join(fields)


@dataclass(frozen=True)
class GlobalSpec:
    family: str
    geometry: str
    psi: str
    scale: float

    @property
    def slug(self) -> str:
        return f"{self.family}_{self.geometry}_{self.psi}_scale{self.scale:g}"


@dataclass(frozen=True)
class ALGFrameworkSpec:
    local: LocalSpec
    global_: GlobalSpec
    lambda_: float

    @property
    def metric_name(self) -> str:
        return f"algfw_L-{self.local.slug}_G-{self.global_.slug}_lambda{self.lambda_:.1f}"

    def as_record(self) -> dict[str, object]:
        return {
            "metric": self.metric_name,
            "local_family": self.local.family,
            "local_geometry": self.local.geometry,
            "local_k": self.local.k,
            "local_alpha": self.local.alpha,
            "global_family": self.global_.family,
            "global_geometry": self.global_.geometry,
            "global_psi": self.global_.psi,
            "global_scale": self.global_.scale,
            "lambda": self.lambda_,
        }


def local_grid_specs() -> tuple[LocalSpec, ...]:
    """Return the 64 bounded local-agreement options."""
    specs: list[LocalSpec] = []
    for geometry in GEOMETRIES:
        specs.extend(LocalSpec("coverage", geometry, k=k) for k in LOCAL_KS)
        specs.extend(LocalSpec("manifold_f1", geometry, k=k) for k in MANIFOLD_KS)
        specs.extend(LocalSpec("local_scaling", geometry, k=k) for k in LOCAL_KS)
        specs.append(LocalSpec("best_buddies", geometry))
        specs.extend(LocalSpec("dcd", geometry, alpha=alpha) for alpha in DCD_ALPHAS)
    return tuple(specs)


def _global_raw_specs() -> tuple[tuple[str, str], ...]:
    return (
        *(("centroid", geometry) for geometry in GEOMETRIES),
        ("energy", "euclidean"),
        ("mmd_rbf", "euclidean"),
        ("gaussian_frechet", "euclidean"),
        ("sliced_wasserstein", "euclidean"),
    )


def global_grid_specs() -> tuple[GlobalSpec, ...]:
    """Return 240 normalized global-agreement options."""
    return tuple(
        GlobalSpec(family, geometry, psi, scale)
        for family, geometry in _global_raw_specs()
        for psi in PSI_BY_GEOMETRY[geometry]
        for scale in SCALE_MULTIPLIERS
    )


def iter_framework_specs(
    local_specs: Iterable[LocalSpec] | None = None,
    global_specs: Iterable[GlobalSpec] | None = None,
    lambdas: Iterable[float] = LAMBDAS,
) -> Iterator[ALGFrameworkSpec]:
    locals_ = tuple(local_grid_specs() if local_specs is None else local_specs)
    globals_ = tuple(global_grid_specs() if global_specs is None else global_specs)
    for local in locals_:
        for global_ in globals_:
            for lambda_ in lambdas:
                yield ALGFrameworkSpec(local, global_, float(lambda_))


def framework_grid_size() -> int:
    return len(local_grid_specs()) * len(global_grid_specs()) * len(LAMBDAS)


def _as_cloud_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1] or min(len(a), len(b)) < 2:
        raise ValueError("ALG expects two finite 2-D clouds with equal feature dimension and at least two points")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("ALG clouds must contain only finite values")
    return a, b


def _safe_distance_matrix(a: np.ndarray, b: np.ndarray, geometry: str) -> np.ndarray:
    distance = cdist(a, b, metric=SCIPY_METRIC[geometry])
    if geometry == "cosine":
        distance = np.nan_to_num(distance, nan=1.0, posinf=2.0, neginf=0.0)
    return distance


def _internal_radii(distance: np.ndarray, k: int) -> np.ndarray:
    if distance.shape[0] < 2:
        return np.zeros(distance.shape[0], dtype=float)
    k_eff = min(max(int(k), 1), distance.shape[0] - 1)
    without_self = distance.copy()
    np.fill_diagonal(without_self, np.inf)
    return np.partition(without_self, k_eff - 1, axis=1)[:, k_eff - 1]


def _harmonic(left: float, right: float) -> float:
    return 2.0 * left * right / (left + right) if left + right else 0.0


def _local_score(
    spec: LocalSpec,
    d_aa: np.ndarray,
    d_bb: np.ndarray,
    d_ab: np.ndarray,
) -> float:
    nearest_a_to_b = d_ab.min(axis=1)
    nearest_b_to_a = d_ab.min(axis=0)

    if spec.family in {"coverage", "manifold_f1", "local_scaling"}:
        if spec.k is None:
            raise ValueError(f"{spec.family} requires k")
        radius_a = np.maximum(_internal_radii(d_aa, spec.k), np.finfo(float).eps)
        radius_b = np.maximum(_internal_radii(d_bb, spec.k), np.finfo(float).eps)

    if spec.family == "coverage":
        a_to_b = float(np.mean(nearest_a_to_b <= radius_a))
        b_to_a = float(np.mean(nearest_b_to_a <= radius_b))
        return 0.5 * (a_to_b + b_to_a)

    if spec.family == "manifold_f1":
        precision = float(np.mean(np.any(d_ab <= radius_a[:, None], axis=0)))
        recall = float(np.mean(np.any(d_ab <= radius_b[None, :], axis=1)))
        return _harmonic(precision, recall)

    if spec.family == "local_scaling":
        nearest_b = np.argmin(d_ab, axis=1)
        nearest_a = np.argmin(d_ab, axis=0)
        affinity_a = np.exp(
            -(d_ab[np.arange(len(d_ab)), nearest_b] ** 2)
            / np.maximum(radius_a * radius_b[nearest_b], np.finfo(float).eps)
        )
        affinity_b = np.exp(
            -(d_ab[nearest_a, np.arange(d_ab.shape[1])] ** 2)
            / np.maximum(radius_a[nearest_a] * radius_b, np.finfo(float).eps)
        )
        return float(0.5 * (affinity_a.mean() + affinity_b.mean()))

    if spec.family == "best_buddies":
        a_to_b = np.argmin(d_ab, axis=1)
        b_to_a = np.argmin(d_ab, axis=0)
        mutual = sum(int(b_to_a[index_b] == index_a) for index_a, index_b in enumerate(a_to_b))
        return float(mutual / max(min(d_ab.shape), 1))

    if spec.family == "dcd":
        if spec.alpha is None:
            raise ValueError("dcd requires alpha")
        a_to_b = np.argmin(d_ab, axis=1)
        b_to_a = np.argmin(d_ab, axis=0)
        count_b = np.maximum(np.bincount(a_to_b, minlength=d_ab.shape[1]), 1)
        count_a = np.maximum(np.bincount(b_to_a, minlength=d_ab.shape[0]), 1)
        term_a = np.exp(-spec.alpha * d_ab[np.arange(d_ab.shape[0]), a_to_b] ** 2) / count_b[a_to_b]
        term_b = np.exp(-spec.alpha * d_ab[b_to_a, np.arange(d_ab.shape[1])] ** 2) / count_a[b_to_a]
        return float(0.5 * (term_a.mean() + term_b.mean()))

    raise KeyError(spec.family)


def _psi(name: str, value: float) -> float:
    z = max(float(value), 0.0)
    if name == "linear":
        return max(0.0, 1.0 - 0.5 * z)
    if name == "affine_clipped":
        return max(0.0, 1.0 - z)
    if name == "rational":
        return 1.0 / (1.0 + z)
    if name == "cauchy":
        return 1.0 / (1.0 + z * z)
    if name == "exponential":
        return math.exp(-z)
    if name == "gaussian":
        return math.exp(-0.5 * z * z)
    if name == "logistic":
        return 2.0 / (1.0 + math.exp(min(z, 700.0)))
    raise KeyError(name)


def _global_raw_distance(
    family: str,
    geometry: str,
    a: np.ndarray,
    b: np.ndarray,
    distance_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sliced_projections: int,
) -> tuple[float, float]:
    d_aa, d_bb, d_ab = distance_cache[geometry]
    base_scale = float(
        np.median(np.concatenate([_internal_radii(d_aa, 1), _internal_radii(d_bb, 1)]))
    )
    base_scale = max(base_scale, np.finfo(float).eps)

    if family == "centroid":
        raw = float(_safe_distance_matrix(a.mean(axis=0, keepdims=True), b.mean(axis=0, keepdims=True), geometry)[0, 0])
        return raw, 1.0 if geometry == "cosine" else base_scale

    if family == "energy":
        raw = max(2.0 * float(d_ab.mean()) - float(d_aa.mean()) - float(d_bb.mean()), 0.0)
        return raw, base_scale

    if family == "mmd_rbf":
        squared = cdist(np.vstack([a, b]), np.vstack([a, b]), metric="sqeuclidean")
        positive = squared[squared > 0]
        bandwidth2 = max(float(np.median(positive)) if len(positive) else 1.0, np.finfo(float).eps)
        kaa = np.exp(-cdist(a, a, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
        kbb = np.exp(-cdist(b, b, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
        kab = np.exp(-cdist(a, b, metric="sqeuclidean") / (2.0 * bandwidth2)).mean()
        return max(float(kaa + kbb - 2.0 * kab), 0.0), 1.0

    if family == "gaussian_frechet":
        mean_a, mean_b = a.mean(axis=0), b.mean(axis=0)
        covariance_a = np.atleast_2d(np.cov(a, rowvar=False))
        covariance_b = np.atleast_2d(np.cov(b, rowvar=False))
        # Preserve the historical implementation whenever it is finite so
        # resumed shards remain bit-for-bit methodologically compatible.
        direct_raw = math.nan
        try:
            covariance_mean = sqrtm(covariance_a @ covariance_b)
            if np.iscomplexobj(covariance_mean):
                covariance_mean = covariance_mean.real
            direct_raw = float(
                np.sum((mean_a - mean_b) ** 2)
                + np.trace(covariance_a + covariance_b - 2.0 * covariance_mean)
            )
        except Exception:
            # The stable Bures branch below is the intended fallback.
            direct_raw = math.nan
        if math.isfinite(direct_raw):
            return max(direct_raw, 0.0), base_scale * base_scale

        # sqrtm(C_a C_b) is numerically fragile when point clouds have fewer
        # samples than dimensions. The equivalent symmetric Bures expression
        # below is the fallback for rank-deficient PSD covariances. A common
        # rescaling improves conditioning and cancels in raw/reference_scale.
        joint_scale = max(
            float(np.max(np.abs(a))) if a.size else 0.0,
            float(np.max(np.abs(b))) if b.size else 0.0,
            1.0,
        )
        safe_a = a / joint_scale
        safe_b = b / joint_scale
        mean_a, mean_b = safe_a.mean(axis=0), safe_b.mean(axis=0)
        covariance_a = np.atleast_2d(np.cov(safe_a, rowvar=False))
        covariance_b = np.atleast_2d(np.cov(safe_b, rowvar=False))
        covariance_a = 0.5 * (covariance_a + covariance_a.T)
        covariance_b = 0.5 * (covariance_b + covariance_b.T)
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(covariance_a)
        eigenvalues_b, eigenvectors_b = np.linalg.eigh(covariance_b)
        eigenvalues_a = np.clip(eigenvalues_a, 0.0, None)
        eigenvalues_b = np.clip(eigenvalues_b, 0.0, None)
        covariance_b = (eigenvectors_b * eigenvalues_b) @ eigenvectors_b.T
        sqrt_a = (eigenvectors_a * np.sqrt(eigenvalues_a)) @ eigenvectors_a.T
        middle = sqrt_a @ covariance_b @ sqrt_a
        middle = 0.5 * (middle + middle.T)
        middle_eigenvalues = np.linalg.eigvalsh(middle)
        if not np.isfinite(middle_eigenvalues).all():
            raise FloatingPointError("non-finite Bures covariance spectrum")
        # Both covariance matrices have already been projected to the PSD
        # cone. Any remaining negative eigenvalue is roundoff from the matrix
        # products/eigensolver and is therefore projected back to zero.
        middle_eigenvalues = np.maximum(middle_eigenvalues, 0.0)
        trace_covariance_mean = float(np.sqrt(middle_eigenvalues).sum())
        raw = float(
            np.sum((mean_a - mean_b) ** 2)
            + np.trace(covariance_a)
            + np.trace(covariance_b)
            - 2.0 * trace_covariance_mean
        )
        if not math.isfinite(raw):
            raise FloatingPointError("non-finite Gaussian Frechet distance")
        safe_d_aa = _safe_distance_matrix(safe_a, safe_a, "euclidean")
        safe_d_bb = _safe_distance_matrix(safe_b, safe_b, "euclidean")
        safe_reference_scale = float(
            np.median(
                np.concatenate(
                    [_internal_radii(safe_d_aa, 1), _internal_radii(safe_d_bb, 1)]
                )
            )
        )
        safe_reference_scale = max(safe_reference_scale, np.finfo(float).eps)
        return max(raw, 0.0), safe_reference_scale * safe_reference_scale

    if family == "sliced_wasserstein":
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(sliced_projections, a.shape[1]))
        directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), np.finfo(float).eps)
        raw = float(np.mean([wasserstein_distance(a @ direction, b @ direction) for direction in directions]))
        return raw, base_scale

    raise KeyError(family)


def alg_framework_components(
    a: np.ndarray,
    b: np.ndarray,
    local_specs: Iterable[LocalSpec] | None = None,
    global_specs: Iterable[GlobalSpec] | None = None,
    sliced_projections: int = 32,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Compute reusable local and normalized global component scores."""
    started = time.perf_counter()
    a, b = _as_cloud_pair(a, b)
    locals_ = tuple(local_grid_specs() if local_specs is None else local_specs)
    globals_ = tuple(global_grid_specs() if global_specs is None else global_specs)
    required_geometries = {spec.geometry for spec in locals_} | {spec.geometry for spec in globals_}

    distance_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, float] = {}
    for geometry in required_geometries:
        geometry_started = time.perf_counter()
        distance_cache[geometry] = (
            _safe_distance_matrix(a, a, geometry),
            _safe_distance_matrix(b, b, geometry),
            _safe_distance_matrix(a, b, geometry),
        )
        diagnostics[f"time_algfw_distance_{geometry}_sec"] = time.perf_counter() - geometry_started

    local_scores: dict[str, float] = {}
    for spec in locals_:
        local_started = time.perf_counter()
        local_scores[spec.slug] = float(_local_score(spec, *distance_cache[spec.geometry]))
        diagnostics[f"time_algfw_local__{spec.slug}_sec"] = time.perf_counter() - local_started

    raw_cache: dict[tuple[str, str], tuple[float, float]] = {}
    global_scores: dict[str, float] = {}
    for spec in globals_:
        raw_key = (spec.family, spec.geometry)
        if raw_key not in raw_cache:
            raw_started = time.perf_counter()
            raw_cache[raw_key] = _global_raw_distance(
                spec.family,
                spec.geometry,
                a,
                b,
                distance_cache,
                sliced_projections,
            )
            diagnostics[
                f"time_algfw_global_raw__{spec.family}_{spec.geometry}_sec"
            ] = time.perf_counter() - raw_started
        raw_distance, reference_scale = raw_cache[raw_key]
        denominator = max(spec.scale * reference_scale, np.finfo(float).eps)
        normalization_started = time.perf_counter()
        global_scores[spec.slug] = _psi(spec.psi, raw_distance / denominator)
        diagnostics[
            f"time_algfw_global_normalization__{spec.slug}_sec"
        ] = time.perf_counter() - normalization_started

    diagnostics["time_algfw_components_sec"] = time.perf_counter() - started
    diagnostics["algfw_n_local_components"] = float(len(local_scores))
    diagnostics["algfw_n_global_components"] = float(len(global_scores))
    return local_scores, global_scores, diagnostics


def fuse_framework_components(
    local_scores: dict[str, float],
    global_scores: dict[str, float],
    specs: Iterable[ALGFrameworkSpec] | None = None,
) -> dict[str, float]:
    """Fuse cached components for any subset of framework configurations."""
    configurations = iter_framework_specs() if specs is None else iter(specs)
    values: dict[str, float] = {}
    for spec in configurations:
        local = local_scores[spec.local.slug]
        global_ = global_scores[spec.global_.slug]
        values[spec.metric_name] = spec.lambda_ * local + (1.0 - spec.lambda_) * global_
    return values


def iter_framework_score_chunks(
    local_scores: dict[str, float],
    global_scores: dict[str, float],
    specs: Iterable[ALGFrameworkSpec] | None = None,
    chunk_size: int = 4096,
) -> Iterator[tuple[list[ALGFrameworkSpec], np.ndarray]]:
    """Yield fused scores in bounded-memory chunks for large grid searches."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    configurations = iter_framework_specs() if specs is None else iter(specs)
    batch: list[ALGFrameworkSpec] = []
    for spec in configurations:
        batch.append(spec)
        if len(batch) == chunk_size:
            yield batch, np.asarray(
                [
                    item.lambda_ * local_scores[item.local.slug]
                    + (1.0 - item.lambda_) * global_scores[item.global_.slug]
                    for item in batch
                ],
                dtype=float,
            )
            batch = []
    if batch:
        yield batch, np.asarray(
            [
                item.lambda_ * local_scores[item.local.slug]
                + (1.0 - item.lambda_) * global_scores[item.global_.slug]
                for item in batch
            ],
            dtype=float,
        )


def alg_framework_scores(
    a: np.ndarray,
    b: np.ndarray,
    specs: Iterable[ALGFrameworkSpec] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute requested framework scores and reusable timing diagnostics."""
    configurations = tuple(iter_framework_specs() if specs is None else specs)
    local_specs = tuple(dict.fromkeys(spec.local for spec in configurations))
    global_specs = tuple(dict.fromkeys(spec.global_ for spec in configurations))
    local_scores, global_scores, diagnostics = alg_framework_components(
        a,
        b,
        local_specs=local_specs,
        global_specs=global_specs,
    )
    fusion_started = time.perf_counter()
    values = fuse_framework_components(local_scores, global_scores, configurations)
    diagnostics["time_algfw_fusion_sec"] = time.perf_counter() - fusion_started
    diagnostics["algfw_n_fused_configurations"] = float(len(values))
    return values, diagnostics


assert len(local_grid_specs()) == 64
assert len(global_grid_specs()) == 240
assert framework_grid_size() == 168_960

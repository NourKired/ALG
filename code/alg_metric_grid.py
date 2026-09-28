"""Shared, vector-free ALG hyperparameter grid for point-cloud pairs.

The expensive neighbourhood computations are performed once per geometry.
The 1,320 scalar variants are then derived from those shared components.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


GEOMETRIES = ("euclidean", "manhattan", "chebyshev", "cosine")
PSI_BY_GEOMETRY = {
    "euclidean": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "manhattan": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "chebyshev": ("affine_clipped", "rational", "cauchy", "exponential", "gaussian", "logistic"),
    "cosine": ("linear", "rational", "cauchy", "exponential", "gaussian", "logistic"),
}
SCALE_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMBDAS = tuple(float(x) for x in np.linspace(0.0, 1.0, 11))


def alg_grid_metric_names() -> list[str]:
    return [
        f"alg_{geometry}_{psi}_scale_{scale:g}_lambda_{lam:.1f}"
        for geometry in GEOMETRIES
        for psi in PSI_BY_GEOMETRY[geometry]
        for scale in SCALE_MULTIPLIERS
        for lam in LAMBDAS
    ]


def _local_components(x: np.ndarray, y: np.ndarray, metric: str) -> tuple[float, float]:
    nn_x = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="brute").fit(x)
    nn_y = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="brute").fit(y)
    radius_x = nn_x.kneighbors(x)[0][:, 1]
    radius_y = nn_y.kneighbors(y)[0][:, 1]
    cross_x = nn_y.kneighbors(x, n_neighbors=1)[0][:, 0]
    cross_y = nn_x.kneighbors(y, n_neighbors=1)[0][:, 0]
    local = 0.5 * (np.mean(cross_x <= radius_x) + np.mean(cross_y <= radius_y))
    # Symmetric, data-internal scale. This is not fitted from pair labels.
    scale = np.median(np.concatenate([radius_x, radius_y]))
    return float(local), max(float(scale), np.finfo(float).eps)


def _psi(name: str, z: float) -> float:
    if name == "linear":
        # Cosine distance spans [0, 2]; divide by two before clipping.
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


def alg_grid_scores(x: np.ndarray, y: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    """Return all grid scores plus reusable component/timing diagnostics."""
    started = time.perf_counter()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1] or min(len(x), len(y)) < 2:
        raise ValueError("ALG expects two finite 2-D clouds with the same feature dimension")

    values: dict[str, float] = {}
    diagnostics: dict[str, float] = {}
    centroids = (x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True))
    for geometry, sklearn_metric, scipy_metric in (
        ("euclidean", "euclidean", "euclidean"),
        ("manhattan", "manhattan", "cityblock"),
        ("chebyshev", "chebyshev", "chebyshev"),
        ("cosine", "cosine", "cosine"),
    ):
        geometry_started = time.perf_counter()
        local, internal_scale = _local_components(x, y, sklearn_metric)
        distance = float(cdist(*centroids, metric=scipy_metric)[0, 0])
        if not math.isfinite(distance):
            distance = 1.0 if geometry == "cosine" else math.inf
        diagnostics[f"alg_component_{geometry}_local"] = local
        diagnostics[f"alg_component_{geometry}_global_distance"] = distance
        diagnostics[f"alg_component_{geometry}_internal_scale"] = internal_scale
        for multiplier in SCALE_MULTIPLIERS:
            # Cosine distance is already dimensionless. Other geometries use
            # the median within-set nearest-neighbour distance as local unit.
            denominator = multiplier if geometry == "cosine" else multiplier * internal_scale
            z = distance / max(denominator, np.finfo(float).eps)
            for psi_name in PSI_BY_GEOMETRY[geometry]:
                global_similarity = _psi(psi_name, z)
                for lam in LAMBDAS:
                    name = f"alg_{geometry}_{psi_name}_scale_{multiplier:g}_lambda_{lam:.1f}"
                    values[name] = lam * local + (1.0 - lam) * global_similarity
        diagnostics[f"time_alg_{geometry}_family_sec"] = time.perf_counter() - geometry_started
    diagnostics["time_alg_grid_sec"] = time.perf_counter() - started
    return values, diagnostics


assert len(alg_grid_metric_names()) == 1320

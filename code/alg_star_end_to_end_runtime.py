#!/usr/bin/env python3
"""Isolated, cache-free end-to-end implementations of ALG* and Initial ALG."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

from alg_framework_grid import ALGFrameworkSpec, GlobalSpec, LocalSpec


ALG_STAR_METRIC = "alg_star_end_to_end"
INITIAL_ALG_METRIC = "initial_alg_end_to_end"
ALG_STAR_NAME = (
    "algfw_L-local_scaling_manhattan_k3_"
    "G-centroid_cosine_exponential_scale2_lambda0.7"
)
ALG_STAR_SPEC = ALGFrameworkSpec(
    local=LocalSpec("local_scaling", "manhattan", k=3),
    global_=GlobalSpec("centroid", "cosine", "exponential", 2.0),
    lambda_=0.7,
)

_REPETITIONS = 3


def configure_runtime(repetitions: int) -> None:
    global _REPETITIONS
    if repetitions < 1:
        raise ValueError("runtime repetitions must be >= 1")
    _REPETITIONS = int(repetitions)


def _cap_cloud(x: np.ndarray, max_points: int) -> np.ndarray:
    cloud = np.asarray(x, dtype=float)
    if cloud.ndim != 2 or len(cloud) < 2 or not np.isfinite(cloud).all():
        raise ValueError("runtime benchmark expects a finite 2-D cloud with at least two points")
    if max_points > 0 and len(cloud) > max_points:
        cloud = cloud[np.linspace(0, len(cloud) - 1, max_points, dtype=int)]
    return np.ascontiguousarray(cloud)


def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    unit_a = np.divide(a, norm_a[:, None], out=np.zeros_like(a), where=norm_a[:, None] > 0)
    unit_b = np.divide(b, norm_b[:, None], out=np.zeros_like(b), where=norm_b[:, None] > 0)
    distances = 1.0 - unit_a @ unit_b.T
    zero_a = norm_a == 0
    zero_b = norm_b == 0
    distances[np.ix_(zero_a, zero_b)] = 0.0
    distances[np.ix_(zero_a, ~zero_b)] = 1.0
    distances[np.ix_(~zero_a, zero_b)] = 1.0
    return np.clip(distances, 0.0, 2.0)


def score_alg_star(a: np.ndarray, b: np.ndarray, max_points: int = 256) -> float:
    """Compute the frozen ALG* configuration from raw clouds, without reuse."""
    cloud_a = _cap_cloud(a, max_points)
    cloud_b = _cap_cloud(b, max_points)
    d_aa = cdist(cloud_a, cloud_a, metric="cityblock")
    d_bb = cdist(cloud_b, cloud_b, metric="cityblock")
    d_ab = cdist(cloud_a, cloud_b, metric="cityblock")
    k_a = min(3, len(cloud_a) - 1)
    k_b = min(3, len(cloud_b) - 1)
    np.fill_diagonal(d_aa, np.inf)
    np.fill_diagonal(d_bb, np.inf)
    radius_a = np.maximum(
        np.partition(d_aa, k_a - 1, axis=1)[:, k_a - 1], np.finfo(float).eps
    )
    radius_b = np.maximum(
        np.partition(d_bb, k_b - 1, axis=1)[:, k_b - 1], np.finfo(float).eps
    )
    nearest_b = np.argmin(d_ab, axis=1)
    nearest_a = np.argmin(d_ab, axis=0)
    affinity_a = np.exp(
        -(d_ab[np.arange(len(cloud_a)), nearest_b] ** 2)
        / np.maximum(radius_a * radius_b[nearest_b], np.finfo(float).eps)
    )
    affinity_b = np.exp(
        -(d_ab[nearest_a, np.arange(len(cloud_b))] ** 2)
        / np.maximum(radius_a[nearest_a] * radius_b, np.finfo(float).eps)
    )
    local = float(0.5 * (affinity_a.mean() + affinity_b.mean()))
    centroid_distance = float(
        cdist(
            cloud_a.mean(axis=0, keepdims=True),
            cloud_b.mean(axis=0, keepdims=True),
            metric="cosine",
        )[0, 0]
    )
    if not np.isfinite(centroid_distance):
        centroid_distance = 1.0
    global_similarity = float(np.exp(-max(centroid_distance, 0.0) / 2.0))
    return 0.7 * local + 0.3 * global_similarity


def score_initial_alg(a: np.ndarray, b: np.ndarray, max_points: int = 256) -> float:
    """Compute cosine overlap + pooled cosine from raw clouds, without reuse."""
    cloud_a = _cap_cloud(a, max_points)
    cloud_b = _cap_cloud(b, max_points)
    d_aa = _cosine_distance_matrix(cloud_a, cloud_a)
    d_bb = _cosine_distance_matrix(cloud_b, cloud_b)
    np.fill_diagonal(d_aa, np.inf)
    np.fill_diagonal(d_bb, np.inf)
    d_ab = _cosine_distance_matrix(cloud_a, cloud_b)
    local = 0.5 * (
        float(np.mean(d_ab.min(axis=1) <= d_aa.min(axis=1)))
        + float(np.mean(d_ab.min(axis=0) <= d_bb.min(axis=1)))
    )
    pooled = float(
        cosine_similarity(
            cloud_a.mean(axis=0, keepdims=True),
            cloud_b.mean(axis=0, keepdims=True),
        )[0, 0]
    )
    return local + pooled


def _time_call(function, a: np.ndarray, b: np.ndarray, max_points: int) -> tuple[float, float, float]:
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    score = float(function(a, b, max_points=max_points))
    cpu = time.process_time() - cpu_started
    wall = time.perf_counter() - wall_started
    return score, wall, cpu


def benchmark_pair(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_points: int = 256,
    repetitions: int | None = None,
    reverse_first: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Time both methods symmetrically; every repetition starts from raw clouds."""
    n_repetitions = _REPETITIONS if repetitions is None else int(repetitions)
    if n_repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    functions = [
        (ALG_STAR_METRIC, score_alg_star),
        (INITIAL_ALG_METRIC, score_initial_alg),
    ]
    wall: dict[str, list[float]] = {name: [] for name, _ in functions}
    cpu: dict[str, list[float]] = {name: [] for name, _ in functions}
    scores: dict[str, float] = {}
    for repetition in range(n_repetitions):
        order = functions[::-1] if bool(reverse_first) ^ bool(repetition % 2) else functions
        for name, function in order:
            value, wall_sec, cpu_sec = _time_call(function, a, b, max_points)
            if name in scores and not np.isclose(scores[name], value, rtol=0.0, atol=1e-12):
                raise AssertionError(f"non-deterministic score for {name}")
            scores[name] = value
            wall[name].append(wall_sec)
            cpu[name].append(cpu_sec)

    diagnostics: dict[str, float] = {
        "alg_runtime_repetitions": float(n_repetitions),
        "alg_runtime_n_a": float(min(len(a), max_points) if max_points > 0 else len(a)),
        "alg_runtime_n_b": float(min(len(b), max_points) if max_points > 0 else len(b)),
    }
    for name, _ in functions:
        diagnostics[f"time_{name}_sec"] = float(np.median(wall[name]))
        diagnostics[f"cpu_{name}_sec"] = float(np.median(cpu[name]))
        diagnostics[f"time_{name}_first_sec"] = float(wall[name][0])
        diagnostics[f"time_{name}_total_sec"] = float(np.sum(wall[name]))
    return scores, diagnostics

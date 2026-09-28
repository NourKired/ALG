#!/usr/bin/env python3
"""Isolated cold and cached implementation of the frozen ALG configuration."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import time

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.neighbors import NearestNeighbors


FROZEN_ALG_RUNTIME_METRIC = "alg_frozen_cosine_rational_scale_1_lambda_0.6_isolated"


@dataclass
class PreparedCosineCloud:
    cloud: np.ndarray
    radii: np.ndarray
    neighbors: NearestNeighbors
    centroid: np.ndarray

    @property
    def retained_bytes(self) -> int:
        return int(self.cloud.nbytes + self.radii.nbytes + self.centroid.nbytes)


def _as_cloud(x: np.ndarray, max_points: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Frozen ALG expects a finite 2-D cloud with at least two points")
    if max_points > 0 and len(x) > max_points:
        x = x[np.linspace(0, len(x) - 1, max_points, dtype=int)]
    return np.ascontiguousarray(x)


def _prepare(x: np.ndarray, max_points: int) -> PreparedCosineCloud:
    cloud = _as_cloud(x, max_points)
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute").fit(cloud)
    radii = neighbors.kneighbors(cloud, return_distance=True)[0][:, 1]
    return PreparedCosineCloud(cloud, radii, neighbors, cloud.mean(axis=0))


def _centroid_cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 1.0
    return float(cosine(a, b))


def _compare(a: PreparedCosineCloud, b: PreparedCosineCloud) -> float:
    cross_a = b.neighbors.kneighbors(a.cloud, n_neighbors=1, return_distance=True)[0][:, 0]
    cross_b = a.neighbors.kneighbors(b.cloud, n_neighbors=1, return_distance=True)[0][:, 0]
    local = 0.5 * (np.mean(cross_a <= a.radii) + np.mean(cross_b <= b.radii))
    distance = _centroid_cosine_distance(a.centroid, b.centroid)
    global_similarity = 1.0 / (1.0 + distance)
    return float(0.6 * local + 0.4 * global_similarity)


class FrozenALGCache:
    """Small deterministic LRU cache for same-cloud radii and centroids."""

    def __init__(self, max_entries: int = 4096):
        self.max_entries = int(max_entries)
        self._entries: OrderedDict[str, PreparedCosineCloud] = OrderedDict()

    @staticmethod
    def _key(x: np.ndarray, max_points: int) -> str:
        cloud = _as_cloud(x, max_points)
        digest = hashlib.blake2b(cloud.view(np.uint8), digest_size=16).hexdigest()
        return f"{cloud.shape}:{cloud.dtype}:{digest}"

    def prepare(self, x: np.ndarray, max_points: int) -> tuple[PreparedCosineCloud, bool, float]:
        key = self._key(x, max_points)
        if key in self._entries:
            value = self._entries.pop(key)
            self._entries[key] = value
            return value, True, 0.0
        started = time.perf_counter()
        value = _prepare(x, max_points)
        elapsed = time.perf_counter() - started
        self._entries[key] = value
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return value, False, elapsed


GLOBAL_FROZEN_ALG_CACHE = FrozenALGCache()


def benchmark_frozen_alg_pair(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_points: int = 256,
    cache: FrozenALGCache = GLOBAL_FROZEN_ALG_CACHE,
) -> tuple[float, dict[str, float]]:
    """Return the score plus cold/cached timing and retained-cache diagnostics."""
    cold_started = time.perf_counter()
    cold_a = _prepare(a, max_points)
    cold_b = _prepare(b, max_points)
    cold_score = _compare(cold_a, cold_b)
    cold_sec = time.perf_counter() - cold_started

    cached_a, hit_a, prep_a_sec = cache.prepare(a, max_points)
    cached_b, hit_b, prep_b_sec = cache.prepare(b, max_points)
    cached_started = time.perf_counter()
    cached_score = _compare(cached_a, cached_b)
    cached_sec = time.perf_counter() - cached_started
    if not np.isclose(cold_score, cached_score, rtol=0.0, atol=1e-12):
        raise AssertionError("Cold and cached frozen ALG scores differ")
    diagnostics = {
        "time_alg_frozen_cold_sec": cold_sec,
        "time_alg_frozen_cached_sec": cached_sec,
        "time_alg_frozen_cache_prepare_sec": prep_a_sec + prep_b_sec,
        "alg_frozen_cache_hit_a": float(hit_a),
        "alg_frozen_cache_hit_b": float(hit_b),
        "alg_frozen_pair_retained_cache_bytes": float(
            cached_a.retained_bytes + cached_b.retained_bytes
        ),
    }
    return cached_score, diagnostics

#!/usr/bin/env python3
"""Shared hyperparameter grids for fair development-selected baselines.

All variants follow the project convention that larger values mean more
similar clouds. Expensive quantities are shared within a pair so that tuning
does not repeat identical distance-matrix work.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


MMD_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
NEIGHBOR_K = (1, 3, 5, 10)
PRC_K = (1, 2, 3)
PRC_C = (1, 2, 3, 4)
SINKHORN_REG = (0.01, 0.03, 0.1, 0.3, 1.0)
SLICED_PROJECTIONS = (8, 16, 32, 64, 128)


def tuned_baseline_metric_names() -> list[str]:
    names = [f"tuned_mmd_rbf_scale_{scale:g}" for scale in MMD_SCALES]
    for k in NEIGHBOR_K:
        names.extend(
            [
                f"tuned_local_scaling_k_{k}",
                f"tuned_prdc_density_sym_k_{k}",
                f"tuned_coverage_sym_k_{k}",
            ]
        )
    names.extend(f"tuned_prc_f1_k_{k}_c_{c}" for k in PRC_K for c in PRC_C)
    names.extend(f"tuned_sinkhorn_reg_{reg:g}" for reg in SINKHORN_REG)
    names.extend(f"tuned_sliced_wasserstein_proj_{n}" for n in SLICED_PROJECTIONS)
    return names


def tuned_baseline_family(name: str) -> str:
    for family in (
        "mmd_rbf",
        "local_scaling",
        "prdc_density_sym",
        "coverage_sym",
        "prc_f1",
        "sinkhorn",
        "sliced_wasserstein",
    ):
        if name.startswith(f"tuned_{family}_"):
            return family
    raise KeyError(name)


def _subsample(x: np.ndarray, max_points: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite 2-D cloud with at least two points")
    if max_points > 0 and len(x) > max_points:
        return x[np.linspace(0, len(x) - 1, max_points, dtype=int)]
    return x


def _kth_radii(square: np.ndarray, k: int) -> np.ndarray:
    k_eff = min(max(int(k), 1), len(square) - 1)
    work = square.copy()
    np.fill_diagonal(work, np.inf)
    return np.partition(work, k_eff - 1, axis=1)[:, k_eff - 1]


def tuned_baseline_grid_scores(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_points: int = 64,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute all fair-tuning variants and family-level timing diagnostics."""
    started = time.perf_counter()
    a = _subsample(a, max_points)
    b = _subsample(b, max_points)
    values: dict[str, float] = {}
    diagnostics: dict[str, float] = {}

    distance_started = time.perf_counter()
    d_ab = cdist(a, b, metric="euclidean")
    d_aa = cdist(a, a, metric="euclidean")
    d_bb = cdist(b, b, metric="euclidean")
    diagnostics["time_tuned_shared_distances_sec"] = time.perf_counter() - distance_started

    family_started = time.perf_counter()
    z = np.vstack([a, b])
    squared = cdist(z, z, metric="sqeuclidean")
    positive = squared[squared > 0]
    base_bandwidth2 = max(
        float(np.median(positive)) if len(positive) else 1.0,
        np.finfo(float).eps,
    )
    aa2 = squared[: len(a), : len(a)]
    bb2 = squared[len(a) :, len(a) :]
    ab2 = squared[: len(a), len(a) :]
    for scale in MMD_SCALES:
        bandwidth2 = base_bandwidth2 * scale
        kaa = np.exp(-aa2 / (2.0 * bandwidth2)).mean()
        kbb = np.exp(-bb2 / (2.0 * bandwidth2)).mean()
        kab = np.exp(-ab2 / (2.0 * bandwidth2)).mean()
        values[f"tuned_mmd_rbf_scale_{scale:g}"] = -float(max(kaa + kbb - 2.0 * kab, 0.0))
    diagnostics["time_tuned_mmd_rbf_family_sec"] = time.perf_counter() - family_started

    family_started = time.perf_counter()
    nearest_b = np.argmin(d_ab, axis=1)
    nearest_a = np.argmin(d_ab, axis=0)
    cross_a = d_ab[np.arange(len(a)), nearest_b]
    cross_b = d_ab[nearest_a, np.arange(len(b))]
    for k in NEIGHBOR_K:
        k_eff = min(k, len(a) - 1, len(b) - 1)
        radii_a = np.maximum(_kth_radii(d_aa, k_eff), np.finfo(float).eps)
        radii_b = np.maximum(_kth_radii(d_bb, k_eff), np.finfo(float).eps)
        affinity_a = np.exp(-(cross_a**2) / (radii_a * radii_b[nearest_b]))
        affinity_b = np.exp(-(cross_b**2) / (radii_a[nearest_a] * radii_b))
        values[f"tuned_local_scaling_k_{k}"] = float(0.5 * (affinity_a.mean() + affinity_b.mean()))
        density_b_in_a = np.mean(np.sum(d_ab <= radii_a[:, None], axis=0) / k_eff)
        density_a_in_b = np.mean(np.sum(d_ab <= radii_b[None, :], axis=1) / k_eff)
        values[f"tuned_prdc_density_sym_k_{k}"] = float(0.5 * (density_a_in_b + density_b_in_a))
        coverage_a = np.mean(d_ab.min(axis=1) <= radii_a)
        coverage_b = np.mean(d_ab.min(axis=0) <= radii_b)
        values[f"tuned_coverage_sym_k_{k}"] = float(0.5 * (coverage_a + coverage_b))
    diagnostics["time_tuned_neighborhood_families_sec"] = time.perf_counter() - family_started

    family_started = time.perf_counter()
    for k in PRC_K:
        for c in PRC_C:
            ka = min(c * k, len(a) - 1)
            kb = min(c * k, len(b) - 1)
            radii_a = _kth_radii(d_aa, ka)
            radii_b = _kth_radii(d_bb, kb)
            precision = float(np.mean(np.sum(d_ab <= radii_a[:, None], axis=1) >= min(k, len(b))))
            recall = float(np.mean(np.sum(d_ab <= radii_b[None, :], axis=0) >= min(k, len(a))))
            values[f"tuned_prc_f1_k_{k}_c_{c}"] = (
                2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
            )
    diagnostics["time_tuned_prc_family_sec"] = time.perf_counter() - family_started

    family_started = time.perf_counter()
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(max(SLICED_PROJECTIONS), a.shape[1]))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), np.finfo(float).eps)
    per_projection = np.asarray([wasserstein_distance(a @ u, b @ u) for u in directions])
    for n_projection in SLICED_PROJECTIONS:
        values[f"tuned_sliced_wasserstein_proj_{n_projection}"] = -float(per_projection[:n_projection].mean())
    diagnostics["time_tuned_sliced_wasserstein_family_sec"] = time.perf_counter() - family_started

    family_started = time.perf_counter()
    try:
        import ot

        positive_distances = np.concatenate(
            [d_ab[d_ab > 0], d_aa[d_aa > 0], d_bb[d_bb > 0]]
        )
        scale = float(np.median(positive_distances)) if len(positive_distances) else 1.0
        wa = np.full(len(a), 1.0 / len(a))
        wb = np.full(len(b), 1.0 / len(b))

        def sinkhorn(xw: np.ndarray, yw: np.ndarray, cost: np.ndarray, reg: float) -> float:
            return float(
                ot.sinkhorn2(
                    xw,
                    yw,
                    cost,
                    max(reg * scale, np.finfo(float).eps),
                    method="sinkhorn_log",
                    numItermax=2000,
                    stopThr=1e-8,
                )
            )

        for reg in SINKHORN_REG:
            divergence = sinkhorn(wa, wb, d_ab, reg)
            divergence -= 0.5 * sinkhorn(wa, wa, d_aa, reg)
            divergence -= 0.5 * sinkhorn(wb, wb, d_bb, reg)
            values[f"tuned_sinkhorn_reg_{reg:g}"] = -float(max(divergence, 0.0))
    except Exception:
        for reg in SINKHORN_REG:
            values[f"tuned_sinkhorn_reg_{reg:g}"] = math.nan
    diagnostics["time_tuned_sinkhorn_family_sec"] = time.perf_counter() - family_started
    diagnostics["time_tuned_baseline_grid_sec"] = time.perf_counter() - started
    return values, diagnostics


assert len(tuned_baseline_metric_names()) == 39

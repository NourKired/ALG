"""One consistent metric suite for controlled and feature-based protocols."""
from __future__ import annotations

import time

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from alg_similarity.alg_framework_grid import ALGFrameworkSpec, GlobalSpec, LocalSpec, alg_framework_scores
from alg_similarity.alg_metric_grid import alg_grid_scores
from run_naeem2020_toy_protocol import directional_prdc, radii
from run_overlap_metric_large_scale import score_pair


SELECTED_ALG_SPEC = ALGFrameworkSpec(
    local=LocalSpec("local_scaling", "manhattan", k=3),
    global_=GlobalSpec("centroid", "cosine", "exponential", scale=2.0),
    lambda_=0.7,
)


def deterministic_cap(x: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or len(x) <= cap:
        return x
    return x[np.linspace(0, len(x) - 1, cap, dtype=int)]


def cmmd_distance(x: np.ndarray, y: np.ndarray, sigma: float = 10.0) -> float:
    """Official CMMD estimator on precomputed CLIP embeddings."""
    if min(len(x), len(y)) < 2:
        return float("nan")
    gamma = 1.0 / (2.0 * sigma * sigma)
    k_xx = rbf_kernel(x, x, gamma=gamma)
    k_yy = rbf_kernel(y, y, gamma=gamma)
    k_xy = rbf_kernel(x, y, gamma=gamma)
    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)
    estimate = (
        k_xx.sum() / (len(x) * (len(x) - 1))
        + k_yy.sum() / (len(y) * (len(y) - 1))
        - 2.0 * k_xy.mean()
    )
    return float(1000.0 * estimate)


def full_literature_scores(x: np.ndarray, y: np.ndarray, k: int, metric_max_points: int) -> dict[str, object]:
    """PRDC originals, every baseline, and the shared current ALG grid."""
    directional_started = time.perf_counter()
    radius_x, radius_y = radii(x, k), radii(y, k)
    precision, density, coverage = directional_prdc(x, y, radius_x, k)
    recall, _, _ = directional_prdc(y, x, radius_y, k)
    result: dict[str, object] = {
        "precision": precision,
        "recall": recall,
        "density": density,
        "coverage": coverage,
        "time_directional_prdc_full_sec": time.perf_counter() - directional_started,
    }
    result.update({f"baseline_{name}": value for name, value in score_pair(x, y).items()})
    grid_x = deterministic_cap(x, metric_max_points)
    grid_y = deterministic_cap(y, metric_max_points)
    grid, diagnostics = alg_grid_scores(grid_x, grid_y)
    result.update(grid)
    result.update(diagnostics)
    # The frozen framework configuration selected on the training datasets is
    # evaluated as an ordinary method.  It is not reselected on controlled
    # perturbations and therefore cannot leak controlled-condition outcomes.
    selected_alg, selected_diagnostics = alg_framework_scores(
        grid_x,
        grid_y,
        specs=(SELECTED_ALG_SPEC,),
    )
    result.update(selected_alg)
    result.update({f"selected_{key}": value for key, value in selected_diagnostics.items()})
    result["alg_grid_protocol"] = "symmetric_internal_scale_v1"
    result["alg_grid_n_real"] = len(grid_x)
    result["alg_grid_n_synthetic"] = len(grid_y)
    return result

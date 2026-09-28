"""Toy-data protocol from Naeem et al. (ICML 2020), with ALG added.

Reproduces their specified 64-D Gaussian translation/outlier experiment and
their 10-mode sequential/simultaneous mode-dropping design.  It reports their
directional Precision, Recall, Density and Coverage (not the project's
symmetric aggregates), alongside ALG's Euclidean local, pooled and raw scores.
The paper does not specify mixture means/covariances/seeds; those are exposed
as command-line settings and recorded in the output CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

from alg_similarity.pointset_metrics import coverage_similarity
from run_overlap_metric_large_scale import overlap_cosine_percentage, score_pair


def radii(x: np.ndarray, k: int) -> np.ndarray:
    return NearestNeighbors(n_neighbors=k + 1, algorithm="brute").fit(x).kneighbors(x)[0][:, k]


def local_agreement(x: np.ndarray, y: np.ndarray, metric: str) -> tuple[float, float]:
    """Symmetric k=1 coverage and reference radius scale for any Lp geometry."""
    nn_x = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="brute").fit(x)
    nn_y = NearestNeighbors(n_neighbors=2, metric=metric, algorithm="brute").fit(y)
    rx = nn_x.kneighbors(x)[0][:, 1]
    ry = nn_y.kneighbors(y)[0][:, 1]
    dx = nn_y.kneighbors(x, n_neighbors=1)[0][:, 0]
    dy = nn_x.kneighbors(y, n_neighbors=1)[0][:, 0]
    return float(0.5 * (np.mean(dx <= rx) + np.mean(dy <= ry))), max(float(np.median(rx)), np.finfo(float).eps)


def add_normalized_variants(result: dict[str, float], geometry: str, local: float, distance: float, scale: float) -> None:
    """Full finite grid of normalized ALG variants for one non-cosine geometry."""
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        z = distance / (multiplier * scale)
        psi_values = {
            "affine_clipped": max(0.0, 1.0 - z), "rational": 1.0 / (1.0 + z),
            "cauchy": 1.0 / (1.0 + z * z), "exponential": np.exp(-z),
            "gaussian": np.exp(-0.5 * z * z), "logistic": 2.0 / (1.0 + np.exp(z)),
        }
        for psi_name, global_agreement in psi_values.items():
            for lam in np.linspace(0.0, 1.0, 11):
                result[f"alg_{geometry}_{psi_name}_scale_{multiplier:g}_lambda_{lam:.1f}"] = lam * local + (1 - lam) * float(global_agreement)


def directional_prdc(reference: np.ndarray, query: np.ndarray, radius: np.ndarray, k: int, chunk: int = 512) -> tuple[float, float, float]:
    """Precision, Density and reference Coverage using Naeem et al.'s definitions."""
    reference_hit = np.zeros(len(reference), dtype=bool)
    query_in_manifold = 0
    density_count = 0
    for start in range(0, len(query), chunk):
        d = cdist(reference, query[start : start + chunk])
        inside = d <= radius[:, None]
        query_in_manifold += int(inside.any(axis=0).sum())
        density_count += int(inside.sum())
        reference_hit |= inside.any(axis=1)
    return query_in_manifold / len(query), density_count / (k * len(query)), float(reference_hit.mean())


def scores(x: np.ndarray, y: np.ndarray, k: int, all_baselines: bool) -> dict[str, float]:
    t_directional = time.perf_counter()
    rx, ry = radii(x, k), radii(y, k)
    precision, density, coverage = directional_prdc(x, y, rx, k)
    recall, _, _ = directional_prdc(y, x, ry, k)
    directional_elapsed = time.perf_counter() - t_directional
    t_alg = time.perf_counter()
    local = coverage_similarity(x, y, k=1)
    global_distance = float(np.linalg.norm(x.mean(axis=0) - y.mean(axis=0)))
    # Reference-only local scale: it is fixed by X, not tuned from the test
    # score.  Multipliers are explicitly reported for later dev-set selection.
    scale = max(float(np.median(radii(x, 1))), np.finfo(float).eps)
    local_cosine, _, _ = overlap_cosine_percentage(x, y)
    mx, my = x.mean(axis=0), y.mean(axis=0)
    cosine_global = float(mx @ my / max(np.linalg.norm(mx) * np.linalg.norm(my), np.finfo(float).eps))
    result = {"precision": precision, "recall": recall, "density": density,
            "coverage": coverage, "alg_local_k1": local,
            "alg_global_distance": global_distance, "alg_raw": local - global_distance}
    lambdas = np.linspace(0.0, 1.0, 11)
    add_normalized_variants(result, "euclidean", local, global_distance, scale)
    for geometry, nn_metric, cdist_metric in (("manhattan", "manhattan", "cityblock"), ("chebyshev", "chebyshev", "chebyshev")):
        local_lp, scale_lp = local_agreement(x, y, nn_metric)
        distance_lp = float(cdist(x.mean(axis=0, keepdims=True), y.mean(axis=0, keepdims=True), metric=cdist_metric)[0, 0])
        result[f"alg_{geometry}_local_k1"] = local_lp
        result[f"alg_{geometry}_global_distance"] = distance_lp
        result[f"alg_{geometry}_raw"] = local_lp - distance_lp
        add_normalized_variants(result, geometry, local_lp, distance_lp, scale_lp)
    for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0):
        z = (1.0 - cosine_global) / multiplier
        for psi_name, global_agreement in {
            "linear": max(0.0, 1.0 - z / 2.0),
            "rational": 1.0 / (1.0 + z),
            "cauchy": 1.0 / (1.0 + z * z),
            "exponential": np.exp(-z),
            "gaussian": np.exp(-0.5 * z * z),
            "logistic": 2.0 / (1.0 + np.exp(z)),
        }.items():
            for lam in lambdas:
                result[f"alg_cosine_{psi_name}_scale_{multiplier:g}_lambda_{lam:.1f}"] = lam * local_cosine + (1 - lam) * float(global_agreement)
    result["alg_reference_scale_k1"] = scale
    result["time_alg_grid_sec"] = time.perf_counter() - t_alg
    result["time_naeem_directional_prdc_sec"] = directional_elapsed
    if all_baselines:
        # Preserve the individual timers emitted by the common 27-baseline
        # implementation; they are required for the efficiency RQ.
        result.update({f"baseline_{name}": value for name, value in score_pair(x, y).items()})
    return result


def mixture(rng: np.random.Generator, n: int, d: int, modes: int, separation: float, weights: np.ndarray) -> np.ndarray:
    # ``np.eye(modes, d)`` already has exactly the required (modes, d)
    # shape; slicing the destination caused a 64-D broadcast failure.
    centers = separation * np.eye(modes, d)
    labels = rng.choice(modes, size=n, p=weights)
    return centers[labels] + rng.normal(size=(n, d))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/naeem2020_toy_protocol"))
    p.add_argument("--samples", type=int, default=10_000)
    p.add_argument("--dimension", type=int, default=64)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--repetitions", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode-separation", type=float, default=8.0)
    p.add_argument("--include-all-baselines", action="store_true")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []
    mus = np.linspace(-1.0, 1.0, 21)
    for rep in range(args.repetitions):
        rng = np.random.default_rng(args.seed + rep)
        x = rng.normal(size=(args.samples, args.dimension))
        for case in ("none", "real_outlier", "fake_outlier"):
            for mu in mus:
                y = rng.normal(loc=mu, size=(args.samples, args.dimension))
                xx, yy = x, y
                outlier = 3.0 * np.ones((1, args.dimension))
                if case == "real_outlier": xx = np.vstack([x, outlier])
                if case == "fake_outlier": yy = np.vstack([y, outlier])
                rows.append({"experiment": "translation_outlier", "case": case, "level": mu, "repetition": rep, **scores(xx, yy, args.k, args.include_all_baselines)})
        weights = np.full(10, 0.1)
        x = mixture(rng, args.samples, args.dimension, 10, args.mode_separation, weights)
        for remaining in range(10, 0, -1):
            w = np.zeros(10); w[:remaining] = 1 / remaining
            y = mixture(rng, args.samples, args.dimension, 10, args.mode_separation, w)
            rows.append({"experiment": "mode_sequential", "case": "none", "level": remaining, "repetition": rep, **scores(x, y, args.k, args.include_all_baselines)})
        for t in np.linspace(0.0, 1.0, 10):
            first_mass = 0.1 + 0.9 * t
            w = np.full(10, (1 - first_mass) / 9); w[0] = first_mass
            y = mixture(rng, args.samples, args.dimension, 10, args.mode_separation, w)
            rows.append({"experiment": "mode_simultaneous", "case": "none", "level": first_mass, "repetition": rep, **scores(x, y, args.k, args.include_all_baselines)})
    pd.DataFrame(rows).to_csv(args.output_dir / "naeem2020_toy_scores.csv", index=False)
    print(f"Saved protocol results to {args.output_dir / 'naeem2020_toy_scores.csv'}")


if __name__ == "__main__":
    main()

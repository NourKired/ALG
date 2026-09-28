"""Controlled sanity checks for Adaptive Local--Global (ALG) similarity.

The protocol follows the logic of Naeem et al. (2020): generate point-cloud
pairs where the desired qualitative change is known before evaluating a score.
It intentionally measures both desirable responses and known failure modes.

Example
-------
python scripts/run_alg_controlled_sanity_checks.py \
  --output-dir outputs/alg_controlled_sanity --repetitions 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pointset_metrics import coverage_similarity, mmd_rbf_similarity, prdc_density_similarity
from run_overlap_metric_large_scale import overlap_percentage_3d


def make_mixture(rng: np.random.Generator, n_modes: int, points_per_mode: int, dim: int) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return an isotropic Gaussian mixture and its index sets by mode."""
    centers = rng.normal(size=(n_modes, dim))
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12)
    centers *= 5.0
    clouds = [center + 0.45 * rng.normal(size=(points_per_mode, dim)) for center in centers]
    return np.vstack(clouds), clouds


def score(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    local, _, a_to_b, b_to_a = overlap_percentage_3d(a, b)
    global_similarity = -float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    return {
        "local_euclidean": local,
        "local_a_to_b": a_to_b,
        "local_b_to_a": b_to_a,
        "pooled_euclidean": global_similarity,
        "alg_euclidean_raw": local + global_similarity,
        "prdc_coverage_k5": coverage_similarity(a, b, k=5),
        "prdc_density_k5": prdc_density_similarity(a, b),
        "rbf_mmd": mmd_rbf_similarity(a, b),
    }


def experiment_pairs(
    a: np.ndarray, modes: list[np.ndarray], level: int, condition: str, dim: int
) -> np.ndarray:
    """Create B for one controlled condition and severity level."""
    if condition == "translation":
        direction = np.ones(dim) / np.sqrt(dim)
        return a + (0.2 * level) * direction
    if condition == "local_deformation":
        b = a.copy()
        n = len(modes[0])
        direction = np.ones(dim) / np.sqrt(dim)
        b[:n] += (0.45 * level) * direction
        # Recenter: isolate an internal-geometry change from a mean shift.
        return b - b.mean(axis=0) + a.mean(axis=0)
    if condition == "mode_removal":
        keep = max(1, len(modes) - level)
        return np.vstack(modes[:keep])
    if condition == "outliers":
        if level == 0:
            return a.copy()
        direction = np.ones(dim) / np.sqrt(dim)
        outliers = a.mean(axis=0) + 14.0 * direction + np.arange(level)[:, None] * 0.1 * direction
        return np.vstack([a, outliers])
    raise ValueError(f"Unknown condition: {condition}")


def run(args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    conditions = {
        "translation": range(0, 9),
        "local_deformation": range(0, 9),
        "mode_removal": range(0, args.modes),
        "outliers": range(0, 9),
    }
    for repetition in range(args.repetitions):
        rng = np.random.default_rng(args.seed + repetition)
        a, modes = make_mixture(rng, args.modes, args.points_per_mode, args.dimension)
        for condition, levels in conditions.items():
            for level in levels:
                b = experiment_pairs(a, modes, level, condition, args.dimension)
                values = score(a, b)
                rows.append({"condition": condition, "level": level, "repetition": repetition, **values})
    return pd.DataFrame(rows)


def plot(results: pd.DataFrame, destination: Path) -> None:
    metrics = ["local_euclidean", "pooled_euclidean", "alg_euclidean_raw", "prdc_coverage_k5", "prdc_density_k5", "rbf_mmd"]
    labels = ["Local coverage (k=1)", "Pooled Euclidean", "ALG raw", "PRDC coverage (k=5)", "PRDC density (k=5)", "RBF-MMD"]
    conditions = ["translation", "local_deformation", "mode_removal", "outliers"]
    titles = ["Global translation", "Local deformation; fixed centroid", "Progressive mode removal", "Added outliers"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    colors = plt.cm.tab10(np.arange(len(metrics)))
    for axis, condition, title in zip(axes.flat, conditions, titles):
        subset = results.loc[results["condition"] == condition]
        for metric, label, color in zip(metrics, labels, colors):
            grouped = subset.groupby("level")[metric]
            mean = grouped.mean()
            se = grouped.sem().fillna(0.0)
            axis.plot(mean.index, mean.values, marker="o", linewidth=1.6, markersize=3, label=label, color=color)
            axis.fill_between(mean.index, mean - se, mean + se, alpha=0.12, color=color)
        axis.set_title(title)
        axis.set_xlabel("Perturbation level")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel("Score (native scale)")
    axes[1, 0].set_ylabel("Score (native scale)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.savefig(destination, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_alg_components(results: pd.DataFrame, destination: Path) -> None:
    """Publication-oriented view: each ALG component keeps its own scale."""
    conditions = ["translation", "local_deformation", "mode_removal", "outliers"]
    titles = ["Global translation", "Local deformation\n(fixed centroid)", "Progressive mode removal", "Added outliers"]
    components = [
        ("local_euclidean", "Local coverage $L$", "#1f77b4"),
        ("pooled_euclidean", "Pooled similarity $-D(\\mu_A,\\mu_B)$", "#ff7f0e"),
        ("alg_euclidean_raw", "Raw ALG $L-D$", "#2ca02c"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(12, 6.4), sharex="col", constrained_layout=True)
    for col, (condition, title) in enumerate(zip(conditions, titles)):
        subset = results.loc[results["condition"] == condition]
        for row, (metric, ylabel, color) in enumerate(components):
            axis = axes[row, col]
            grouped = subset.groupby("level")[metric]
            mean = grouped.mean()
            se = grouped.sem().fillna(0.0)
            axis.plot(mean.index, mean.values, marker="o", color=color, linewidth=1.8, markersize=3)
            axis.fill_between(mean.index, mean - se, mean + se, color=color, alpha=0.15)
            axis.grid(alpha=0.2)
            if row == 0:
                axis.set_title(title)
            if col == 0:
                axis.set_ylabel(ylabel)
            if row == 2:
                axis.set_xlabel("Perturbation level")
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alg_controlled_sanity"))
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--points-per-mode", type=int, default=30)
    parser.add_argument("--dimension", type=int, default=32)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = run(args)
    results.to_csv(args.output_dir / "controlled_sanity_scores.csv", index=False)
    summary = results.groupby(["condition", "level"], as_index=False).mean(numeric_only=True)
    summary.to_csv(args.output_dir / "controlled_sanity_summary.csv", index=False)
    plot(results, args.output_dir / "controlled_sanity_curves.png")
    plot_alg_components(results, args.output_dir / "alg_component_sanity_curves.png")
    print(f"Saved {len(results)} rows to {args.output_dir}")


if __name__ == "__main__":
    main()

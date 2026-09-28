#!/usr/bin/env python3
"""Aggregate external neural-representation validation without reselection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alg_framework_grid import framework_grid_size, iter_framework_specs


DEFAULT_FROZEN = (
    "algfw_L-local_scaling_manhattan_k3_"
    "G-centroid_cosine_exponential_scale2_lambda0.7"
)


def fractional_wins(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fractional = np.zeros(matrix.shape[0], dtype=float)
    strict = np.zeros(matrix.shape[0], dtype=int)
    tied = np.zeros(matrix.shape[0], dtype=int)
    for column in range(matrix.shape[1]):
        finite = np.isfinite(matrix[:, column])
        if not finite.any():
            continue
        best = np.nanmax(matrix[:, column])
        winners = np.flatnonzero(finite & np.isclose(matrix[:, column], best, atol=1e-12, rtol=0))
        fractional[winners] += 1 / len(winners)
        if len(winners) == 1:
            strict[winners[0]] += 1
        else:
            tied[winners] += 1
    return fractional, strict, tied


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-metric", default=DEFAULT_FROZEN)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs = pd.DataFrame([spec.as_record() for spec in iter_framework_specs()])
    frozen_matches = np.flatnonzero(configs["metric"].to_numpy() == args.frozen_metric)
    if len(configs) != framework_grid_size() or len(frozen_matches) != 1:
        raise RuntimeError("Framework grid or frozen metric mismatch")
    frozen_index = int(frozen_matches[0])

    archives = []
    for path in sorted((args.results_root / "datasets").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            archives.append(
                {
                    "dataset": str(archive["dataset"].item()),
                    "f1": archive["f1"].astype(float),
                    "n_pairs": int(archive["n_pairs"].item()),
                    "component_compute_sec": float(archive["component_compute_sec"].item()),
                    "evaluation_sec": float(archive["exhaustive_evaluation_sec"].item()),
                }
            )
    if len(archives) != 3:
        raise RuntimeError(f"Expected 3 external neural datasets, found {len(archives)}")
    datasets = [row["dataset"] for row in archives]
    matrix = np.column_stack([row["f1"] for row in archives])
    if matrix.shape != (framework_grid_size(), 3):
        raise RuntimeError(f"Unexpected framework matrix shape: {matrix.shape}")

    wins, strict, tied = fractional_wins(matrix)
    ranking = configs.copy()
    ranking["n_datasets"] = 3
    ranking["mean_f1"] = matrix.mean(axis=1)
    ranking["median_f1"] = np.median(matrix, axis=1)
    ranking["n_dataset_wins_within_framework"] = wins
    ranking["n_strict_wins_within_framework"] = strict
    ranking["n_tied_wins_within_framework"] = tied
    ranking.sort_values(["mean_f1", "n_dataset_wins_within_framework"], ascending=False).to_csv(
        args.output_dir / f"framework_all_{framework_grid_size()}_neural_ranking.csv", index=False
    )

    frozen = pd.DataFrame(
        {
            "support": "neural_models_external",
            "dataset": datasets,
            "metric": args.frozen_metric,
            "f1": matrix[frozen_index],
            "configuration_status": "frozen_before_external_neural_evaluation",
        }
    )
    frozen.to_csv(args.output_dir / "frozen_alg_neural_per_dataset.csv", index=False)

    baseline_parts = []
    for path in sorted((args.results_root / "baselines").glob("*.csv")):
        frame = pd.read_csv(path)
        if not frame.empty:
            baseline_parts.append(frame[["support", "dataset", "metric", "f1"]])
    if not baseline_parts:
        raise RuntimeError("No baseline results found")
    combined = pd.concat([frozen, *baseline_parts], ignore_index=True, sort=False)
    methods = sorted(combined["metric"].unique())
    comparison = combined.pivot_table(index="metric", columns="dataset", values="f1", aggfunc="first")
    values = comparison.reindex(index=methods, columns=datasets).to_numpy(float)
    wins, strict, tied = fractional_wins(values)
    leaderboard = pd.DataFrame(
        {
            "metric": methods,
            "n_datasets": np.isfinite(values).sum(axis=1),
            "mean_f1": np.nanmean(values, axis=1),
            "median_f1": np.nanmedian(values, axis=1),
            "n_dataset_wins": wins,
            "n_strict_wins": strict,
            "n_tied_wins": tied,
        }
    ).sort_values(["mean_f1", "n_dataset_wins"], ascending=False)
    leaderboard.to_csv(args.output_dir / "frozen_alg_plus_baselines_neural_leaderboard.csv", index=False)
    combined.to_csv(args.output_dir / "all_methods_neural_per_dataset.csv", index=False)

    metadata_rows = []
    for path in sorted(args.components_root.glob("shard_*/neural_protocol_metadata.json")):
        metadata = json.loads(path.read_text())
        metadata.pop("models", None)
        metadata_rows.append(metadata)
    pd.DataFrame(metadata_rows).to_csv(args.output_dir / "neural_dataset_metadata.csv", index=False)
    pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "n_pairs": row["n_pairs"],
                "component_compute_sec": row["component_compute_sec"],
                "exhaustive_evaluation_sec": row["evaluation_sec"],
            }
            for row in archives
        ]
    ).to_csv(args.output_dir / "neural_framework_runtime.csv", index=False)
    print(
        f"Aggregated frozen ALG and baselines on {len(datasets)} external neural datasets; "
        f"preserved all {framework_grid_size():,} exploratory configurations."
    )


if __name__ == "__main__":
    main()

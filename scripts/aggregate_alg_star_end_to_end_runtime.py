#!/usr/bin/env python3
"""Aggregate paired, end-to-end ALG* versus Initial ALG runtime measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alg_similarity.alg_star_end_to_end_runtime import ALG_STAR_METRIC, INITIAL_ALG_METRIC, ALG_STAR_NAME


METHODS = (ALG_STAR_METRIC, INITIAL_ALG_METRIC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-datasets", type=int, default=183)
    args = parser.parse_args()

    paths = sorted(args.input_root.glob("shard_*/pair_scores.csv"))
    if not paths:
        raise SystemExit(f"No shard pair_scores.csv files found under {args.input_root}")
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    raw = pd.concat(frames, ignore_index=True)
    n_datasets = int(raw["dataset"].nunique())
    if n_datasets != args.expected_datasets:
        raise SystemExit(f"Expected {args.expected_datasets} datasets, found {n_datasets}")

    required = ["dataset", "support", *METHODS]
    for method in METHODS:
        required.extend([f"time_{method}_sec", f"cpu_{method}_sec"])
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    if not np.isfinite(raw[required[2:]].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite ALG runtime scores or timings detected")

    long_parts = []
    for method in METHODS:
        part = raw[["dataset", "support"]].copy()
        part["method"] = method
        part["score"] = raw[method].astype(float)
        part["wall_sec"] = raw[f"time_{method}_sec"].astype(float)
        part["cpu_sec"] = raw[f"cpu_{method}_sec"].astype(float)
        long_parts.append(part)
    long = pd.concat(long_parts, ignore_index=True)

    per_dataset = (
        long.groupby(["dataset", "support", "method"], as_index=False)
        .agg(
            n_pairs=("wall_sec", "size"),
            mean_wall_sec=("wall_sec", "mean"),
            median_wall_sec=("wall_sec", "median"),
            p95_wall_sec=("wall_sec", lambda values: float(np.quantile(values, 0.95))),
            mean_cpu_sec=("cpu_sec", "mean"),
        )
    )
    summaries = []
    for method, group in long.groupby("method", sort=False):
        dataset_group = per_dataset[per_dataset["method"] == method]
        summaries.append(
            {
                "method": method,
                "n_datasets": dataset_group["dataset"].nunique(),
                "n_pairs": len(group),
                "pair_micro_mean_ms": 1000.0 * group["wall_sec"].mean(),
                "pair_median_ms": 1000.0 * group["wall_sec"].median(),
                "pair_p95_ms": 1000.0 * group["wall_sec"].quantile(0.95),
                "dataset_macro_mean_ms": 1000.0 * dataset_group["mean_wall_sec"].mean(),
                "dataset_macro_median_ms": 1000.0 * dataset_group["median_wall_sec"].mean(),
                "pair_micro_mean_cpu_ms": 1000.0 * group["cpu_sec"].mean(),
            }
        )
    leaderboard = pd.DataFrame(summaries).sort_values("dataset_macro_mean_ms")

    ratio = raw[["dataset", "support"]].copy()
    ratio["initial_over_alg_star_wall_ratio"] = (
        raw[f"time_{INITIAL_ALG_METRIC}_sec"] / raw[f"time_{ALG_STAR_METRIC}_sec"]
    )
    ratio_summary = (
        ratio.groupby(["dataset", "support"], as_index=False)
        .agg(
            n_pairs=("initial_over_alg_star_wall_ratio", "size"),
            median_initial_over_alg_star=("initial_over_alg_star_wall_ratio", "median"),
            mean_initial_over_alg_star=("initial_over_alg_star_wall_ratio", "mean"),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "all_pair_runtime.csv.gz", index=False, compression="gzip")
    per_dataset.to_csv(args.output_dir / "runtime_per_dataset.csv", index=False)
    leaderboard.to_csv(args.output_dir / "runtime_leaderboard.csv", index=False)
    ratio_summary.to_csv(args.output_dir / "paired_runtime_ratio_per_dataset.csv", index=False)
    payload = {
        "alg_star_configuration": ALG_STAR_NAME,
        "methods": list(METHODS),
        "n_shards": len(paths),
        "n_datasets": n_datasets,
        "n_pair_rows": len(raw),
        "timing_definition": "median of independent end-to-end calls from raw clouds; no algorithmic cache",
    }
    (args.output_dir / "runtime_summary.json").write_text(json.dumps(payload, indent=2))
    print(leaderboard.to_string(index=False))


if __name__ == "__main__":
    main()


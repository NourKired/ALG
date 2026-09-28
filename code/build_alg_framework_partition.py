#!/usr/bin/env python3
"""Freeze the exact ALG-eligible development/test dataset partition."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-development", type=int, default=60)
    parser.add_argument("--expected-test", type=int, default=183)
    args = parser.parse_args()

    frozen = args.aggregate_root / "frozen_selection"
    split = pd.read_csv(frozen / "dataset_partition_frozen.csv")
    with (frozen / "selected_alg_configuration.json").open() as handle:
        selection = json.load(handle)
    metric = str(selection["selected_alg_metric"])

    usecols = ["support", "dataset", "metric", "f1"]
    summary = pd.read_csv(args.aggregate_root / "metric_summary_all_datasets.csv", usecols=usecols)
    eligible = summary[
        summary["metric"].astype(str).eq(metric)
        & np.isfinite(pd.to_numeric(summary["f1"], errors="coerce"))
    ][["support", "dataset"]].drop_duplicates("dataset")
    out = split.merge(eligible, on="dataset", how="inner", validate="one_to_one")
    coverage_path = args.aggregate_root / "expected_dataset_coverage.csv"
    if not coverage_path.is_file():
        raise FileNotFoundError(
            f"Missing {coverage_path}; deterministic job ordering cannot be reconstructed"
        )
    coverage = pd.read_csv(coverage_path, usecols=["dataset"]).drop_duplicates("dataset")
    coverage["job_order"] = np.arange(len(coverage), dtype=int)
    out = out.merge(coverage, on="dataset", how="left", validate="one_to_one")
    if out["job_order"].isna().any():
        missing_order = out.loc[out["job_order"].isna(), "dataset"].astype(str).tolist()
        raise RuntimeError(f"Datasets missing deterministic job order: {missing_order[:10]}")
    out["job_order"] = out["job_order"].astype(int)
    out = out[["support", "dataset", "partition", "job_order"]].sort_values("job_order")

    counts = out["partition"].value_counts().to_dict()
    observed_dev = int(counts.get("development", 0))
    observed_test = int(counts.get("test", 0))
    if (observed_dev, observed_test) != (args.expected_development, args.expected_test):
        raise RuntimeError(
            "ALG-eligible partition mismatch: "
            f"development={observed_dev}, test={observed_test}; expected "
            f"{args.expected_development}/{args.expected_test}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    metadata = {
        "eligibility_metric": metric,
        "source_aggregate_root": str(args.aggregate_root),
        "n_development": observed_dev,
        "n_test": observed_test,
        "n_total": len(out),
        "split_seed": selection.get("split_seed"),
    }
    with args.output.with_suffix(".json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"Saved exact ALG-eligible partition: {observed_dev} development + {observed_test} test")


if __name__ == "__main__":
    main()

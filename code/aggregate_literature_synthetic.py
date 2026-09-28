#!/usr/bin/env python3
"""Concatenate repeated controlled-literature protocol outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.input_dir.glob("literature_synthetic_seed_*.csv"))
    if not paths:
        raise SystemExit(f"No literature synthetic CSV files found under {args.input_dir}")
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    result = pd.concat(frames, ignore_index=True, sort=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    group_cols = [
        col for col in ("protocol", "scenario", "case", "implementation_status", "dimension", "level")
        if col in result.columns
    ]
    metric_cols = [
        col for col in result.select_dtypes(include="number").columns
        if col not in {"seed", "dimension", "level", "n_real", "n_synthetic"}
    ]
    summary = result.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "sem"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(args.output.with_name(args.output.stem + "_mean_se.csv"), index=False)
    print(f"Aggregated {len(paths)} seeds and {len(result)} rows into {args.output}")


if __name__ == "__main__":
    main()

"""Aggregate outputs from submit_alg_literature_array.sh."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def read_many(paths: list[Path]) -> pd.DataFrame:
    # Empty failure files are the normal success signal from array tasks.
    frames = []
    for path in paths:
        if not path.stat().st_size:
            continue
        try:
            frames.append(pd.read_csv(path, low_memory=False))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    score_paths = sorted(args.input_dir.glob("task_*_scores.csv"))
    failure_paths = sorted(args.input_dir.glob("task_*_failures.csv"))
    if not score_paths:
        raise SystemExit(f"No task score CSV files found under {args.input_dir}")
    scores = read_many(score_paths)
    failures = read_many(failure_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "literature_all_scores.csv", index=False)
    failures.to_csv(args.output_dir / "literature_all_failures.csv", index=False)
    print(f"Aggregated {len(scores)} score rows from {len(score_paths)} tasks")
    print(f"Failures: {len(failures)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate public controlled-feature inputs before Slurm submission."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"dataset", "model", "real_features", "synthetic_features"}
# The public campaign intentionally excludes licensed ImageNet and protocols
# whose author-generated samples are not distributed with reproducible inputs.
REQUIRED_DATASETS = {"CIFAR10", "MNIST"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    table = pd.read_csv(args.manifest)
    if missing := REQUIRED_COLUMNS - set(table.columns):
        raise SystemExit(f"Manifest columns missing: {sorted(missing)}")
    present = set(table["dataset"].astype(str))
    if missing_datasets := REQUIRED_DATASETS - present:
        raise SystemExit(f"Literature datasets missing from manifest: {sorted(missing_datasets)}")
    base = args.manifest.resolve().parent
    missing_paths = []
    for row_index, row in table.iterrows():
        for column in ("real_features", "synthetic_features"):
            path = Path(str(row[column])).expanduser()
            if not path.is_absolute():
                path = base / path
            if not path.is_file():
                missing_paths.append((int(row_index), column, str(path)))
    if missing_paths:
        preview = "\n".join(f"  row {i} {column}: {path}" for i, column, path in missing_paths[:20])
        raise SystemExit(f"{len(missing_paths)} feature files are missing:\n{preview}")
    print(
        f"Validated {len(table)} public controlled conditions across {table['dataset'].nunique()} datasets."
    )


if __name__ == "__main__":
    main()

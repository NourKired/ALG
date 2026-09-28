#!/usr/bin/env python3
"""Export the complete composable ALG local/global search space."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from alg_framework_grid import (
    LAMBDAS,
    framework_grid_size,
    global_grid_specs,
    iter_framework_specs,
    local_grid_specs,
)


def write_records(path: Path, records: list[dict[str, object]]) -> None:
    if not records:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alg_framework_grid"))
    parser.add_argument(
        "--write-full-fusion-grid",
        action="store_true",
        help="Write the 158,400-row Cartesian product manifest.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_records = [
        {
            "local_id": spec.slug,
            "family": spec.family,
            "geometry": spec.geometry,
            "k": spec.k,
            "alpha": spec.alpha,
        }
        for spec in local_grid_specs()
    ]
    global_records = [
        {
            "global_id": spec.slug,
            "family": spec.family,
            "geometry": spec.geometry,
            "psi": spec.psi,
            "scale": spec.scale,
        }
        for spec in global_grid_specs()
    ]
    write_records(args.output_dir / "local_options.csv", local_records)
    write_records(args.output_dir / "global_options.csv", global_records)

    if args.write_full_fusion_grid:
        fusion_path = args.output_dir / "fusion_grid_158400.csv"
        with fusion_path.open("w", newline="") as handle:
            fieldnames = list(next(iter_framework_specs()).as_record())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for spec in iter_framework_specs():
                writer.writerow(spec.as_record())

    summary = {
        "n_local_options": len(local_records),
        "n_global_options": len(global_records),
        "n_lambda_values": len(LAMBDAS),
        "lambda_values": list(LAMBDAS),
        "n_fusion_configurations": framework_grid_size(),
        "full_fusion_manifest_written": bool(args.write_full_fusion_grid),
        "selection_protocol": (
            "Select the complete local/global/lambda configuration only on development datasets, "
            "freeze it, then evaluate once on the held-out test datasets."
        ),
    }
    (args.output_dir / "grid_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Collect task-, batch-, and extern-step accounting for one Slurm job/array."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


FIELDS = [
    "JobIDRaw",
    "JobName",
    "State",
    "ElapsedRaw",
    "TotalCPU",
    "CPUTimeRAW",
    "UserCPU",
    "SystemCPU",
    "MaxRSS",
    "MaxVMSize",
    "AveRSS",
    "AllocCPUS",
    "ReqMem",
    "ExitCode",
    "NodeList",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True, help="Slurm job or array ID")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    command = [
        "sacct",
        "--parsable2",
        "--noheader",
        "--units=K",
        "--jobs",
        str(args.job_id),
        f"--format={','.join(FIELDS)}",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        for line in completed.stdout.splitlines():
            if not line.strip():
                continue
            values = line.split("|")
            if values and values[-1] == "":
                values.pop()
            if len(values) < len(FIELDS):
                values.extend([""] * (len(FIELDS) - len(values)))
            writer.writerow(values[: len(FIELDS)])

    print(f"Saved Slurm accounting for {args.job_id} to {args.output}")


if __name__ == "__main__":
    main()

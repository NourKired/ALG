#!/usr/bin/env python3
"""Controlled protocols from PRC, Clipped D/C, and Geometry Score papers.

Every row is evaluated with the same external metric collection and all 1,320
ALG configurations. Rows explicitly record whether the implemented scenario is
an exact parameterization or a controlled adaptation of the cited experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pointset_metrics import configure_metrics
from literature_metric_suite import full_literature_scores


def disk(rng: np.random.Generator, n: int) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


def ring(rng: np.random.Generator, n: int) -> np.ndarray:
    angle = rng.uniform(0, 2 * np.pi, n)
    radius = rng.normal(1.0, 0.06, n)
    return np.column_stack([radius * np.cos(angle), radius * np.sin(angle)])


def checkerboard(rng: np.random.Generator, n: int) -> np.ndarray:
    cells = np.asarray([(i, j) for i in range(4) for j in range(4) if (i + j) % 2 == 0])
    chosen = cells[rng.integers(0, len(cells), n)]
    return (chosen + rng.uniform(0, 1, size=(n, 2))) / 4.0


def record(
    rows: list[dict[str, object]], metadata: dict[str, object], x: np.ndarray, y: np.ndarray,
    k: int, metric_max_points: int,
) -> None:
    rows.append({
        **metadata,
        "n_real": len(x),
        "n_synthetic": len(y),
        **full_literature_scores(x, y, k, metric_max_points),
    })


def run_prc(rows: list[dict[str, object]], rng: np.random.Generator, samples: int, seed: int, k: int, metric_max_points: int) -> None:
    for dimension in (1, 2, 3, 8, 32):
        for shift in np.linspace(0.0, 1.5, 7):
            x = rng.uniform(0.0, 1.0, size=(samples, dimension))
            y = rng.uniform(shift, 1.0 + shift, size=(samples, dimension))
            record(rows, {
                "protocol": "cheema_urner_2023", "scenario": "uniform_translation",
                "implementation_status": "paper_family_parameterized", "seed": seed,
                "dimension": dimension, "level": shift,
            }, x, y, k, metric_max_points)
    for dimension in (2, 8):
        x = rng.uniform(-1.0, 1.0, size=(samples, dimension))
        for scale in (0.25, 0.5, 0.75, 1.0, 1.25):
            y = rng.uniform(-scale, scale, size=(samples, dimension))
            record(rows, {
                "protocol": "cheema_urner_2023", "scenario": "nested_support",
                "implementation_status": "paper_family_parameterized", "seed": seed,
                "dimension": dimension, "level": scale,
            }, x, y, k, metric_max_points)
    for shape_name, generator in (("disk", disk), ("ring", ring), ("checkerboard", checkerboard)):
        for shift in (0.0, 0.25, 0.5, 1.0):
            x = generator(rng, samples)
            y = generator(rng, samples) + np.array([shift, 0.0])
            record(rows, {
                "protocol": "cheema_urner_2023", "scenario": f"nonconvex_{shape_name}",
                "implementation_status": "controlled_adaptation", "seed": seed,
                "dimension": 2, "level": shift,
            }, x, y, k, metric_max_points)


def run_clipped(rows: list[dict[str, object]], rng: np.random.Generator, samples: int, seed: int, k: int, metric_max_points: int) -> None:
    dimension = 32
    for mu in np.linspace(-1.0, 1.0, 21):
        base_x = rng.normal(size=(samples, dimension))
        base_y = rng.normal(loc=mu, size=(samples, dimension))
        for case in ("none", "real_outlier", "synthetic_bad", "both"):
            x, y = base_x, base_y
            if case in {"real_outlier", "both"}:
                x = np.vstack([x, np.full((1, dimension), 3.0)])
            if case in {"synthetic_bad", "both"}:
                y = np.vstack([y, np.full((1, dimension), -3.0)])
            record(rows, {
                "protocol": "salvy_talbot_thirion_2026", "scenario": "gaussian_translation_outliers",
                "case": case, "implementation_status": "exact_except_sample_count" if samples != 25_000 else "exact_parameterization",
                "seed": seed, "dimension": dimension, "level": mu,
            }, x, y, k, metric_max_points)


def run_geometry(rows: list[dict[str, object]], rng: np.random.Generator, samples: int, seed: int, k: int, metric_max_points: int) -> None:
    x = ring(rng, samples)
    collapsed_angle = rng.uniform(0, np.pi / 3, samples)
    scenarios = {
        "iid_ring": ring(rng, samples),
        "collapsed_arc": np.column_stack([
            np.cos(collapsed_angle),
            np.sin(collapsed_angle),
        ]),
        "filled_disk": disk(rng, samples),
        "translated_ring": ring(rng, samples) + np.array([0.75, 0.0]),
    }
    for scenario, y in scenarios.items():
        record(rows, {
            "protocol": "khrulkov_oseledets_2018", "scenario": scenario,
            "implementation_status": "controlled_adaptation", "seed": seed,
            "dimension": 2, "level": 0.0,
        }, x, y, k, metric_max_points)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--protocol", choices=["all", "prc", "clipped", "geometry"], default="all")
    parser.add_argument("--metric-max-points", type=int, default=256)
    parser.add_argument("--geometry-repeats", type=int, default=16)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    configure_metrics(
        manifold_k=args.k,
        prc_k=3,
        prc_c=3,
        max_points=args.metric_max_points,
        geometry_repeats=args.geometry_repeats,
    )
    rows: list[dict[str, object]] = []
    if args.protocol in {"all", "prc"}:
        run_prc(rows, rng, args.samples, args.seed, args.k, args.metric_max_points)
    if args.protocol in {"all", "clipped"}:
        run_clipped(rows, rng, args.samples, args.seed, args.k, args.metric_max_points)
    if args.protocol in {"all", "geometry"}:
        run_geometry(rows, rng, args.samples, args.seed, args.k, args.metric_max_points)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"literature_synthetic_seed_{args.seed:04d}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} controlled protocol rows to {path}")


if __name__ == "__main__":
    main()

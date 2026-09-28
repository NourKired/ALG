#!/usr/bin/env python3
"""Evaluate the complete ALG grid with stratified five-fold thresholds.

The input is the component campaign's shard ``pair_scores.csv`` files. Results
are checkpointed once per dataset as compressed NumPy archives, so an array job
can be resumed without recomputing completed datasets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from alg_framework_grid import (
    LAMBDAS,
    framework_grid_size,
    global_grid_specs,
    iter_framework_specs,
    local_grid_specs,
)
from run_overlap_metric_large_scale import RANKING_SCORE_COLS


META_COLUMNS = {"support", "dataset", "object_a", "object_b", "class_a", "class_b", "label"}


def dataset_key(dataset: str) -> str:
    digest = hashlib.sha1(dataset.encode("utf-8")).hexdigest()[:12]
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in dataset)[:80]
    return f"{safe}__{digest}"


def framework_index_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locals_ = local_grid_specs()
    globals_ = global_grid_specs()
    local_lookup = {spec: index for index, spec in enumerate(locals_)}
    global_lookup = {spec: index for index, spec in enumerate(globals_)}
    specs = tuple(iter_framework_specs())
    return (
        np.asarray([local_lookup[spec.local] for spec in specs], dtype=np.int16),
        np.asarray([global_lookup[spec.global_] for spec in specs], dtype=np.int16),
        np.asarray([spec.lambda_ for spec in specs], dtype=np.float32),
    )


def optimal_thresholds_matrix(y: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Exact F1-maximizing strict thresholds for every score column.

    This is the vectorized equivalent of ``threshold_from_train``. Candidate
    thresholds at intermediate members of a tie are masked because the rule is
    strictly ``score > threshold``.
    """
    labels = np.asarray(y, dtype=np.int8)
    values = np.asarray(scores, dtype=np.float64)
    order = np.argsort(values, axis=0, kind="mergesort")
    sorted_scores = np.take_along_axis(values, order, axis=0)
    sorted_labels = np.take_along_axis(
        np.broadcast_to(labels[:, None], values.shape), order, axis=0
    )
    cumulative_positive = np.cumsum(sorted_labels, axis=0, dtype=np.int32)
    total_positive = cumulative_positive[-1]
    n_rows, n_cols = values.shape

    tp = total_positive[None, :] - cumulative_positive
    predicted_positive = (n_rows - 1 - np.arange(n_rows, dtype=np.int32))[:, None]
    fp = predicted_positive - tp
    fn = total_positive[None, :] - tp
    denominator = 2 * tp + fp + fn
    f1 = np.divide(
        2.0 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=np.float64),
        where=denominator > 0,
    )
    last_in_tie = np.ones_like(sorted_scores, dtype=bool)
    if n_rows > 1:
        last_in_tie[:-1] = sorted_scores[:-1] < sorted_scores[1:]
    f1[~last_in_tie] = -1.0

    all_positive_denominator = n_rows + total_positive
    all_positive_f1 = np.divide(
        2.0 * total_positive,
        all_positive_denominator,
        out=np.zeros(n_cols, dtype=np.float64),
        where=all_positive_denominator > 0,
    )
    candidates = np.vstack([all_positive_f1[None, :], f1])
    best = np.argmax(candidates, axis=0)
    thresholds = np.nextafter(sorted_scores[0], -np.inf)
    ordinary = best > 0
    if ordinary.any():
        columns = np.arange(n_cols)[ordinary]
        thresholds[ordinary] = sorted_scores[best[ordinary] - 1, columns]
    return thresholds


def evaluate_score_matrix_cv(
    labels: np.ndarray,
    scores: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    n_configs = scores.shape[1]
    tp = np.zeros(n_configs, dtype=np.int32)
    fp = np.zeros(n_configs, dtype=np.int32)
    tn = np.zeros(n_configs, dtype=np.int32)
    fn = np.zeros(n_configs, dtype=np.int32)
    thresholds = np.zeros((len(folds), n_configs), dtype=np.float64)
    fold_f1 = np.zeros((len(folds), n_configs), dtype=np.float64)
    for fold_id, (train_index, test_index) in enumerate(folds):
        threshold = optimal_thresholds_matrix(labels[train_index], scores[train_index])
        thresholds[fold_id] = threshold
        prediction = scores[test_index] > threshold[None, :]
        y_test = labels[test_index, None].astype(bool)
        fold_tp = np.sum(prediction & y_test, axis=0, dtype=np.int32)
        fold_fp = np.sum(prediction & ~y_test, axis=0, dtype=np.int32)
        fold_tn = np.sum(~prediction & ~y_test, axis=0, dtype=np.int32)
        fold_fn = np.sum(~prediction & y_test, axis=0, dtype=np.int32)
        tp += fold_tp
        fp += fold_fp
        tn += fold_tn
        fn += fold_fn
        denom = 2 * fold_tp + fold_fp + fold_fn
        fold_f1[fold_id] = np.divide(
            2.0 * fold_tp,
            denom,
            out=np.zeros(n_configs, dtype=np.float64),
            where=denom > 0,
        )
    f1_denom = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "f1": np.divide(2.0 * tp, f1_denom, out=np.zeros(n_configs), where=f1_denom > 0),
        "precision": np.divide(tp, tp + fp, out=np.zeros(n_configs), where=(tp + fp) > 0),
        "recall": np.divide(tp, tp + fn, out=np.zeros(n_configs), where=(tp + fn) > 0),
        "specificity": np.divide(tn, tn + fp, out=np.zeros(n_configs), where=(tn + fp) > 0),
        "accuracy": (tp + tn) / len(labels),
        "mean_fold_f1": fold_f1.mean(axis=0),
        "std_fold_f1": fold_f1.std(axis=0, ddof=1),
        "threshold_mean": thresholds.mean(axis=0),
        "threshold_std": thresholds.std(axis=0, ddof=1),
        "threshold_min": thresholds.min(axis=0),
        "threshold_max": thresholds.max(axis=0),
    }


def evaluate_dataset(
    group: pd.DataFrame,
    output_dir: Path,
    chunk_size: int,
    cv_folds: int,
    cv_seed: int,
) -> None:
    dataset = str(group["dataset"].iloc[0])
    support = str(group["support"].iloc[0])
    key = dataset_key(dataset)
    result_path = output_dir / "datasets" / f"{key}.npz"
    baseline_path = output_dir / "baselines" / f"{key}.csv"
    timing_path = output_dir / "component_timings" / f"{key}.csv"
    metadata_path = output_dir / "metadata" / f"{key}.json"
    if (
        result_path.is_file()
        and baseline_path.is_file()
        and timing_path.is_file()
        and metadata_path.is_file()
    ):
        print(f"resume: {dataset} already complete")
        return

    local_columns = [f"algfw_local__{spec.slug}" for spec in local_grid_specs()]
    global_columns = [f"algfw_global__{spec.slug}" for spec in global_grid_specs()]
    missing = [column for column in local_columns + global_columns if column not in group]
    if missing:
        raise RuntimeError(f"{dataset}: {len(missing)} framework component columns missing")
    local = group[local_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    global_ = group[global_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(local).all() or not np.isfinite(global_).all():
        invalid_local = np.argwhere(~np.isfinite(local))
        invalid_global = np.argwhere(~np.isfinite(global_))
        examples = [
            local_columns[column]
            for _, column in invalid_local[:3]
        ] + [
            global_columns[column]
            for _, column in invalid_global[:3]
        ]
        raise RuntimeError(
            f"{dataset}: non-finite framework components; "
            f"local_cells={len(invalid_local)}, global_cells={len(invalid_global)}, "
            f"examples={examples}"
        )
    labels = group["label"].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=2)
    if len(np.unique(labels)) != 2 or counts.min() < cv_folds:
        raise RuntimeError(f"{dataset}: insufficient pair labels for {cv_folds}-fold CV")
    folds = list(
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=cv_seed).split(
            np.zeros(len(labels)), labels
        )
    )

    local_index, global_index, lambda_values = framework_index_arrays()
    n_configs = framework_grid_size()
    result_names = [
        "f1", "precision", "recall", "specificity", "accuracy",
        "mean_fold_f1", "std_fold_f1", "threshold_mean", "threshold_std",
        "threshold_min", "threshold_max",
    ]
    results = {name: np.empty(n_configs, dtype=np.float32) for name in result_names}
    counts_out = {name: np.empty(n_configs, dtype=np.int32) for name in ["tp", "fp", "tn", "fn"]}
    chunk_seconds: list[float] = []
    fusion_seconds: list[float] = []
    cv_seconds: list[float] = []
    started = time.perf_counter()
    for start in range(0, n_configs, chunk_size):
        stop = min(start + chunk_size, n_configs)
        chunk_started = time.perf_counter()
        li = local_index[start:stop]
        gi = global_index[start:stop]
        weight = lambda_values[start:stop].astype(np.float64)
        fusion_started = time.perf_counter()
        scores = local[:, li] * weight[None, :] + global_[:, gi] * (1.0 - weight[None, :])
        fusion_seconds.append(time.perf_counter() - fusion_started)
        cv_started = time.perf_counter()
        evaluated = evaluate_score_matrix_cv(labels, scores, folds)
        cv_seconds.append(time.perf_counter() - cv_started)
        for name in result_names:
            results[name][start:stop] = evaluated[name].astype(np.float32)
        for name in counts_out:
            counts_out[name][start:stop] = evaluated[name]
        chunk_seconds.append(time.perf_counter() - chunk_started)
        if start == 0 or stop == n_configs or (start // chunk_size) % 50 == 0:
            print(f"{dataset}: {stop}/{n_configs} configurations")

    baseline_rows: list[dict[str, object]] = []
    for metric in RANKING_SCORE_COLS:
        if metric not in group.columns or metric.startswith("alg_"):
            continue
        values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            continue
        one = evaluate_score_matrix_cv(labels, values[:, None], folds)
        baseline_rows.append(
            {
                "support": support,
                "dataset": dataset,
                "metric": metric,
                **{name: float(one[name][0]) for name in result_names},
                **{name: int(one[name][0]) for name in counts_out},
            }
        )

    total_seconds = time.perf_counter() - started
    component_series = (
        group["time_algfw_components_sec"]
        if "time_algfw_components_sec" in group
        else pd.Series(0.0, index=group.index)
    )
    component_seconds = float(pd.to_numeric(component_series, errors="coerce").fillna(0.0).sum())
    for directory in [result_path.parent, baseline_path.parent, timing_path.parent, metadata_path.parent]:
        directory.mkdir(parents=True, exist_ok=True)
    temporary = result_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            dataset=np.asarray(dataset),
            support=np.asarray(support),
            n_pairs=np.asarray(len(group), dtype=np.int32),
            n_positive=np.asarray(labels.sum(), dtype=np.int32),
            cv_folds=np.asarray(cv_folds, dtype=np.int16),
            cv_seed=np.asarray(cv_seed, dtype=np.int32),
            component_compute_sec=np.asarray(component_seconds),
            exhaustive_evaluation_sec=np.asarray(total_seconds),
            chunk_seconds=np.asarray(chunk_seconds, dtype=np.float32),
            fusion_seconds=np.asarray(fusion_seconds, dtype=np.float32),
            cv_seconds=np.asarray(cv_seconds, dtype=np.float32),
            **results,
            **counts_out,
        )
    os.replace(temporary, result_path)
    pd.DataFrame(baseline_rows).to_csv(baseline_path, index=False)
    timing_columns = [
        column
        for column in group.columns
        if column.startswith("time_algfw_") and column.endswith("_sec")
    ]
    timing_rows = []
    for column in timing_columns:
        values = pd.to_numeric(group[column], errors="coerce")
        finite = values[np.isfinite(values)].to_numpy(dtype=float)
        timing_rows.append(
            {
                "support": support,
                "dataset": dataset,
                "timing": column,
                "n_pairs": len(group),
                "n_finite": len(finite),
                "total_sec": float(finite.sum()) if len(finite) else np.nan,
                "mean_sec_per_pair": float(finite.mean()) if len(finite) else np.nan,
                "median_sec_per_pair": float(np.median(finite)) if len(finite) else np.nan,
                "p95_sec_per_pair": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
            }
        )
    pd.DataFrame(timing_rows).to_csv(timing_path, index=False)
    metadata = {
        "dataset": dataset,
        "support": support,
        "n_pairs": len(group),
        "n_positive": int(labels.sum()),
        "n_configurations": n_configs,
        "cv_folds": cv_folds,
        "cv_seed": cv_seed,
        "chunk_size": chunk_size,
        "component_compute_sec": component_seconds,
        "exhaustive_evaluation_sec": total_seconds,
        "component_source_columns": {"local": len(local_columns), "global": len(global_columns)},
        "baseline_metrics_evaluated": len(baseline_rows),
    }
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"saved {dataset}: {n_configs} configurations in {total_seconds:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-seed", type=int, default=42)
    args = parser.parse_args()
    if args.chunk_size < 1 or args.shard_count < 1:
        raise ValueError("chunk-size and shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")

    pair_files = sorted(args.components_root.glob("shard_*/pair_scores.csv"))
    if not pair_files:
        raise FileNotFoundError(f"No shard_*/pair_scores.csv under {args.components_root}")
    assigned = pair_files[args.shard_index :: args.shard_count]
    print(f"evaluation shard {args.shard_index + 1}/{args.shard_count}: {len(assigned)} component files")
    for pair_file in assigned:
        table = pd.read_csv(pair_file, low_memory=False)
        for _, group in table.groupby(["support", "dataset"], sort=False):
            evaluate_dataset(group, args.output_dir, args.chunk_size, args.cv_folds, args.cv_seed)


if __name__ == "__main__":
    main()

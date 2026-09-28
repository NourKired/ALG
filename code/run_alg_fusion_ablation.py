#!/usr/bin/env python3
"""Evaluate alternative local/global fusion rules from existing pair scores.

The scalar rules reuse the frozen cosine local and rational global components,
so no point clouds need to be regenerated. Rank fusion is fitted on each
training fold only. The two-feature logistic model is reported explicitly as
a supervised upper bound, not as another unsupervised similarity.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

import run_overlap_metric_large_scale as benchmark


LOCAL_COLUMN = "alg_cosine_rational_scale_1_lambda_1.0"
GLOBAL_COLUMN = "alg_cosine_rational_scale_1_lambda_0.0"
IDENTITY_COLUMNS = ["support", "dataset", "label"]
SCALAR_FUSIONS = (
    "fusion_local_only",
    "fusion_global_only",
    "fusion_equal_convex",
    "fusion_analytic_one_third",
    "fusion_selected_lambda_0.6",
    "fusion_product",
    "fusion_harmonic",
)
SPECIAL_FUSIONS = ("fusion_rank_average", "fusion_logistic_two_feature_upper_bound")
FUSION_ROLE = {
    **{name: "unsupervised_pointwise_similarity" for name in SCALAR_FUSIONS},
    "fusion_rank_average": "training_fold_calibrated_diagnostic",
    "fusion_logistic_two_feature_upper_bound": "supervised_upper_bound",
}


def find_pair_score_file(shard: Path) -> Path | None:
    for name in ("pair_scores.csv", "_checkpoint_pair_scores.csv"):
        path = shard / name
        if path.is_file() and path.stat().st_size > 0:
            return path
    return None


def add_scalar_fusions(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    local = pd.to_numeric(out[LOCAL_COLUMN], errors="coerce").to_numpy(float)
    global_score = pd.to_numeric(out[GLOBAL_COLUMN], errors="coerce").to_numpy(float)
    out["fusion_local_only"] = local
    out["fusion_global_only"] = global_score
    out["fusion_equal_convex"] = 0.5 * local + 0.5 * global_score
    out["fusion_analytic_one_third"] = (local + 2.0 * global_score) / 3.0
    out["fusion_selected_lambda_0.6"] = 0.6 * local + 0.4 * global_score
    out["fusion_product"] = local * global_score
    denominator = local + global_score
    out["fusion_harmonic"] = np.divide(
        2.0 * local * global_score,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return out


def _cdf(train: np.ndarray, values: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(train, dtype=float))
    return np.searchsorted(ordered, values, side="right") / max(len(ordered), 1)


def evaluate_special_fusions(
    group: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = group["label"].astype(int).to_numpy()
    features = group[[LOCAL_COLUMN, GLOBAL_COLUMN]].astype(float).to_numpy()
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(features, labels))
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []

    for metric in SPECIAL_FUSIONS:
        oof_score = np.zeros(len(group), dtype=float)
        oof_prediction = np.zeros(len(group), dtype=int)
        for fold, (train_idx, test_idx) in enumerate(folds):
            if metric == "fusion_rank_average":
                train_score = 0.5 * (
                    _cdf(features[train_idx, 0], features[train_idx, 0])
                    + _cdf(features[train_idx, 1], features[train_idx, 1])
                )
                test_score = 0.5 * (
                    _cdf(features[train_idx, 0], features[test_idx, 0])
                    + _cdf(features[train_idx, 1], features[test_idx, 1])
                )
                threshold = benchmark.threshold_from_train(labels[train_idx], train_score)
                prediction = (test_score > threshold).astype(int)
                classifier = "training_fold_ecdf_plus_scalar_threshold"
            else:
                model = LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=random_state,
                ).fit(features[train_idx], labels[train_idx])
                test_score = model.predict_proba(features[test_idx])[:, 1]
                prediction = (test_score >= 0.5).astype(int)
                threshold = 0.5
                classifier = "logistic_regression_two_features"
            oof_score[test_idx] = test_score
            oof_prediction[test_idx] = prediction
            y_test = labels[test_idx]
            tp, fp, tn, fn = benchmark._classification_counts(y_test, prediction)
            fold_rows.append(
                {
                    "support": group["support"].iloc[0],
                    "dataset": group["dataset"].iloc[0],
                    "metric": metric,
                    "fusion_role": FUSION_ROLE[metric],
                    "fold": fold,
                    "classifier": classifier,
                    "threshold_train": threshold,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "f1": f1_score(y_test, prediction, zero_division=0),
                    "precision": precision_score(y_test, prediction, zero_division=0),
                    "recall": recall_score(y_test, prediction, zero_division=0),
                    "roc_auc": roc_auc_score(y_test, test_score),
                }
            )
        tp, fp, tn, fn = benchmark._classification_counts(labels, oof_prediction)
        summary_rows.append(
            {
                "support": group["support"].iloc[0],
                "dataset": group["dataset"].iloc[0],
                "metric": metric,
                "fusion_role": FUSION_ROLE[metric],
                "classifier": classifier,
                "cv_folds": n_splits,
                "roc_auc": roc_auc_score(labels, oof_score),
                "average_precision": average_precision_score(labels, oof_score),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "f1": f1_score(labels, oof_prediction, zero_division=0),
                "precision": precision_score(labels, oof_prediction, zero_division=0),
                "recall": recall_score(labels, oof_prediction, zero_division=0),
                "n_pairs": len(group),
                "n_test": len(group),
                "n_test_positive": int(labels.sum()),
                "n_test_negative": int(len(labels) - labels.sum()),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    shards = sorted(path for path in args.input_root.glob("shard_*") if path.is_dir())
    if args.shard_index is not None:
        shards = [args.input_root / f"shard_{args.shard_index:03d}"]
    summary_parts: list[pd.DataFrame] = []
    fold_parts: list[pd.DataFrame] = []
    benchmark.RANKING_SCORE_COLS[:] = list(SCALAR_FUSIONS)

    for shard in shards:
        source = find_pair_score_file(shard)
        if source is None:
            continue
        columns = pd.read_csv(source, nrows=0).columns
        required = [*IDENTITY_COLUMNS, LOCAL_COLUMN, GLOBAL_COLUMN]
        missing = set(required) - set(columns)
        if missing:
            raise ValueError(f"{source} lacks {sorted(missing)}")
        table = pd.read_csv(source, usecols=required)
        table = add_scalar_fusions(table)
        for _, group in table.groupby(["support", "dataset"], sort=False):
            if not np.isfinite(group[[LOCAL_COLUMN, GLOBAL_COLUMN]].to_numpy(float)).all():
                continue
            summary, folds, error = benchmark.evaluate_dataset_pairs_cv(
                group,
                n_splits=args.folds,
                random_state=args.seed,
            )
            if error is None:
                summary["fusion_role"] = summary["metric"].map(FUSION_ROLE)
                folds["fusion_role"] = folds["metric"].map(FUSION_ROLE)
                summary_parts.append(summary)
                fold_parts.append(folds)
                special_summary, special_folds = evaluate_special_fusions(
                    group,
                    n_splits=args.folds,
                    random_state=args.seed,
                )
                summary_parts.append(special_summary)
                fold_parts.append(special_folds)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.shard_index:03d}" if args.shard_index is not None else ""
    summary_out = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    folds_out = pd.concat(fold_parts, ignore_index=True) if fold_parts else pd.DataFrame()
    summary_out.to_csv(args.output_dir / f"fusion_summary{suffix}.csv", index=False)
    folds_out.to_csv(args.output_dir / f"fusion_folds{suffix}.csv", index=False)
    print(f"Saved {len(summary_out)} dataset-method summaries from {len(shards)} shard(s).")


if __name__ == "__main__":
    main()

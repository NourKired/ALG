#!/usr/bin/env python3
"""Generate leakage-controlled neural-representation benchmark components.

Networks are trained only on each dataset's official training split.  Hidden
activation clouds are extracted only on its official test split.  A pair is
positive when two independently trained models share the same architecture and
negative when their architectures differ.  The script writes the same pair
score schema as the main ALG component campaign, including all available
baselines and all reusable ALG framework components.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from run_overlap_metric_large_scale import (
    configure_metrics,
    enable_alg_framework_components,
    global_scale_clouds,
    make_balanced_pairs,
    score_pair,
)


DATASETS = ("MNIST", "KMNIST", "USPS")
HIDDEN_SIZES = ((32,), (64,), (128,), (64, 32))


def load_torchvision_split(name: str, root: Path, train: bool):
    from torchvision import datasets

    cls = getattr(datasets, name)
    return cls(root=str(root), train=train, download=False)


def arrays(dataset) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(dataset, "data"):
        x = dataset.data
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        y = getattr(dataset, "targets", getattr(dataset, "labels", None))
        if hasattr(y, "detach"):
            y = y.detach().cpu().numpy()
        x = np.asarray(x)
        y = np.asarray(y)
    else:
        values, labels = [], []
        for image, label in dataset:
            values.append(np.asarray(image, dtype=np.float32))
            labels.append(int(label))
        x, y = np.asarray(values), np.asarray(labels)
    x = x.reshape(len(x), -1).astype(np.float32)
    maximum = float(np.max(x)) if len(x) else 1.0
    if maximum > 1.0:
        x /= maximum
    return x, y.astype(int)


def stratified_subset(
    x: np.ndarray,
    y: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if limit <= 0 or len(x) <= limit:
        return x, y
    selected: list[int] = []
    labels = np.unique(y)
    per_class = max(1, limit // len(labels))
    for label in labels:
        candidates = np.flatnonzero(y == label)
        take = min(per_class, len(candidates))
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < limit:
        remainder = np.setdiff1d(np.arange(len(y)), np.asarray(selected), assume_unique=False)
        extra = rng.choice(remainder, size=min(limit - len(selected), len(remainder)), replace=False)
        selected.extend(extra.tolist())
    selected = selected[:limit]
    rng.shuffle(selected)
    index = np.asarray(selected, dtype=int)
    return x[index], y[index]


def relu_hidden(model: MLPClassifier, x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x @ model.coefs_[0] + model.intercepts_[0])


def activation_profile_cloud(activations: np.ndarray) -> np.ndarray:
    a = np.asarray(activations, dtype=float)
    return np.column_stack(
        [
            a.mean(axis=1),
            a.std(axis=1),
            a.max(axis=1),
            np.quantile(a, 0.25, axis=1),
            np.quantile(a, 0.50, axis=1),
            np.quantile(a, 0.75, axis=1),
            (a <= 1e-12).mean(axis=1),
        ]
    )


def evaluate_dataset(args: argparse.Namespace, name: str) -> None:
    rng = np.random.default_rng(args.seed)
    train_ds = load_torchvision_split(name, args.torchvision_root, train=True)
    test_ds = load_torchvision_split(name, args.torchvision_root, train=False)
    x_train, y_train = arrays(train_ds)
    x_probe, y_probe = arrays(test_ds)
    x_train, y_train = stratified_subset(x_train, y_train, args.train_samples, rng)
    x_probe, y_probe = stratified_subset(x_probe, y_probe, args.probe_samples, rng)

    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_probe = scaler.transform(x_probe)
    model_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for hidden in HIDDEN_SIZES:
        architecture = "x".join(map(str, hidden))
        for seed in range(args.model_seeds):
            model = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=args.max_iter,
                random_state=seed,
                early_stopping=True,
                n_iter_no_change=8,
            )
            fit_started = time.perf_counter()
            model.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - fit_started
            accuracy = accuracy_score(y_probe, model.predict(x_probe))
            cloud = activation_profile_cloud(relu_hidden(model, x_probe))
            model_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "cloud": cloud,
                    "test_accuracy": float(accuracy),
                    "fit_seconds": float(fit_seconds),
                    "n_iter": int(model.n_iter_),
                }
            )

    labels = np.asarray([row["architecture"] for row in model_rows])
    clouds = global_scale_clouds([row["cloud"] for row in model_rows])
    pairs = make_balanced_pairs(labels, args.n_pairs, rng)
    rows: list[dict[str, object]] = []
    dataset_name = f"{name}_mlp_activation_clouds_external"
    for a, b, label in pairs:
        object_a = f"{labels[a]}:seed{model_rows[a]['seed']}"
        object_b = f"{labels[b]}:seed{model_rows[b]['seed']}"
        rows.append(
            {
                "support": "neural_models_external",
                "dataset": dataset_name,
                "object_a": object_a,
                "object_b": object_b,
                "class_a": labels[a],
                "class_b": labels[b],
                "label": label,
                **score_pair(
                    clouds[a], clouds[b], object_a=object_a, object_b=object_b, label=label
                ),
            }
        )

    shard_dir = args.output_root / f"shard_{args.dataset_index:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(shard_dir / "pair_scores.csv", index=False)
    metadata = {
        "dataset": dataset_name,
        "source_dataset": name,
        "support": "neural_models_external",
        "official_training_split_used_for_model_fit": True,
        "official_test_split_used_for_activation_probe": True,
        "train_and_probe_disjoint": True,
        "alg_configuration_selection_used_this_dataset": False,
        "post_hoc_external_validation": True,
        "train_samples": len(x_train),
        "probe_samples": len(x_probe),
        "architectures": ["x".join(map(str, hidden)) for hidden in HIDDEN_SIZES],
        "model_seeds": args.model_seeds,
        "n_models": len(model_rows),
        "n_pairs": len(rows),
        "positive_pairs": int(sum(row["label"] for row in rows)),
        "negative_pairs": int(len(rows) - sum(row["label"] for row in rows)),
        "mean_model_test_accuracy": float(np.mean([row["test_accuracy"] for row in model_rows])),
        "training_seconds": float(sum(row["fit_seconds"] for row in model_rows)),
        "total_seconds": float(time.perf_counter() - started),
        "models": [{k: v for k, v in row.items() if k != "cloud"} for row in model_rows],
    }
    with (shard_dir / "neural_protocol_metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    print(f"saved {dataset_name}: {len(model_rows)} models, {len(rows)} balanced pairs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchvision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-index", type=int, required=True)
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--probe-samples", type=int, default=1000)
    parser.add_argument("--model-seeds", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--n-pairs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 <= args.dataset_index < len(DATASETS):
        raise ValueError(f"dataset-index must be in [0,{len(DATASETS) - 1}]")
    if args.model_seeds < 5:
        raise ValueError("model-seeds must be >=5 for five-fold pair evaluation")
    enable_alg_framework_components()
    configure_metrics(max_points=256)
    evaluate_dataset(args, DATASETS[args.dataset_index])


if __name__ == "__main__":
    main()

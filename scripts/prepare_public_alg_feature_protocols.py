"""Download public vision datasets and create reproducible feature matrices for ALG.

This deliberately uses only CIFAR-10 and MNIST, both downloaded by torchvision.
It creates fixed, class-balanced real-reference splits and a 50% mode-drop
comparison.  The representation is standardised pixels, explicitly labelled as
such in the manifest; it is not presented as Inception, CLIP, or DINO features.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_dataset(name: str, root: Path):
    from torchvision import datasets

    if name == "CIFAR10":
        return datasets.CIFAR10(root=str(root), train=True, download=True)
    if name == "MNIST":
        return datasets.MNIST(root=str(root), train=True, download=True)
    raise ValueError(name)


def labels_of(dataset) -> np.ndarray:
    labels = getattr(dataset, "targets")
    return np.asarray(labels, dtype=int)


def to_features(dataset, indices: np.ndarray) -> np.ndarray:
    arrays = []
    for index in indices:
        image, _ = dataset[int(index)]
        arrays.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
    values = np.vstack(arrays)
    return values


def balanced_indices(labels: np.ndarray, classes: list[int], per_class: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < per_class:
            raise ValueError(f"class {label} has only {len(candidates)} examples")
        selected.extend(rng.choice(candidates, size=per_class, replace=False))
    return np.asarray(selected, dtype=int)


def write_dataset(name: str, root: Path, output: Path, per_class: int, seed: int) -> None:
    dataset = load_dataset(name, root)
    labels = labels_of(dataset)
    classes = sorted(np.unique(labels).tolist())
    rng = np.random.default_rng(seed)

    # Independent IID reference: two disjoint balanced sets.
    first = balanced_indices(labels, classes, per_class, rng)
    available = np.ones(len(labels), dtype=bool)
    available[first] = False
    second_labels = labels.copy()
    second_labels[~available] = -1
    second = balanced_indices(second_labels, classes, per_class, rng)
    # Controlled 50% support loss: the same total size from exactly half classes.
    dropped = balanced_indices(labels, classes[: len(classes) // 2], per_class * 2, rng)

    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "real_split_A.npy", to_features(dataset, first))
    np.save(destination / "real_split_B.npy", to_features(dataset, second))
    np.save(destination / "mode_drop_50.npy", to_features(dataset, dropped))
    print(f"{name}: wrote {len(first)}+{len(second)} reference and {len(dropped)} mode-drop features to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torchvision-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_class < 2:
        raise ValueError("--per-class must be >= 2")
    write_dataset("CIFAR10", args.torchvision_root, args.output_root, args.per_class, args.seed)
    write_dataset("MNIST", args.torchvision_root, args.output_root, args.per_class, args.seed + 1)


if __name__ == "__main__":
    main()

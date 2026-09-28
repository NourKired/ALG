"""Evaluate literature real/generated feature pairs with every baseline and ALG grid.

The input manifest has one row per independent evaluation unit and requires:

    dataset,model,real_features,synthetic_features

Optional metadata columns (protocol, scenario, severity, seed, embedding) are
copied verbatim to the output. Feature files may be .npy, .npz, .csv or .csv.gz.
This runner deliberately consumes features rather than silently substituting
raw pixels: FID/KID/CMMD and recent PRDC experiments depend on the embedding.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from pointset_metrics import configure_metrics
from literature_metric_suite import cmmd_distance, full_literature_scores


REQUIRED_COLUMNS = {"dataset", "model", "real_features", "synthetic_features"}


def load_features(path: Path, npz_key: str) -> np.ndarray:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        values = np.load(path, allow_pickle=False)
    elif suffixes.endswith(".npz"):
        archive = np.load(path, allow_pickle=False)
        key = npz_key if npz_key in archive.files else archive.files[0]
        values = archive[key]
    elif suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        values = pd.read_csv(path).select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError(f"Unsupported feature file: {path}")
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite 2-D feature matrix with >=2 rows: {path}")
    return values


def deterministic_sample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n <= 0 or len(x) <= n:
        return x
    rng = np.random.default_rng(seed)
    return x[np.sort(rng.choice(len(x), size=n, replace=False))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--npz-key", type=str, default="features")
    parser.add_argument("--metric-max-points", type=int, default=256)
    parser.add_argument("--geometry-repeats", type=int, default=16)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    missing = REQUIRED_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    if args.task_index is not None:
        if args.task_index < 0 or args.task_index >= len(manifest):
            raise IndexError(f"task-index {args.task_index} outside [0, {len(manifest)})")
        manifest = manifest.iloc[[args.task_index]]

    configure_metrics(
        manifold_k=args.k,
        max_points=args.metric_max_points,
        geometry_repeats=args.geometry_repeats,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    base_dir = args.manifest.resolve().parent

    for manifest_index, row in manifest.iterrows():
        started = time.perf_counter()
        try:
            real_path = Path(str(row["real_features"])).expanduser()
            synthetic_path = Path(str(row["synthetic_features"])).expanduser()
            if not real_path.is_absolute():
                real_path = base_dir / real_path
            if not synthetic_path.is_absolute():
                synthetic_path = base_dir / synthetic_path
            real = deterministic_sample(load_features(real_path, args.npz_key), args.samples, args.seed + int(manifest_index) * 2)
            synthetic = deterministic_sample(
                load_features(synthetic_path, args.npz_key),
                args.samples,
                args.seed + int(manifest_index) * 2 + 1,
            )
            if real.shape[1] != synthetic.shape[1]:
                raise ValueError(
                    f"Feature dimensions differ: real={real.shape[1]}, synthetic={synthetic.shape[1]}"
                )
            metric_values = full_literature_scores(real, synthetic, args.k, args.metric_max_points)
            embedding = str(row.get("embedding", "")).lower()
            if "clip" in embedding:
                cmmd_started = time.perf_counter()
                cmmd = cmmd_distance(real, synthetic, sigma=10.0)
                metric_values["cmmd_distance"] = cmmd
                metric_values["cmmd_similarity"] = -cmmd
                metric_values["cmmd_sigma"] = 10.0
                metric_values["time_cmmd_sec"] = time.perf_counter() - cmmd_started
                metric_values["cmmd_implementation_status"] = (
                    "official_formula_on_manifest_clip_features"
                )
            if "inception" in embedding and "clean-fid" in embedding:
                similarity = metric_values.get("baseline_gaussian_frechet_similarity", np.nan)
                metric_values["clean_fid_distance"] = -float(similarity)
                metric_values["clean_kid_similarity"] = metric_values.get(
                    "baseline_kid_polynomial_similarity", np.nan
                )
                metric_values["clean_fid_implementation_status"] = (
                    "official_distance_on_manifest_clean_fid_features"
                )
            metadata = {
                key: value
                for key, value in row.to_dict().items()
                if key not in {"real_features", "synthetic_features"}
            }
            rows.append(
                {
                    "manifest_index": int(manifest_index),
                    **metadata,
                    "n_real": len(real),
                    "n_synthetic": len(synthetic),
                    "feature_dimension": real.shape[1],
                    "metric_max_points": args.metric_max_points,
                    "elapsed_total_sec": time.perf_counter() - started,
                    **metric_values,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "manifest_index": int(manifest_index),
                    "dataset": row.get("dataset", ""),
                    "model": row.get("model", ""),
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_total_sec": time.perf_counter() - started,
                }
            )

    stem = f"task_{args.task_index:04d}" if args.task_index is not None else "all_tasks"
    pd.DataFrame(rows).to_csv(args.output_dir / f"{stem}_scores.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output_dir / f"{stem}_failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} manifest row(s) failed; see {stem}_failures.csv")
    print(f"Saved {len(rows)} literature evaluation row(s) to {args.output_dir}")


if __name__ == "__main__":
    main()

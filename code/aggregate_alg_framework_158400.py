#!/usr/bin/env python3
"""Aggregate the complete ALG framework campaign and freeze valid selections."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from alg_framework_grid import framework_grid_size, iter_framework_specs


RESULT_FIELDS = (
    "f1", "precision", "recall", "specificity", "accuracy", "mean_fold_f1",
    "std_fold_f1", "threshold_mean", "threshold_std", "threshold_min", "threshold_max",
    "tp", "fp", "tn", "fn",
)


def configuration_frame() -> pd.DataFrame:
    return pd.DataFrame([spec.as_record() for spec in iter_framework_specs()])


def fractional_wins(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_methods = matrix.shape[0]
    fractional = np.zeros(n_methods, dtype=np.float64)
    strict = np.zeros(n_methods, dtype=np.int32)
    tied = np.zeros(n_methods, dtype=np.int32)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        best = np.nanmax(values)
        winners = np.flatnonzero(finite & np.isclose(values, best, rtol=0.0, atol=1e-12))
        fractional[winners] += 1.0 / len(winners)
        if len(winners) == 1:
            strict[winners[0]] += 1
        else:
            tied[winners] += 1
    return fractional, strict, tied


def summarize_matrix(configs: pd.DataFrame, matrix: np.ndarray, prefix: str = "") -> pd.DataFrame:
    wins, strict, tied = fractional_wins(matrix)
    out = configs.copy()
    out[f"{prefix}n_datasets"] = np.isfinite(matrix).sum(axis=1)
    out[f"{prefix}mean_f1"] = np.nanmean(matrix, axis=1)
    out[f"{prefix}median_f1"] = np.nanmedian(matrix, axis=1)
    out[f"{prefix}std_f1"] = np.nanstd(matrix, axis=1, ddof=1)
    out[f"{prefix}min_f1"] = np.nanmin(matrix, axis=1)
    out[f"{prefix}n_dataset_wins"] = wins
    out[f"{prefix}n_strict_wins"] = strict
    out[f"{prefix}n_tied_wins"] = tied
    return out.sort_values(
        [f"{prefix}mean_f1", f"{prefix}n_dataset_wins"], ascending=[False, False]
    )


def load_results(results_root: Path) -> tuple[dict[str, dict[str, object]], pd.DataFrame]:
    records: dict[str, dict[str, object]] = {}
    for path in sorted((results_root / "datasets").glob("*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            dataset = str(archive["dataset"].item())
            records[dataset] = {
                "path": path,
                "support": str(archive["support"].item()),
                "n_pairs": int(archive["n_pairs"].item()),
                "component_compute_sec": float(archive["component_compute_sec"].item()),
                "exhaustive_evaluation_sec": float(archive["exhaustive_evaluation_sec"].item()),
                "fusion_sec": float(archive["fusion_seconds"].sum()),
                "cv_sec": float(archive["cv_seconds"].sum()),
            }
    baseline_frames = []
    for path in sorted((results_root / "baselines").glob("*.csv")):
        try:
            frame = pd.read_csv(path)
        except EmptyDataError:
            continue
        if not frame.empty:
            baseline_frames.append(frame)
    baselines = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    return records, baselines


def result_vector(record: dict[str, object], field: str) -> np.ndarray:
    with np.load(Path(record["path"]), allow_pickle=False) as archive:
        return archive[field].astype(np.float32, copy=False)


def save_matrix(path: Path, matrix: np.ndarray, datasets: list[str], supports: list[str], partitions: list[str]) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            f1=matrix.astype(np.float32, copy=False),
            dataset=np.asarray(datasets),
            support=np.asarray(supports),
            partition=np.asarray(partitions),
        )


def baseline_matrix(baselines: pd.DataFrame, datasets: list[str]) -> tuple[list[str], np.ndarray]:
    methods = sorted(baselines["metric"].astype(str).unique()) if not baselines.empty else []
    lookup = baselines.pivot_table(index="metric", columns="dataset", values="f1", aggfunc="first")
    return methods, lookup.reindex(index=methods, columns=datasets).to_numpy(dtype=np.float32)


def timing_estimates(configs: pd.DataFrame, results_root: Path, runtime: pd.DataFrame) -> pd.DataFrame:
    timing_paths = sorted((results_root / "component_timings").glob("*.csv"))
    if not timing_paths:
        return pd.DataFrame()
    timings = pd.concat([pd.read_csv(path) for path in timing_paths], ignore_index=True)
    grouped = timings.groupby("timing", as_index=True).agg(
        total_sec=("total_sec", "sum"), n_finite=("n_finite", "sum")
    )
    mean = (grouped["total_sec"] / grouped["n_finite"]).to_dict()
    fusion_per_config = runtime["fusion_sec"].sum() / max(
        framework_grid_size() * runtime["n_pairs"].sum(), 1
    )
    cv_per_config = runtime["cv_sec"].sum() / max(framework_grid_size() * len(runtime), 1)

    rows = []
    for row in configs.to_dict("records"):
        weight = float(row["lambda"])
        use_local = weight > 0.0
        use_global = weight < 1.0
        geometries = set()
        if use_local:
            geometries.add(row["local_geometry"])
        if use_global:
            geometries.add(row["global_geometry"])
        distance = sum(mean.get(f"time_algfw_distance_{geometry}_sec", np.nan) for geometry in geometries)
        local = (
            mean.get(f"time_algfw_local__{row['metric'].split('_G-')[0].split('algfw_L-')[1]}_sec", np.nan)
            if use_local else 0.0
        )
        global_raw = (
            mean.get(
                f"time_algfw_global_raw__{row['global_family']}_{row['global_geometry']}_sec", np.nan
            ) if use_global else 0.0
        )
        global_normalization = (
            mean.get(
                "time_algfw_global_normalization__"
                f"{row['global_family']}_{row['global_geometry']}_{row['global_psi']}_scale{row['global_scale']:g}_sec",
                np.nan,
            ) if use_global else 0.0
        )
        compute = float(np.nansum([distance, local, global_raw, global_normalization, fusion_per_config]))
        rows.append(
            {
                "metric": row["metric"],
                "estimated_distance_sec_per_pair": distance,
                "estimated_local_sec_per_pair": local,
                "estimated_global_raw_sec_per_pair": global_raw,
                "estimated_global_normalization_sec_per_pair": global_normalization,
                "measured_marginal_fusion_sec_per_pair": fusion_per_config,
                "estimated_metric_compute_sec_per_pair": compute,
                "measured_cv_evaluation_sec_per_config_per_dataset": cv_per_config,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=None,
        help="Existing per-dataset metric_summary CSV from which non-ALG baselines are reused.",
    )
    parser.add_argument(
        "--baseline-efficiency",
        type=Path,
        default=None,
        help="Existing per-dataset baseline timing CSV to copy into the consolidated results.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-development", type=int, default=60)
    parser.add_argument("--expected-test", type=int, default=183)
    parser.add_argument("--top-per-category", type=int, default=100)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    partition = pd.read_csv(args.partition)
    records, baselines = load_results(args.results_root)
    if args.baseline_summary is not None:
        historical = pd.read_csv(args.baseline_summary, low_memory=False)
        required = {"support", "dataset", "metric", "f1"}
        if not required.issubset(historical.columns):
            raise ValueError(f"baseline summary lacks columns: {sorted(required - set(historical.columns))}")
        historical = historical[
            ~historical["metric"].astype(str).str.startswith("alg_")
        ].copy()
        baselines = historical
    missing = sorted(set(partition["dataset"].astype(str)) - set(records))
    if missing:
        raise RuntimeError(f"Missing {len(missing)} dataset result archives; first: {missing[:10]}")
    partition = partition[partition["dataset"].astype(str).isin(records)].copy()
    counts = partition["partition"].value_counts().to_dict()
    if (counts.get("development", 0), counts.get("test", 0)) != (
        args.expected_development, args.expected_test
    ):
        raise RuntimeError(f"Partition count mismatch: {counts}")

    configs = configuration_frame()
    if len(configs) != framework_grid_size():
        raise AssertionError("framework manifest size mismatch")
    datasets = partition["dataset"].astype(str).tolist()
    supports = partition["support"].astype(str).tolist()
    partitions = partition["partition"].astype(str).tolist()
    f1 = np.column_stack([result_vector(records[dataset], "f1") for dataset in datasets])
    expected_shape = (framework_grid_size(), len(datasets))
    if f1.shape != expected_shape:
        raise RuntimeError(f"Unexpected complete F1 matrix shape: {f1.shape}")
    development_mask = np.asarray(partitions) == "development"
    test_mask = np.asarray(partitions) == "test"
    f1_development = f1[:, development_mask]
    f1_test = f1[:, test_mask]
    test_datasets = np.asarray(datasets)[test_mask].tolist()
    test_supports = np.asarray(supports)[test_mask].tolist()

    grid_size = framework_grid_size()
    configs.to_csv(args.output_dir / f"framework_configurations_{grid_size}.csv", index=False)
    save_matrix(args.output_dir / "f1_matrix_all243.npz", f1, datasets, supports, partitions)
    save_matrix(
        args.output_dir / "f1_matrix_exploratory_test183.npz",
        f1_test,
        test_datasets,
        test_supports,
        ["test"] * len(test_datasets),
    )
    development_ranking = summarize_matrix(configs, f1_development)
    exploratory_ranking = summarize_matrix(configs, f1_test)
    development_ranking.to_csv(args.output_dir / "framework_development_ranking.csv", index=False)
    exploratory_ranking.to_csv(args.output_dir / "framework_exploratory_test_ranking.csv", index=False)

    category_summary = configs.copy()
    top_parts = []
    for support in sorted(set(test_supports)):
        mask = np.asarray(test_supports) == support
        summary = summarize_matrix(configs, f1_test[:, mask])
        suffix = "".join(c if c.isalnum() else "_" for c in support)
        indexed = summary.set_index("metric")
        category_summary[f"mean_f1__{suffix}"] = category_summary["metric"].map(indexed["mean_f1"])
        category_summary[f"n_dataset_wins__{suffix}"] = category_summary["metric"].map(indexed["n_dataset_wins"])
        summary.insert(0, "support", support)
        top_parts.append(summary.head(args.top_per_category))
    category_summary.to_csv(args.output_dir / "framework_exploratory_category_summary.csv.gz", index=False)
    pd.concat(top_parts, ignore_index=True).to_csv(
        args.output_dir / "framework_top_by_category.csv", index=False
    )

    selected_index = int(np.nanargmax(np.nanmean(f1_development, axis=1)))
    selected_metric = str(configs.iloc[selected_index]["metric"])
    selected_global = pd.DataFrame(
        {
            "support": test_supports,
            "dataset": test_datasets,
            "metric": selected_metric,
            "f1": f1_test[selected_index],
            "selection_partition": "development",
            "evaluation_partition": "test",
        }
    )
    selected_global.to_csv(args.output_dir / "frozen_global_selection_test183.csv", index=False)

    category_selection_rows = []
    category_test_rows = []
    dev_supports = np.asarray(supports)[development_mask]
    for support in sorted(set(dev_supports)):
        dev_mask = dev_supports == support
        if not dev_mask.any():
            continue
        enough_for_category_selection = int(dev_mask.sum()) >= 5
        index = (
            int(np.nanargmax(np.nanmean(f1_development[:, dev_mask], axis=1)))
            if enough_for_category_selection
            else selected_index
        )
        category_selection_rows.append(
            {
                "support": support,
                "metric": configs.iloc[index]["metric"],
                "configuration_index": index,
                "n_development_datasets": int(dev_mask.sum()),
                "development_mean_f1": float(np.nanmean(f1_development[index, dev_mask])),
                "selection_status": (
                    "category_specific_frozen_on_development"
                    if enough_for_category_selection
                    else "insufficient_category_development_data_global_fallback"
                ),
            }
        )
        for dataset_index, (dataset, test_support) in enumerate(zip(test_datasets, test_supports)):
            if test_support == support:
                category_test_rows.append(
                    {
                        "support": support,
                        "dataset": dataset,
                        "metric": configs.iloc[index]["metric"],
                        "f1": float(f1_test[index, dataset_index]),
                        "selection_status": (
                            "category_specific_frozen_on_development"
                            if enough_for_category_selection
                            else "insufficient_category_development_data_global_fallback"
                        ),
                    }
                )
    pd.DataFrame(category_selection_rows).to_csv(
        args.output_dir / "frozen_category_selections.csv", index=False
    )
    pd.DataFrame(category_test_rows).to_csv(
        args.output_dir / "frozen_category_selection_test_results.csv", index=False
    )

    test_baselines = baselines[baselines["dataset"].astype(str).isin(test_datasets)].copy()
    test_baselines.to_csv(args.output_dir / "baseline_results_test183.csv.gz", index=False)
    if args.baseline_efficiency is not None:
        baseline_efficiency = pd.read_csv(args.baseline_efficiency, low_memory=False)
        if "dataset" in baseline_efficiency:
            baseline_efficiency = baseline_efficiency[
                baseline_efficiency["dataset"].astype(str).isin(test_datasets)
            ]
        if "metric" in baseline_efficiency:
            baseline_efficiency = baseline_efficiency[
                ~baseline_efficiency["metric"].astype(str).str.startswith("alg_")
            ]
        baseline_efficiency.to_csv(
            args.output_dir / "baseline_efficiency_test183.csv.gz", index=False
        )
    baseline_methods, base_matrix = baseline_matrix(test_baselines, test_datasets)
    framework_best = np.nanmax(f1_test, axis=0)
    baseline_best = np.nanmax(base_matrix, axis=0) if len(base_matrix) else np.full(len(test_datasets), -np.inf)
    overall_best = np.maximum(framework_best, baseline_best)
    fw_fractional = np.zeros(len(configs), dtype=float)
    fw_strict = np.zeros(len(configs), dtype=int)
    fw_tied = np.zeros(len(configs), dtype=int)
    base_fractional = np.zeros(len(baseline_methods), dtype=float)
    base_strict = np.zeros(len(baseline_methods), dtype=int)
    base_tied = np.zeros(len(baseline_methods), dtype=int)
    for column, best in enumerate(overall_best):
        fw = np.flatnonzero(np.isclose(f1_test[:, column], best, rtol=0.0, atol=1e-12))
        base = np.flatnonzero(np.isclose(base_matrix[:, column], best, rtol=0.0, atol=1e-12)) if len(base_matrix) else np.asarray([], dtype=int)
        total = len(fw) + len(base)
        if not total:
            continue
        fw_fractional[fw] += 1.0 / total
        base_fractional[base] += 1.0 / total
        if total == 1:
            if len(fw): fw_strict[fw[0]] += 1
            else: base_strict[base[0]] += 1
        else:
            fw_tied[fw] += 1
            base_tied[base] += 1
    all_methods = configs[["metric"]].copy()
    all_methods["method_type"] = "ALG_framework"
    all_methods["n_datasets"] = np.isfinite(f1_test).sum(axis=1)
    all_methods["mean_f1"] = np.nanmean(f1_test, axis=1)
    all_methods["median_f1"] = np.nanmedian(f1_test, axis=1)
    all_methods["n_dataset_wins"] = fw_fractional
    all_methods["n_strict_wins"] = fw_strict
    all_methods["n_tied_wins"] = fw_tied
    base_rows = pd.DataFrame(
        {
            "metric": baseline_methods,
            "method_type": "baseline",
            "n_datasets": np.isfinite(base_matrix).sum(axis=1) if len(base_matrix) else [],
            "mean_f1": np.nanmean(base_matrix, axis=1) if len(base_matrix) else [],
            "median_f1": np.nanmedian(base_matrix, axis=1) if len(base_matrix) else [],
            "n_dataset_wins": base_fractional,
            "n_strict_wins": base_strict,
            "n_tied_wins": base_tied,
        }
    )
    all_methods = pd.concat([all_methods, base_rows], ignore_index=True).sort_values(
        ["mean_f1", "n_dataset_wins"], ascending=[False, False]
    )
    all_methods.to_csv(
        args.output_dir / f"all_{grid_size}_plus_baselines_exploratory_leaderboard.csv",
        index=False,
    )

    confirmatory_matrix = np.vstack([f1_test[selected_index][None, :], base_matrix])
    confirmatory_methods = [selected_metric] + baseline_methods
    conf_wins, conf_strict, conf_tied = fractional_wins(confirmatory_matrix)
    confirmatory = pd.DataFrame(
        {
            "metric": confirmatory_methods,
            "method_type": ["ALG_selected_on_development"] + ["baseline"] * len(baseline_methods),
            "n_datasets": np.isfinite(confirmatory_matrix).sum(axis=1),
            "mean_f1": np.nanmean(confirmatory_matrix, axis=1),
            "median_f1": np.nanmedian(confirmatory_matrix, axis=1),
            "n_dataset_wins": conf_wins,
            "n_strict_wins": conf_strict,
            "n_tied_wins": conf_tied,
        }
    ).sort_values(["mean_f1", "n_dataset_wins"], ascending=[False, False])
    confirmatory.to_csv(args.output_dir / "confirmatory_selected_alg_plus_baselines.csv", index=False)

    runtime = pd.DataFrame(
        [
            {
                "support": records[dataset]["support"],
                "dataset": dataset,
                "partition": partition.set_index("dataset").loc[dataset, "partition"],
                **{key: value for key, value in records[dataset].items() if key not in {"path", "support"}},
            }
            for dataset in datasets
        ]
    )
    runtime.to_csv(args.output_dir / "framework_dataset_runtime.csv", index=False)
    efficiency = timing_estimates(configs, args.results_root, runtime)
    efficiency.to_csv(args.output_dir / "framework_configuration_efficiency.csv", index=False)

    selected_all_fields = {field: [] for field in RESULT_FIELDS}
    for dataset in test_datasets:
        with np.load(Path(records[dataset]["path"]), allow_pickle=False) as archive:
            for field in RESULT_FIELDS:
                selected_all_fields[field].append(archive[field][selected_index].item())
    for field, values in selected_all_fields.items():
        selected_global[field] = values
    selected_global.to_csv(args.output_dir / "frozen_global_selection_test183_all_metrics.csv", index=False)

    summary = {
        "n_configurations": len(configs),
        "n_development_datasets": int(development_mask.sum()),
        "n_test_datasets": int(test_mask.sum()),
        "n_total_evaluations_test": int(len(configs) * test_mask.sum()),
        "selected_global_configuration_index": selected_index,
        "selected_global_metric": selected_metric,
        "selection_rule": "maximum macro F1 across the 60 development datasets",
        "confirmatory_file": "confirmatory_selected_alg_plus_baselines.csv",
        "exploratory_warning": (
            f"The exhaustive {grid_size:,} x {test_mask.sum()} test ranking is exploratory "
            "and must not be reported as a frozen confirmatory selection."
        ),
        "files_preserve_all_per_dataset_metrics": str(args.results_root / "datasets"),
    }
    with (args.output_dir / "campaign_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(
        f"Aggregated {len(configs)} configurations over {development_mask.sum()} development "
        f"and {test_mask.sum()} test datasets. Frozen selection: {selected_metric}"
    )


if __name__ == "__main__":
    main()

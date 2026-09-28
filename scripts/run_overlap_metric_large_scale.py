#!/usr/bin/env python3
"""
Large-scale experiments for the local-overlap metric.

The script compares local overlap with pooled, geometric, distributional,
optimal-transport, manifold, and topological point-set baselines. Directional
overlaps are diagnostics; every ranked score follows "larger is more similar".
Heavy datasets and optional metrics are skipped unless --strict is set.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError
from sklearn.datasets import load_breast_cancer, load_digits, load_iris, load_wine
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from alg_similarity.pointset_metrics import (
    CONFIG as POINTSET_CONFIG,
    METRIC_FUNCTIONS,
    configure_metrics,
)
from alg_similarity.alg_metric_grid import alg_grid_metric_names, alg_grid_scores
from alg_similarity.alg_frozen_runtime import FROZEN_ALG_RUNTIME_METRIC, benchmark_frozen_alg_pair
from alg_similarity.alg_star_end_to_end_runtime import (
    ALG_STAR_METRIC,
    INITIAL_ALG_METRIC,
    benchmark_pair as benchmark_alg_star_pair,
    configure_runtime as configure_alg_star_runtime,
)
from alg_similarity.alg_tuned_baseline_grid import (
    tuned_baseline_grid_scores,
    tuned_baseline_metric_names,
)
from alg_similarity.alg_framework_grid import (
    alg_framework_components,
    global_grid_specs as alg_framework_global_specs,
    local_grid_specs as alg_framework_local_specs,
)


PAIR_SCORE_COLS = [
    "overlap_sym",
    "overlap_f1",
    "overlap_cosine_sym",
    "overlap_cosine_a_to_b",
    "overlap_cosine_b_to_a",
    "overlap_cosine_plus_pooled_cosine",
    "overlap_euclidean_plus_pooled_euclidean",
    "rank_overlap_sym",
    "rank_overlap_a_to_b",
    "rank_overlap_b_to_a",
    "rank_overlap_f1",
    "rank_overlap_f1_a_to_b",
    "rank_overlap_f1_b_to_a",
    "overlap_a_to_b",
    "overlap_b_to_a",
    "cosine_pooled",
    "neg_euclidean_pooled",
    *METRIC_FUNCTIONS.keys(),
]

RANKING_SCORE_COLS = [
    "overlap_sym",
    "overlap_f1",
    "overlap_cosine_sym",
    "overlap_cosine_plus_pooled_cosine",
    "overlap_euclidean_plus_pooled_euclidean",
    "rank_overlap_sym",
    "rank_overlap_f1",
    "cosine_pooled",
    "neg_euclidean_pooled",
    *METRIC_FUNCTIONS.keys(),
]

# None means compute all metrics. During --resume this is narrowed to the
# missing columns for the dataset currently being processed.
ACTIVE_RANKING_SCORE_COLS: set[str] | None = None
ACTIVE_PAIR_RESUME_PLAN: "PairMetricResumePlan | None" = None
ALG_GRID_SCORE_COLS = alg_grid_metric_names()
ALG_GRID_SCORE_SET = set(ALG_GRID_SCORE_COLS)
TUNED_BASELINE_SCORE_COLS = tuned_baseline_metric_names()
TUNED_BASELINE_SCORE_SET = set(TUNED_BASELINE_SCORE_COLS)
ALG_FRAMEWORK_LOCAL_SCORE_COLS = [
    f"algfw_local__{spec.slug}" for spec in alg_framework_local_specs()
]
ALG_FRAMEWORK_GLOBAL_SCORE_COLS = [
    f"algfw_global__{spec.slug}" for spec in alg_framework_global_specs()
]
ALG_FRAMEWORK_COMPONENT_SCORE_COLS = (
    ALG_FRAMEWORK_LOCAL_SCORE_COLS + ALG_FRAMEWORK_GLOBAL_SCORE_COLS
)
ALG_FRAMEWORK_COMPONENT_SCORE_SET = set(ALG_FRAMEWORK_COMPONENT_SCORE_COLS)


def enable_alg_grid() -> None:
    """Add the complete ALG grid to persisted and ranked score columns."""
    for metric in ALG_GRID_SCORE_COLS:
        if metric not in PAIR_SCORE_COLS:
            PAIR_SCORE_COLS.append(metric)
        if metric not in RANKING_SCORE_COLS:
            RANKING_SCORE_COLS.append(metric)


def enable_tuned_baseline_grid() -> None:
    """Add development-selectable competitor variants to persisted scores."""
    for metric in TUNED_BASELINE_SCORE_COLS:
        if metric not in PAIR_SCORE_COLS:
            PAIR_SCORE_COLS.append(metric)
        if metric not in RANKING_SCORE_COLS:
            RANKING_SCORE_COLS.append(metric)


def enable_frozen_alg_runtime() -> None:
    """Add an isolated implementation so resume mode measures it explicitly."""
    if FROZEN_ALG_RUNTIME_METRIC not in PAIR_SCORE_COLS:
        PAIR_SCORE_COLS.append(FROZEN_ALG_RUNTIME_METRIC)
    if FROZEN_ALG_RUNTIME_METRIC not in RANKING_SCORE_COLS:
        RANKING_SCORE_COLS.append(FROZEN_ALG_RUNTIME_METRIC)


def enable_alg_star_end_to_end_runtime() -> None:
    """Enable isolated, cache-free timing of ALG* and Initial ALG."""
    for metric in (ALG_STAR_METRIC, INITIAL_ALG_METRIC):
        if metric not in PAIR_SCORE_COLS:
            PAIR_SCORE_COLS.append(metric)
        if metric not in RANKING_SCORE_COLS:
            RANKING_SCORE_COLS.append(metric)


def enable_alg_framework_components() -> None:
    """Persist the 64 local and 240 global reusable framework components."""
    for metric in ALG_FRAMEWORK_COMPONENT_SCORE_COLS:
        if metric not in PAIR_SCORE_COLS:
            PAIR_SCORE_COLS.append(metric)
        if metric not in RANKING_SCORE_COLS:
            RANKING_SCORE_COLS.append(metric)

# Every other benchmark score follows "larger means more similar". A rank is
# deliberately the opposite: rank 1 is the best match for an anchor object.
LOWER_IS_MORE_SIMILAR = {"rank_overlap_sym", "rank_overlap_f1"}


def set_active_ranking_metrics(metrics: Iterable[str] | None) -> None:
    global ACTIVE_RANKING_SCORE_COLS
    ACTIVE_RANKING_SCORE_COLS = None if metrics is None else set(metrics)


def metric_is_active(metric: str) -> bool:
    return ACTIVE_RANKING_SCORE_COLS is None or metric in ACTIVE_RANKING_SCORE_COLS


def oriented_scores(metric: str, scores: np.ndarray) -> np.ndarray:
    """Orient scores so a larger value always denotes stronger similarity."""
    values = np.asarray(scores, dtype=float)
    return -values if metric in LOWER_IS_MORE_SIMILAR else values


def canonical_pair_value(value: object) -> str:
    """Stable CSV/in-memory representation for pair identity fields."""
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        number = float(value)
        return str(int(number)) if number.is_integer() else format(number, ".17g")
    text = str(value)
    try:
        number = float(text)
        if math.isfinite(number):
            return str(int(number)) if number.is_integer() else format(number, ".17g")
    except (TypeError, ValueError):
        pass
    return text


def pair_identity(object_a: object, object_b: object, label: object) -> tuple[str, str, str]:
    return tuple(canonical_pair_value(value) for value in (object_a, object_b, label))


def add_rank_overlap_scores(pair_scores: pd.DataFrame) -> pd.DataFrame:
    """Rank each overlap score among the observed partners of each endpoint.

    The rank is 1 for the largest overlap score. Ties receive their usual
    mid-rank. With sampled pairs, the candidate set is the sampled pair graph;
    with ``--n-pairs all``, it is every available partner in the dataset.
    """
    out = pair_scores.copy()
    required = {"object_a", "object_b"}
    if out.empty or not required.issubset(out.columns):
        return out

    a = out["object_a"].map(canonical_pair_value)
    b = out["object_b"].map(canonical_pair_value)
    for source_metric, rank_metric in [
        ("overlap_sym", "rank_overlap_sym"),
        ("overlap_f1", "rank_overlap_f1"),
    ]:
        if source_metric not in out.columns:
            continue
        base = out[["object_a", "object_b", source_metric]].copy()
        base["object_a"] = base["object_a"].map(canonical_pair_value)
        base["object_b"] = base["object_b"].map(canonical_pair_value)
        base[source_metric] = pd.to_numeric(base[source_metric], errors="coerce")
        # Duplicate sampled pairs count as one candidate partner, not multiple
        # copies that would artificially change its rank.
        forward = base.rename(columns={"object_a": "anchor", "object_b": "partner", source_metric: "score"})
        backward = base.rename(columns={"object_b": "anchor", "object_a": "partner", source_metric: "score"})
        candidates = pd.concat([forward, backward], ignore_index=True)
        candidates = candidates.groupby(["anchor", "partner"], as_index=False)["score"].mean()
        candidates["rank"] = candidates.groupby("anchor")["score"].rank(
            ascending=False, method="average", na_option="bottom"
        )
        rank_lookup = {
            (row.anchor, row.partner): float(row.rank)
            for row in candidates.itertuples(index=False)
            if math.isfinite(float(row.rank))
        }
        a_to_b = "rank_overlap_a_to_b" if rank_metric == "rank_overlap_sym" else "rank_overlap_f1_a_to_b"
        b_to_a = "rank_overlap_b_to_a" if rank_metric == "rank_overlap_sym" else "rank_overlap_f1_b_to_a"
        out[a_to_b] = [rank_lookup.get((x, y), math.nan) for x, y in zip(a, b)]
        out[b_to_a] = [rank_lookup.get((y, x), math.nan) for x, y in zip(a, b)]
        out[rank_metric] = 0.5 * (out[a_to_b] + out[b_to_a])
    return out


class PairMetricResumePlan:
    """FIFO ledger of missing metrics for every occurrence of an exact pair."""

    def __init__(self, pair_scores: pd.DataFrame):
        self._queues: dict[tuple[str, str, str], deque[set[str]]] = defaultdict(deque)
        if pair_scores.empty:
            return
        for _, row in pair_scores.iterrows():
            missing = {
                metric
                for metric in RANKING_SCORE_COLS
                if metric not in pair_scores.columns
                or not math.isfinite(float(pd.to_numeric(row.get(metric), errors="coerce")))
            }
            self._queues[pair_identity(row.get("object_a"), row.get("object_b"), row.get("label"))].append(missing)

    def missing_for(self, object_a: object, object_b: object, label: object) -> set[str]:
        queue = self._queues.get(pair_identity(object_a, object_b, label))
        return queue.popleft() if queue else set(RANKING_SCORE_COLS)


def set_active_pair_resume_plan(pair_scores: pd.DataFrame | None) -> None:
    global ACTIVE_PAIR_RESUME_PLAN
    ACTIVE_PAIR_RESUME_PLAN = (
        PairMetricResumePlan(pair_scores)
        if pair_scores is not None and not pair_scores.empty
        else None
    )

MAX_GLUE_PAIR_TASKS = [
    "stsb",
    "mrpc",
    "qqp",
    "mnli",
    "qnli",
    "rte",
    "wnli",
]

MAX_TUDATASETS = [
    "MUTAG",
    "PROTEINS",
    "ENZYMES",
    "NCI1",
    "NCI109",
    "DD",
    "Mutagenicity",
    "AIDS",
    "BZR",
    "COX2",
    "DHFR",
    "PTC_MR",
    "PTC_MM",
    "PTC_FR",
    "PTC_FM",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
    "REDDIT-MULTI-12K",
    "COLLAB",
]

MAX_TORCHVISION_DATASETS = [
    "MNIST",
    "FashionMNIST",
    "KMNIST",
    "EMNIST-Balanced",
    "QMNIST",
    "USPS",
    "CIFAR5",
    "CIFAR10",
    "CIFAR100",
    "SVHN",
    "STL10",
    "Caltech101",
    "Caltech256",
    "Flowers102",
    "OxfordIIITPet",
    "Food101",
    "CelebA-Identity",
]


@dataclass
class DatasetResult:
    support: str
    dataset: str
    pairs: pd.DataFrame
    error: str | None = None
    elapsed_sec: float = math.nan
    cpu_sec: float = math.nan
    peak_rss_mb: float = math.nan


def overlap_percentage_3d(
    cloud_a: np.ndarray, cloud_b: np.ndarray
) -> tuple[float, float, float, float]:
    """Return arithmetic and F1 aggregations of directional local coverage."""
    cloud_a = np.asarray(cloud_a, dtype=float)
    cloud_b = np.asarray(cloud_b, dtype=float)
    if cloud_a.size == 0 or cloud_b.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    if cloud_a.shape[0] < 2 or cloud_b.shape[0] < 2:
        return 0.0, 0.0, 0.0, 0.0

    try:
        nn_a = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(cloud_a)
        da, _ = nn_a.kneighbors(cloud_a)
        eps_a = da[:, 1]

        nn_b = NearestNeighbors(n_neighbors=2, algorithm="brute").fit(cloud_b)
        db, _ = nn_b.kneighbors(cloud_b)
        eps_b = db[:, 1]

        nn_b2 = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(cloud_b)
        dist_a2b, _ = nn_b2.kneighbors(cloud_a)
        overlap_a = float(np.mean(dist_a2b[:, 0] <= eps_a))

        nn_a2 = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(cloud_a)
        dist_b2a, _ = nn_a2.kneighbors(cloud_b)
        overlap_b = float(np.mean(dist_b2a[:, 0] <= eps_b))

        overlap_sym = 0.5 * (overlap_a + overlap_b)
        overlap_f1 = (
            2.0 * overlap_a * overlap_b / (overlap_a + overlap_b)
            if overlap_a + overlap_b > 0.0
            else 0.0
        )
        return overlap_sym, overlap_f1, overlap_a, overlap_b
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distances with explicit, deterministic handling of zero vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    unit_a = np.divide(a, norm_a[:, None], out=np.zeros_like(a), where=norm_a[:, None] > 0)
    unit_b = np.divide(b, norm_b[:, None], out=np.zeros_like(b), where=norm_b[:, None] > 0)
    distances = 1.0 - unit_a @ unit_b.T
    zero_a = norm_a == 0
    zero_b = norm_b == 0
    distances[np.ix_(zero_a, zero_b)] = 0.0
    distances[np.ix_(zero_a, ~zero_b)] = 1.0
    distances[np.ix_(~zero_a, zero_b)] = 1.0
    return np.clip(distances, 0.0, 2.0)


def overlap_cosine_percentage(
    cloud_a: np.ndarray, cloud_b: np.ndarray
) -> tuple[float, float, float]:
    """Adaptive local overlap using cosine distance, 1 - cosine similarity."""
    cloud_a = np.asarray(cloud_a, dtype=float)
    cloud_b = np.asarray(cloud_b, dtype=float)
    if cloud_a.ndim != 2 or cloud_b.ndim != 2 or len(cloud_a) < 2 or len(cloud_b) < 2:
        return 0.0, 0.0, 0.0
    try:
        d_aa = cosine_distance_matrix(cloud_a, cloud_a)
        d_bb = cosine_distance_matrix(cloud_b, cloud_b)
        np.fill_diagonal(d_aa, np.inf)
        np.fill_diagonal(d_bb, np.inf)
        radius_a = d_aa.min(axis=1)
        radius_b = d_bb.min(axis=1)
        d_ab = cosine_distance_matrix(cloud_a, cloud_b)
        overlap_a = float(np.mean(d_ab.min(axis=1) <= radius_a))
        overlap_b = float(np.mean(d_ab.min(axis=0) <= radius_b))
        return 0.5 * (overlap_a + overlap_b), overlap_a, overlap_b
    except Exception:
        return 0.0, 0.0, 0.0


def score_pair(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    *,
    object_a: object | None = None,
    object_b: object | None = None,
    label: object | None = None,
) -> dict[str, float]:
    pair_wall_started = time.perf_counter()
    pair_cpu_started = time.process_time()
    cloud_a = np.asarray(cloud_a, dtype=float)
    cloud_b = np.asarray(cloud_b, dtype=float)

    pair_metrics = (
        ACTIVE_PAIR_RESUME_PLAN.missing_for(object_a, object_b, label)
        if ACTIVE_PAIR_RESUME_PLAN is not None
        else set(RANKING_SCORE_COLS)
    )
    if ACTIVE_RANKING_SCORE_COLS is not None:
        pair_metrics &= ACTIVE_RANKING_SCORE_COLS

    scores: dict[str, float] = {}
    if {
        "overlap_sym", "overlap_f1", "overlap_euclidean_plus_pooled_euclidean"
    }.intersection(pair_metrics):
        t0 = time.perf_counter()
        overlap_sym, overlap_f1, overlap_a, overlap_b = overlap_percentage_3d(cloud_a, cloud_b)
        elapsed = time.perf_counter() - t0
        if {"overlap_sym", "overlap_euclidean_plus_pooled_euclidean"}.intersection(pair_metrics):
            scores["overlap_sym"] = overlap_sym
        if "overlap_f1" in pair_metrics:
            scores["overlap_f1"] = overlap_f1
        # Keep the directional quantities for diagnosis whenever either
        # aggregation is calculated. They are not independently ranked.
        scores.update(
            {
                "overlap_a_to_b": overlap_a,
                "overlap_b_to_a": overlap_b,
                "time_overlap_sec": elapsed,
            }
        )

    if {"overlap_cosine_sym", "overlap_cosine_plus_pooled_cosine"}.intersection(pair_metrics):
        t0 = time.perf_counter()
        cosine_sym, cosine_a, cosine_b = overlap_cosine_percentage(cloud_a, cloud_b)
        scores.update(
            {
                "overlap_cosine_sym": cosine_sym,
                "overlap_cosine_a_to_b": cosine_a,
                "overlap_cosine_b_to_a": cosine_b,
                "time_overlap_cosine_sec": time.perf_counter() - t0,
            }
        )

    pooled_metrics = {
        metric
        for metric in ["cosine_pooled", "neg_euclidean_pooled"]
        if metric in pair_metrics
    }
    if "overlap_cosine_plus_pooled_cosine" in pair_metrics:
        pooled_metrics.add("cosine_pooled")
    if "overlap_euclidean_plus_pooled_euclidean" in pair_metrics:
        pooled_metrics.add("neg_euclidean_pooled")
    if pooled_metrics:
        t0 = time.perf_counter()
        pooled_a = cloud_a.mean(axis=0, keepdims=True)
        pooled_b = cloud_b.mean(axis=0, keepdims=True)
        scores["time_pooled_sec"] = time.perf_counter() - t0
        if "cosine_pooled" in pooled_metrics:
            t0 = time.perf_counter()
            scores["cosine_pooled"] = float(cosine_similarity(pooled_a, pooled_b)[0, 0])
            scores["time_cosine_sec"] = time.perf_counter() - t0
        if "neg_euclidean_pooled" in pooled_metrics:
            t0 = time.perf_counter()
            scores["neg_euclidean_pooled"] = -float(euclidean_distances(pooled_a, pooled_b)[0, 0])
            scores["time_euclidean_sec"] = time.perf_counter() - t0

    if "overlap_cosine_plus_pooled_cosine" in pair_metrics:
        scores["overlap_cosine_plus_pooled_cosine"] = (
            scores["overlap_cosine_sym"] + scores["cosine_pooled"]
        )
        scores["time_overlap_cosine_plus_pooled_cosine_sec"] = (
            scores.get("time_overlap_cosine_sec", 0.0)
            + scores.get("time_pooled_sec", 0.0)
            + scores.get("time_cosine_sec", 0.0)
        )
    if "overlap_euclidean_plus_pooled_euclidean" in pair_metrics:
        scores["overlap_euclidean_plus_pooled_euclidean"] = (
            scores["overlap_sym"] + scores["neg_euclidean_pooled"]
        )
        scores["time_overlap_euclidean_plus_pooled_euclidean_sec"] = (
            scores.get("time_overlap_sec", 0.0)
            + scores.get("time_pooled_sec", 0.0)
            + scores.get("time_euclidean_sec", 0.0)
        )

    for metric, metric_fn in METRIC_FUNCTIONS.items():
        if metric not in pair_metrics:
            continue
        t0 = time.perf_counter()
        try:
            value = float(metric_fn(cloud_a, cloud_b))
            scores[metric] = value if math.isfinite(value) else math.nan
        except Exception:
            scores[metric] = math.nan
        scores[f"time_{metric}_sec"] = time.perf_counter() - t0
    requested_alg = pair_metrics & ALG_GRID_SCORE_SET
    if requested_alg:
        try:
            cap = int(POINTSET_CONFIG.max_points)
            grid_a = cloud_a if cap <= 0 or len(cloud_a) <= cap else cloud_a[np.linspace(0, len(cloud_a) - 1, cap, dtype=int)]
            grid_b = cloud_b if cap <= 0 or len(cloud_b) <= cap else cloud_b[np.linspace(0, len(cloud_b) - 1, cap, dtype=int)]
            grid_values, diagnostics = alg_grid_scores(grid_a, grid_b)
            scores.update({metric: grid_values[metric] for metric in requested_alg})
            scores.update(diagnostics)
            scores["alg_grid_n_a"] = len(grid_a)
            scores["alg_grid_n_b"] = len(grid_b)
        except Exception:
            scores.update({metric: math.nan for metric in requested_alg})
            scores["time_alg_grid_sec"] = math.nan
    requested_tuned = pair_metrics & TUNED_BASELINE_SCORE_SET
    if requested_tuned:
        try:
            tuned_values, tuned_diagnostics = tuned_baseline_grid_scores(
                cloud_a,
                cloud_b,
                max_points=int(POINTSET_CONFIG.max_points),
            )
            scores.update({metric: tuned_values[metric] for metric in requested_tuned})
            scores.update(tuned_diagnostics)
        except Exception:
            scores.update({metric: math.nan for metric in requested_tuned})
            scores["time_tuned_baseline_grid_sec"] = math.nan
    requested_framework = pair_metrics & ALG_FRAMEWORK_COMPONENT_SCORE_SET
    if requested_framework:
        try:
            cap = int(POINTSET_CONFIG.max_points)
            framework_a = (
                cloud_a
                if cap <= 0 or len(cloud_a) <= cap
                else cloud_a[np.linspace(0, len(cloud_a) - 1, cap, dtype=int)]
            )
            framework_b = (
                cloud_b
                if cap <= 0 or len(cloud_b) <= cap
                else cloud_b[np.linspace(0, len(cloud_b) - 1, cap, dtype=int)]
            )
            local_values, global_values, diagnostics = alg_framework_components(
                framework_a,
                framework_b,
                sliced_projections=int(POINTSET_CONFIG.sliced_projections),
            )
            component_values = {
                **{f"algfw_local__{name}": value for name, value in local_values.items()},
                **{f"algfw_global__{name}": value for name, value in global_values.items()},
            }
            scores.update({metric: component_values[metric] for metric in requested_framework})
            scores.update(diagnostics)
            scores["algfw_component_n_a"] = len(framework_a)
            scores["algfw_component_n_b"] = len(framework_b)
        except Exception:
            scores.update({metric: math.nan for metric in requested_framework})
            scores["time_algfw_components_sec"] = math.nan
    if FROZEN_ALG_RUNTIME_METRIC in pair_metrics:
        try:
            frozen_score, frozen_diagnostics = benchmark_frozen_alg_pair(
                cloud_a,
                cloud_b,
                max_points=int(POINTSET_CONFIG.max_points),
            )
            scores[FROZEN_ALG_RUNTIME_METRIC] = frozen_score
            scores.update(frozen_diagnostics)
        except Exception:
            scores[FROZEN_ALG_RUNTIME_METRIC] = math.nan
            scores["time_alg_frozen_cold_sec"] = math.nan
            scores["time_alg_frozen_cached_sec"] = math.nan
            scores["time_alg_frozen_cache_prepare_sec"] = math.nan
    requested_end_to_end = pair_metrics & {ALG_STAR_METRIC, INITIAL_ALG_METRIC}
    if requested_end_to_end:
        try:
            identity = f"{object_a}|{object_b}|{label}".encode("utf-8")
            reverse_first = bool(hashlib.blake2b(identity, digest_size=1).digest()[0] & 1)
            runtime_scores, runtime_diagnostics = benchmark_alg_star_pair(
                cloud_a,
                cloud_b,
                max_points=int(POINTSET_CONFIG.max_points),
                reverse_first=reverse_first,
            )
            scores.update({metric: runtime_scores[metric] for metric in requested_end_to_end})
            scores.update(runtime_diagnostics)
        except Exception:
            for metric in requested_end_to_end:
                scores[metric] = math.nan
                scores[f"time_{metric}_sec"] = math.nan
                scores[f"cpu_{metric}_sec"] = math.nan
    scores["time_score_pair_total_sec"] = time.perf_counter() - pair_wall_started
    scores["cpu_score_pair_total_sec"] = time.process_time() - pair_cpu_started
    return scores


def parse_n_pairs(value: str) -> int | None:
    value = str(value).strip().lower()
    if value in {"all", "full", "none", "-1"}:
        return None
    n_pairs = int(value)
    if n_pairs <= 0:
        raise ValueError("--n-pairs must be a positive integer or 'all'")
    return n_pairs


def bounded_pair_count(n_pairs: int | None, cap: int) -> int | None:
    return None if n_pairs is None else min(n_pairs, cap)


def maybe_limit(items, limit: int):
    return list(items) if limit <= 0 else list(items)[:limit]


def make_balanced_pairs(
    labels: np.ndarray,
    n_pairs: int | None,
    rng: np.random.Generator,
    min_per_class: int = 2,
) -> list[tuple[int, int, int]]:
    by_label = {
        lab: np.where(labels == lab)[0]
        for lab in np.unique(labels)
        if int(np.sum(labels == lab)) >= min_per_class
    }
    keys = list(by_label)
    pairs: list[tuple[int, int, int]] = []
    if n_pairs is None:
        eligible = np.concatenate([by_label[lab] for lab in keys])
        eligible = np.asarray(sorted(int(i) for i in eligible), dtype=int)
        for pos_a, a in enumerate(eligible):
            for b in eligible[pos_a + 1 :]:
                pairs.append((int(a), int(b), int(labels[a] == labels[b])))
        return pairs

    for i in range(n_pairs):
        same = i < n_pairs // 2
        if same:
            lab = rng.choice(keys)
            a, b = rng.choice(by_label[lab], size=2, replace=False)
            pairs.append((int(a), int(b), 1))
        else:
            lab_a, lab_b = rng.choice(keys, size=2, replace=False)
            a = rng.choice(by_label[lab_a])
            b = rng.choice(by_label[lab_b])
            pairs.append((int(a), int(b), 0))
    return pairs


def make_disjoint_class_clouds(
    x: np.ndarray,
    labels: np.ndarray,
    cloud_size: int,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Partition class rows into fixed, disjoint clouds for Cartesian pairing.

    Rows are shuffled reproducibly within each class, used exactly once, and
    incomplete trailing blocks are dropped.  The returned object identifiers
    are stable for the dataset-specific RNG seed used by the caller.
    """
    if cloud_size < 2:
        raise ValueError("cloud_size must be >= 2")
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels)
    clouds: list[np.ndarray] = []
    cloud_labels: list[object] = []
    object_ids: list[str] = []
    for lab in np.unique(labels):
        indices = np.where(labels == lab)[0].copy()
        rng.shuffle(indices)
        n_clouds = len(indices) // cloud_size
        for cloud_index in range(n_clouds):
            start = cloud_index * cloud_size
            selected = indices[start : start + cloud_size]
            clouds.append(x[selected])
            cloud_labels.append(lab)
            object_ids.append(f"class:{lab}:cloud:{cloud_index}")
    if len(clouds) < 2:
        raise ValueError("need at least two disjoint class clouds")
    return clouds, np.asarray(cloud_labels), object_ids


def global_scale_clouds(clouds: list[np.ndarray]) -> list[np.ndarray]:
    all_points = np.vstack(clouds).astype(float)
    scaler = StandardScaler().fit(all_points)
    return [scaler.transform(c.astype(float)) for c in clouds]


def digit_patch_cloud(img: np.ndarray, patch: int = 2) -> np.ndarray:
    feats = []
    size = img.shape[0]
    for i in range(0, img.shape[0], patch):
        for j in range(0, img.shape[1], patch):
            p = img[i : i + patch, j : j + patch]
            feats.append([p.mean(), p.std(), p.max(), p.min(), i / size, j / size])
    return np.asarray(feats, dtype=float)


def image_patch_cloud(img, image_size: int = 32, patch: int = 4) -> np.ndarray:
    from PIL import Image

    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[0] in {1, 3}:
            arr = np.moveaxis(arr, 0, -1)
        arr = np.asarray(arr)
        if arr.ndim == 2:
            mode = "L"
        else:
            mode = None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255 if arr.max() > 1.5 else 1)
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.5 else arr.astype(np.uint8)
        img = Image.fromarray(arr, mode=mode)
    if not isinstance(img, Image.Image):
        raise TypeError(f"Unsupported image type: {type(img)}")
    arr = np.asarray(img.convert("RGB").resize((image_size, image_size)), dtype=float) / 255.0
    feats = []
    for i in range(0, image_size, patch):
        for j in range(0, image_size, patch):
            p = arr[i : i + patch, j : j + patch, :]
            stats = []
            for channel in range(3):
                pc = p[:, :, channel]
                stats.extend([pc.mean(), pc.std(), pc.max(), pc.min()])
            stats.extend([i / image_size, j / image_size])
            feats.append(stats)
    return np.asarray(feats, dtype=float)


def make_torchvision_dataset(name: str, root: Path):
    from torchvision import datasets
    from torchvision.datasets.utils import download_and_extract_archive, extract_archive

    if name == "MNIST":
        return datasets.MNIST(root=str(root), train=True, download=True)
    if name == "FashionMNIST":
        return datasets.FashionMNIST(root=str(root), train=True, download=True)
    if name == "KMNIST":
        return datasets.KMNIST(root=str(root), train=True, download=True)
    if name == "EMNIST-Balanced":
        # Older torchvision releases still use the retired itl.nist.gov URL,
        # which now redirects to an HTML landing page.  The binary archive is
        # still published by NIST at this stable endpoint with the same MD5.
        datasets.EMNIST.url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
        return datasets.EMNIST(root=str(root), split="balanced", train=True, download=True)
    if name == "QMNIST":
        return datasets.QMNIST(root=str(root), what="train", download=True)
    if name == "USPS":
        return datasets.USPS(root=str(root), train=True, download=True)
    if name == "CIFAR5":
        # Reproducible five-class benchmark: the first five official CIFAR-10
        # classes (airplane, automobile, bird, cat, deer).
        from torch.utils.data import Subset

        dataset = datasets.CIFAR10(root=str(root), train=True, download=True)
        keep = {0, 1, 2, 3, 4}
        indices = [i for i, target in enumerate(dataset.targets) if int(target) in keep]
        return Subset(dataset, indices)
    if name == "CIFAR10":
        return datasets.CIFAR10(root=str(root), train=True, download=True)
    if name == "CIFAR100":
        return datasets.CIFAR100(root=str(root), train=True, download=True)
    if name == "SVHN":
        return datasets.SVHN(root=str(root), split="train", download=True)
    if name == "STL10":
        return datasets.STL10(root=str(root), split="train", download=True)
    if name == "Caltech101":
        # torchvision versions on long-lived clusters use obsolete Caltech
        # URLs.  CaltechDATA is the current first-party source.  Its ZIP wraps
        # the original image tarball, so extract both archive layers.
        dataset_root = root / "caltech101"
        image_root = dataset_root / "101_ObjectCategories"
        if not image_root.exists():
            dataset_root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(
                "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1",
                download_root=str(dataset_root),
                filename="caltech-101.zip",
                md5="3138e1922a9193bfa496528edbbc45d0",
            )
            nested_archives = list(dataset_root.rglob("101_ObjectCategories.tar.gz"))
            if not image_root.exists() and nested_archives:
                extract_archive(str(nested_archives[0]), str(dataset_root))
        return datasets.Caltech101(root=str(root), target_type="category", download=False)
    if name == "Caltech256":
        dataset_root = root / "caltech256"
        image_root = dataset_root / "256_ObjectCategories"
        if not image_root.exists():
            dataset_root.mkdir(parents=True, exist_ok=True)
            download_and_extract_archive(
                "https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1",
                download_root=str(dataset_root),
                filename="256_ObjectCategories.tar",
                md5="67b4f42ca05d46448c6bb8ecd2220f6d",
            )
        return datasets.Caltech256(root=str(root), download=False)
    if name == "Flowers102":
        return datasets.Flowers102(root=str(root), split="train", download=True)
    if name == "OxfordIIITPet":
        return datasets.OxfordIIITPet(root=str(root), split="trainval", target_types="category", download=True)
    if name == "Food101":
        return datasets.Food101(root=str(root), split="train", download=True)
    if name == "CelebA-Identity":
        return datasets.CelebA(
            root=str(root),
            split="train",
            target_type="identity",
            download=True,
        )
    if name == "ImageNet":
        image_root = root / "imagenet" / "train"
        if not image_root.exists():
            raise FileNotFoundError(
                f"ImageNet is license-gated. Place class folders under {image_root}"
            )
        return datasets.ImageFolder(str(image_root))
    if name == "LSUN-Multi":
        lsun_root = root / "lsun"
        if not lsun_root.exists():
            raise FileNotFoundError(
                f"LSUN must be downloaded manually under {lsun_root}"
            )
        return datasets.LSUN(
            root=str(lsun_root),
            classes=["bedroom_train", "church_outdoor_train", "tower_train"],
        )
    raise ValueError(f"Unsupported TorchVision dataset: {name}")


def run_torchvision_images(
    name: str,
    n_pairs: int | None,
    rng: np.random.Generator,
    max_items: int,
    root: Path,
) -> DatasetResult:
    ds = make_torchvision_dataset(name, root)
    indices = np.arange(len(ds))
    if max_items and len(indices) > max_items:
        indices = rng.choice(indices, size=max_items, replace=False)
    labels = []
    clouds = []
    object_ids = []
    for idx in indices:
        img, label = ds[int(idx)]
        if hasattr(label, "item"):
            label = label.item()
        if isinstance(label, (tuple, list)):
            label = label[0]
        labels.append(str(label))
        clouds.append(image_patch_cloud(img))
        object_ids.append(int(idx))
    labels = np.asarray(labels)
    clouds = global_scale_clouds(clouds)
    rows = []
    for a, b, label in make_balanced_pairs(labels, bounded_pair_count(n_pairs, len(labels) * 2), rng):
        rows.append(
            {
                "support": "images",
                "dataset": f"TorchVision_{name}",
                "object_a": object_ids[a],
                "object_b": object_ids[b],
                "class_a": str(labels[a]),
                "class_b": str(labels[b]),
                "label": label,
                **score_pair(clouds[a], clouds[b], object_a=object_ids[a], object_b=object_ids[b], label=label),
            }
        )
    return DatasetResult("images", f"TorchVision_{name}", pd.DataFrame(rows))


def run_digits_images(n_pairs: int | None, rng: np.random.Generator) -> DatasetResult:
    data = load_digits()
    images = data.images.astype(float) / 16.0
    labels = data.target.astype(int)
    clouds = global_scale_clouds([digit_patch_cloud(img) for img in images])
    rows = []
    for a, b, label in make_balanced_pairs(labels, n_pairs, rng):
        rows.append(
            {
                "support": "images",
                "dataset": "sklearn_digits_patch_clouds",
                "object_a": a,
                "object_b": b,
                "class_a": int(labels[a]),
                "class_b": int(labels[b]),
                "label": label,
                **score_pair(clouds[a], clouds[b], object_a=a, object_b=b, label=label),
            }
        )
    return DatasetResult("images", "sklearn_digits_patch_clouds", pd.DataFrame(rows))


def run_tabular_class_cloud(
    loader: Callable,
    dataset_name: str,
    n_pairs: int | None,
    cloud_size: int,
    rng: np.random.Generator,
) -> DatasetResult:
    data = loader()
    x = StandardScaler().fit_transform(np.asarray(data.data, dtype=float))
    labels = np.asarray(data.target)
    return run_array_class_cloud(
        x, labels, "tabular", dataset_name, n_pairs, cloud_size, rng
    )


def sliding_window_cloud(series: np.ndarray, window: int = 24, stride: int = 4) -> np.ndarray:
    series = np.asarray(series, dtype=float).ravel()
    if len(series) < window:
        return series.reshape(-1, 1)
    windows = [series[i : i + window] for i in range(0, len(series) - window + 1, stride)]
    return np.asarray(windows, dtype=float)


def _parse_ts_values(text: str) -> np.ndarray:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item or item == "?":
            values.append(np.nan)
        else:
            values.append(float(item))
    arr = np.asarray(values, dtype=float)
    if np.isnan(arr).any():
        finite = arr[np.isfinite(arr)]
        fill = float(np.nanmean(finite)) if len(finite) else 0.0
        arr = np.nan_to_num(arr, nan=fill)
    return arr


def read_ucr_ts(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    rows: list[np.ndarray] = []
    labels = []
    in_data = False
    for raw_line in path.read_text(errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data or line.startswith("@"):
            continue
        parts = line.split(":")
        if len(parts) < 2:
            continue
        label = parts[-1].strip()
        dim_parts = parts[:-1]
        dims = [_parse_ts_values(part) for part in dim_parts if part.strip()]
        if not dims:
            continue
        # UCR is mostly univariate. For rare multivariate .ts files, concatenate
        # dimensions so the downstream sliding-window protocol remains uniform.
        rows.append(np.concatenate(dims))
        labels.append(label)
    if not rows:
        raise ValueError(f"No @data rows parsed from {path}")
    return rows, np.asarray(labels)


def read_ucr_file(path: Path) -> tuple[list[np.ndarray], np.ndarray]:
    if path.suffix.lower() == ".ts":
        return read_ucr_ts(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", header=None)
    elif suffix == ".csv":
        df = pd.read_csv(path, sep=",", header=None)
    else:
        # UCR .txt files are whitespace-delimited. Avoid the former overlapping
        # regex ``\s+|\t|,``, which made pandas infer inconsistent row widths.
        df = pd.read_csv(path, sep=r"\s+", header=None)
    labels = df.iloc[:, 0].to_numpy()
    x = df.iloc[:, 1:].to_numpy(dtype=float)
    return [row for row in x], labels


def run_ucr_dataset(
    dataset_dir: Path,
    n_pairs: int | None,
    rng: np.random.Generator,
    window: int,
    stride: int,
    prefix: str = "UCR",
) -> DatasetResult:
    train_files = (
        list(dataset_dir.glob("*_TRAIN.tsv"))
        + list(dataset_dir.glob("*_TRAIN.txt"))
        + list(dataset_dir.glob("*_TRAIN.csv"))
        + list(dataset_dir.glob("*_TRAIN.ts"))
    )
    test_files = (
        list(dataset_dir.glob("*_TEST.tsv"))
        + list(dataset_dir.glob("*_TEST.txt"))
        + list(dataset_dir.glob("*_TEST.csv"))
        + list(dataset_dir.glob("*_TEST.ts"))
    )
    if not train_files or not test_files:
        raise FileNotFoundError(f"Missing UCR TRAIN/TEST files in {dataset_dir}")
    x_train, y_train = read_ucr_file(train_files[0])
    x_test, y_test = read_ucr_file(test_files[0])
    series = list(x_train) + list(x_test)
    labels = np.concatenate([y_train, y_test])
    finite_values = np.concatenate([s[np.isfinite(s)] for s in series if len(s)])
    mean = float(np.mean(finite_values)) if len(finite_values) else 0.0
    std = float(np.std(finite_values)) if len(finite_values) else 1.0
    std = std if std > 1e-12 else 1.0
    series = [(s - mean) / std for s in series]
    local_window = min(window, min(len(s) for s in series if len(s) > 1))
    local_window = max(2, local_window)
    rows = []
    pairs = make_balanced_pairs(labels, bounded_pair_count(n_pairs, len(labels) * 2), rng)
    used_indices = sorted({idx for a, b, _ in pairs for idx in (a, b)})
    # UCR archives can be large. Build only the sampled clouds and avoid a
    # second global scaling pass; each dataset is already standardized above.
    clouds = {
        idx: sliding_window_cloud(series[idx], local_window, stride)
        for idx in used_indices
    }
    for a, b, label in pairs:
        rows.append(
            {
                "support": "time_series",
                "dataset": f"{prefix}_{dataset_dir.name}",
                "object_a": a,
                "object_b": b,
                "class_a": str(labels[a]),
                "class_b": str(labels[b]),
                "label": label,
                **score_pair(clouds[a], clouds[b], object_a=a, object_b=b, label=label),
            }
        )
    return DatasetResult("time_series", f"{prefix}_{dataset_dir.name}", pd.DataFrame(rows))


def encode_openml_features(raw_data) -> tuple[np.ndarray, int, int]:
    """Encode an OpenML feature table without inventing ordinal distances.

    Numeric columns are median-imputed and standardized. Nominal, categorical,
    object, and string columns are one-hot encoded and kept in {0, 1}.  This is
    the usual mixed-tabular geometry and avoids treating arbitrary labels such
    as ``Male``/``Female`` as ordered real numbers.
    """
    frame = raw_data.copy() if isinstance(raw_data, pd.DataFrame) else pd.DataFrame(raw_data)
    numeric_columns = []
    categorical_columns = []
    for column in frame.columns:
        series = frame[column]
        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    parts: list[np.ndarray] = []
    if numeric_columns:
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        for column in numeric.columns:
            finite = numeric[column].dropna()
            fill_value = float(finite.median()) if len(finite) else 0.0
            numeric[column] = numeric[column].fillna(fill_value)
        parts.append(StandardScaler().fit_transform(numeric.to_numpy(dtype=float)))

    if categorical_columns:
        categorical = frame[categorical_columns].copy()
        for column in categorical.columns:
            categorical[column] = (
                categorical[column]
                .astype(object)
                .where(categorical[column].notna(), "__missing__")
                .astype(str)
            )
        one_hot = pd.get_dummies(
            categorical,
            prefix=[str(column) for column in categorical.columns],
            prefix_sep="=",
        )
        parts.append(one_hot.to_numpy(dtype=float))

    if not parts:
        raise ValueError("OpenML dataset has no usable feature column")
    x = np.concatenate(parts, axis=1) if len(parts) > 1 else parts[0]
    return x, len(numeric_columns), len(categorical_columns)


def fetch_openml_with_retry(openml_id: int):
    """Fetch an OpenML dataset, retrying transient/corrupted-cache failures."""
    from sklearn.datasets import fetch_openml

    data_home = os.environ.get("OPENML_CACHE_DIR") or None
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            return fetch_openml(
                data_id=openml_id,
                as_frame=True,
                parser="auto",
                data_home=data_home,
                cache=attempt == 0,
            )
        except (OSError, ValueError) as exc:
            last_error = exc
            if attempt < 2:
                print(
                    f"OpenML {openml_id}: download/cache attempt {attempt + 1} failed; retrying without cache"
                )
    assert last_error is not None
    # sklearn validates against the checksum stored in its local metadata. If
    # that metadata/cache remains inconsistent, use the official OpenML client
    # as an independent transport while preserving a pandas feature table.
    try:
        import openml
        from types import SimpleNamespace

        dataset = openml.datasets.get_dataset(openml_id, download_data=True)
        x, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=dataset.default_target_attribute,
        )
        return SimpleNamespace(data=x, target=y)
    except Exception as fallback_error:
        raise RuntimeError(
            f"OpenML {openml_id} failed via sklearn ({last_error}) and openml ({fallback_error})"
        ) from fallback_error


def run_openml_dataset(
    openml_id: int,
    name: str,
    n_pairs: int | None,
    cloud_size: int,
    rng: np.random.Generator,
    max_rows: int = 0,
) -> DatasetResult:
    data = fetch_openml_with_retry(openml_id)
    frame = data.data.copy() if isinstance(data.data, pd.DataFrame) else pd.DataFrame(data.data)
    labels = pd.Series(data.target, index=frame.index)
    target_mask = labels.notna().to_numpy()
    frame = frame.loc[target_mask].reset_index(drop=True)
    labels = labels.loc[target_mask].reset_index(drop=True)
    if max_rows and len(labels) > max_rows:
        idx = rng.choice(np.arange(len(labels)), size=max_rows, replace=False)
        frame = frame.iloc[idx].reset_index(drop=True)
        labels = labels.iloc[idx].reset_index(drop=True)
    x, n_numeric, n_categorical = encode_openml_features(frame)
    print(
        f"{name}: encoded {n_numeric} numeric and {n_categorical} categorical columns "
        f"into {x.shape[1]} features"
    )
    return run_array_class_cloud(
        x,
        labels.to_numpy(),
        "openml",
        name,
        n_pairs,
        cloud_size,
        rng,
    )


def resolve_openml_suite_specs(
    suite_text: str,
    limit: int,
) -> list[tuple[int, str]]:
    import openml

    suite_id: int | str
    suite_id = int(suite_text) if suite_text.isdigit() else suite_text
    suite = openml.study.get_suite(suite_id)
    data_ids = list(getattr(suite, "data", []) or [])
    if not data_ids:
        task_ids = list(getattr(suite, "tasks", []) or [])
        for task_id in task_ids:
            task = openml.tasks.get_task(task_id)
            data_ids.append(int(task.dataset_id))
    seen = set()
    specs = []
    for data_id in data_ids:
        data_id = int(data_id)
        if data_id in seen:
            continue
        seen.add(data_id)
        specs.append((data_id, f"OpenML_suite_{suite_text}_{data_id}"))
        if limit and len(specs) >= limit:
            break
    return specs


def run_array_class_cloud(
    x: np.ndarray,
    labels: np.ndarray,
    support: str,
    dataset_name: str,
    n_pairs: int | None,
    cloud_size: int,
    rng: np.random.Generator,
) -> DatasetResult:
    by_label = {
        lab: np.where(labels == lab)[0]
        for lab in np.unique(labels)
        if int(np.sum(labels == lab)) >= cloud_size
    }
    keys = list(by_label)
    if len(keys) < 2:
        raise ValueError(f"{dataset_name}: need at least two classes")
    rows = []
    if n_pairs is None:
        clouds, fixed_labels, object_ids = make_disjoint_class_clouds(
            x, labels, cloud_size, rng
        )
        # In Cartesian mode a class represented by a single fixed cloud can
        # still contribute valid negative pairs, so retain singleton classes.
        for a, b, label in make_balanced_pairs(
            fixed_labels, None, rng, min_per_class=1
        ):
            class_a, class_b = str(fixed_labels[a]), str(fixed_labels[b])
            object_a, object_b = object_ids[a], object_ids[b]
            rows.append(
                {
                    "support": support,
                    "dataset": dataset_name,
                    "object_a": object_a,
                    "object_b": object_b,
                    "class_a": class_a,
                    "class_b": class_b,
                    "label": label,
                    **score_pair(
                        clouds[a],
                        clouds[b],
                        object_a=object_a,
                        object_b=object_b,
                        label=label,
                    ),
                }
            )
        return DatasetResult(support, dataset_name, pd.DataFrame(rows))

    for i in range(n_pairs):
        same = i < n_pairs // 2
        if same:
            lab = rng.choice(keys)
            ia = rng.choice(by_label[lab], size=cloud_size, replace=True)
            ib = rng.choice(by_label[lab], size=cloud_size, replace=True)
            label = 1
            class_a = class_b = str(lab)
        else:
            lab_a, lab_b = rng.choice(keys, size=2, replace=False)
            ia = rng.choice(by_label[lab_a], size=cloud_size, replace=True)
            ib = rng.choice(by_label[lab_b], size=cloud_size, replace=True)
            label = 0
            class_a, class_b = str(lab_a), str(lab_b)
        rows.append(
            {
                "support": support,
                "dataset": dataset_name,
                "object_a": f"{class_a}:{i}:a",
                "object_b": f"{class_b}:{i}:b",
                "class_a": class_a,
                "class_b": class_b,
                "label": label,
                **score_pair(x[ia], x[ib], object_a=f"{class_a}:{i}:a", object_b=f"{class_b}:{i}:b", label=label),
            }
        )
    return DatasetResult(support, dataset_name, pd.DataFrame(rows))


def run_text_glue(
    task: str,
    n_pairs: int | None,
    rng: np.random.Generator,
    model_name: str,
) -> DatasetResult:
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    split = "validation_matched" if task == "mnli" else "validation"
    ds = load_dataset("glue", task, split=split)
    rows = []
    limit = len(ds) if n_pairs is None else min(n_pairs * 3, len(ds))
    for row in ds.select(range(limit)):
        if task == "stsb":
            raw_label = float(row["label"])
            if raw_label >= 4.0:
                label = 1
            elif raw_label <= 2.0:
                label = 0
            else:
                continue
            text_a, text_b = row["sentence1"], row["sentence2"]
        elif task == "mrpc":
            label = int(row["label"])
            text_a, text_b = row["sentence1"], row["sentence2"]
        elif task == "qqp":
            label = int(row["label"])
            text_a, text_b = row["question1"], row["question2"]
        elif task == "mnli":
            raw_label = int(row["label"])
            if raw_label == 0:
                label = 1
            elif raw_label == 2:
                label = 0
            else:
                continue
            text_a, text_b = row["premise"], row["hypothesis"]
        elif task == "qnli":
            label = 1 if int(row["label"]) == 0 else 0
            text_a, text_b = row["question"], row["sentence"]
        elif task == "rte":
            label = 1 if int(row["label"]) == 0 else 0
            text_a, text_b = row["sentence1"], row["sentence2"]
        elif task == "wnli":
            label = int(row["label"])
            text_a, text_b = row["sentence1"], row["sentence2"]
        else:
            raise ValueError(f"Unsupported GLUE task: {task}")
        emb = model.encode(
            [text_a, text_b],
            output_value="token_embeddings",
            convert_to_numpy=False,
            device="cpu",
            show_progress_bar=False,
        )
        emb = [
            x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)
            for x in emb
        ]
        rows.append(
            {
                "support": "text",
                "dataset": f"GLUE_{task}",
                "object_a": text_a[:80],
                "object_b": text_b[:80],
                "class_a": label,
                "class_b": label,
                "label": label,
                **score_pair(np.asarray(emb[0]), np.asarray(emb[1]), object_a=text_a[:80], object_b=text_b[:80], label=label),
            }
        )
        if n_pairs is not None and len(rows) >= n_pairs:
            break
    return DatasetResult("text", f"GLUE_{task}", pd.DataFrame(rows))


def graph_to_node_cloud(data, dim: int = 8) -> np.ndarray:
    if getattr(data, "x", None) is not None and data.x is not None:
        return data.x.detach().cpu().numpy().astype(float)
    n = int(data.num_nodes)
    adjacency = np.zeros((n, n), dtype=float)
    edges = data.edge_index.detach().cpu().numpy().T
    for i, j in edges:
        adjacency[int(i), int(j)] = 1.0
    vals, vecs = np.linalg.eigh(adjacency)
    idx = np.argsort(np.abs(vals))[::-1][: min(dim, n)]
    out = np.zeros((n, dim), dtype=float)
    if len(idx):
        out[:, : len(idx)] = vecs[:, idx] * vals[idx]
    return out


def run_tudataset(
    name: str,
    n_pairs: int | None,
    rng: np.random.Generator,
    max_items: int = 0,
) -> DatasetResult:
    # PyG versions used on some CPU cluster images call the newer public API
    # even when the installed PyTorch only exposes torch._dynamo.is_compiling.
    # Data loading does not compile graphs, so forwarding the equivalent query
    # (or returning False on still older PyTorch) is behavior-preserving.
    import torch

    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        class _CompilerCompat:
            @staticmethod
            def is_compiling() -> bool:
                return False

        torch.compiler = _CompilerCompat()
    elif not hasattr(compiler, "is_compiling"):
        dynamo_is_compiling = getattr(getattr(torch, "_dynamo", None), "is_compiling", None)
        compiler.is_compiling = dynamo_is_compiling or (lambda: False)

    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root="/tmp/TUDataset", name=name)
    indices = np.arange(len(ds))
    if max_items and len(indices) > max_items:
        indices = rng.choice(indices, size=max_items, replace=False)
    labels = np.array([int(ds[int(i)].y.item()) for i in indices])
    clouds = global_scale_clouds([graph_to_node_cloud(ds[int(i)]) for i in indices])
    rows = []
    for a, b, label in make_balanced_pairs(labels, bounded_pair_count(n_pairs, len(labels) * 2), rng):
        rows.append(
            {
                "support": "graphs",
                "dataset": f"TUDataset_{name}",
                "object_a": int(indices[a]),
                "object_b": int(indices[b]),
                "class_a": int(labels[a]),
                "class_b": int(labels[b]),
                "label": label,
                **score_pair(clouds[a], clouds[b], object_a=int(indices[a]), object_b=int(indices[b]), label=label),
            }
        )
    return DatasetResult("graphs", f"TUDataset_{name}", pd.DataFrame(rows))


def run_neural_mlp_models(
    n_pairs: int | None,
    rng: np.random.Generator,
    seeds: Iterable[int],
    hidden_sizes: Iterable[tuple[int, ...]],
) -> DatasetResult:
    def activation_profile_cloud(raw_activations: np.ndarray) -> np.ndarray:
        """Fixed-size activation profile, comparable across hidden dimensions."""
        a = np.asarray(raw_activations, dtype=float)
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

    data = load_digits()
    x = StandardScaler().fit_transform(data.data.astype(float) / 16.0)
    y = data.target.astype(int)
    probe = x[:500]
    model_rows = []
    for hidden in hidden_sizes:
        for seed in seeds:
            clf = MLPClassifier(
                hidden_layer_sizes=hidden,
                max_iter=80,
                random_state=int(seed),
                early_stopping=True,
            )
            clf.fit(x, y)
            activ = np.maximum(0.0, probe @ clf.coefs_[0] + clf.intercepts_[0])
            model_rows.append(
                {
                    "architecture": "x".join(map(str, hidden)),
                    "seed": int(seed),
                    "cloud": activation_profile_cloud(activ),
                }
            )
    labels = np.array([m["architecture"] for m in model_rows])
    clouds = global_scale_clouds([m["cloud"] for m in model_rows])
    rows = []
    for a, b, label in make_balanced_pairs(labels, bounded_pair_count(n_pairs, len(labels) * 4), rng):
        rows.append(
            {
                "support": "neural_models",
                "dataset": "digits_mlp_activation_clouds",
                "object_a": f"{labels[a]}:{model_rows[a]['seed']}",
                "object_b": f"{labels[b]}:{model_rows[b]['seed']}",
                "class_a": labels[a],
                "class_b": labels[b],
                "label": label,
                **score_pair(
                    clouds[a],
                    clouds[b],
                    object_a=f"{labels[a]}:{model_rows[a]['seed']}",
                    object_b=f"{labels[b]}:{model_rows[b]['seed']}",
                    label=label,
                ),
            }
        )
    return DatasetResult("neural_models", "digits_mlp_activation_clouds", pd.DataFrame(rows))


def threshold_from_train(y_train: np.ndarray, s_train: np.ndarray) -> float:
    y = np.asarray(y_train, dtype=int)
    scores = np.asarray(s_train, dtype=float)
    valid = np.isfinite(scores)
    y = y[valid]
    scores = scores[valid]
    if len(scores) == 0:
        return 0.0

    values = np.unique(scores)
    if len(values) > 300:
        values = np.quantile(values, np.linspace(0, 1, 300))
        values = np.unique(values)
    # With the downstream strict rule ``score > threshold``, an additional
    # threshold just below the minimum is required to represent the valid
    # classifier that predicts every training pair as positive.
    values = np.concatenate(([np.nextafter(values[0], -np.inf)], values))

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    cum_pos = np.cumsum(sorted_y)
    total_pos = int(cum_pos[-1]) if len(cum_pos) else 0
    if total_pos == 0:
        return float(values[0])

    cut = np.searchsorted(sorted_scores, values, side="right")
    pos_le_threshold = np.where(cut > 0, cum_pos[cut - 1], 0)
    tp = total_pos - pos_le_threshold
    pred_pos = len(scores) - cut
    fp = pred_pos - tp
    fn = total_pos - tp
    denom = (2 * tp + fp + fn).astype(float)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(values[int(np.argmax(f1))])


def _classification_counts(y_true: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=int)
    return (
        int(((p == 1) & (y == 1)).sum()),
        int(((p == 1) & (y == 0)).sum()),
        int(((p == 0) & (y == 0)).sum()),
        int(((p == 0) & (y == 1)).sum()),
    )


def evaluate_dataset_pairs_cv(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None]:
    """Five-fold out-of-fold evaluation of scalar metric thresholds.

    There is deliberately no learned classifier. For each metric and fold, a
    one-dimensional threshold is selected on the other folds and applied once
    to the held-out fold.
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), "empty pair table"
    labels = df["label"].astype(int).to_numpy()
    values, counts = np.unique(labels, return_counts=True)
    if len(values) != 2:
        return pd.DataFrame(), pd.DataFrame(), "requires exactly two pair labels"
    if int(counts.min()) < n_splits:
        return pd.DataFrame(), pd.DataFrame(), (
            f"requires at least {n_splits} pairs in each class; minimum is {int(counts.min())}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    summary_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    support = str(df["support"].iloc[0])
    dataset = str(df["dataset"].iloc[0])

    for metric in RANKING_SCORE_COLS:
        if metric not in df.columns:
            continue
        metric_cv_started = time.perf_counter()
        metric_cv_cpu_started = time.process_time()
        raw_scores = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
        oriented = oriented_scores(metric, raw_scores)
        if not np.isfinite(oriented).all():
            continue
        oof_prediction = np.zeros(len(df), dtype=int)
        thresholds: list[float] = []
        metric_fold_rows: list[dict[str, object]] = []
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            threshold_started = time.perf_counter()
            threshold_cpu_started = time.process_time()
            threshold = threshold_from_train(labels[train_idx], oriented[train_idx])
            threshold_wall_sec = time.perf_counter() - threshold_started
            threshold_cpu_sec = time.process_time() - threshold_cpu_started
            scoring_started = time.perf_counter()
            scoring_cpu_started = time.process_time()
            prediction = (oriented[test_idx] > threshold).astype(int)
            oof_prediction[test_idx] = prediction
            threshold_reported = -threshold if metric in LOWER_IS_MORE_SIMILAR else threshold
            thresholds.append(float(threshold_reported))
            y_test = labels[test_idx]
            test_scores = oriented[test_idx]
            tp, fp, tn, fn = _classification_counts(y_test, prediction)
            try:
                fold_auc = float(roc_auc_score(y_test, test_scores))
            except ValueError:
                fold_auc = math.nan
            try:
                fold_ap = float(average_precision_score(y_test, test_scores))
            except ValueError:
                fold_ap = math.nan
            metric_fold_rows.append(
                {
                    "support": support,
                    "dataset": dataset,
                    "metric": metric,
                    "fold": fold_id,
                    "threshold_train": threshold_reported,
                    "classifier": "scalar_threshold",
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "n_train_positive": int(labels[train_idx].sum()),
                    "n_train_negative": int(len(train_idx) - labels[train_idx].sum()),
                    "n_test_positive": int(y_test.sum()),
                    "n_test_negative": int(len(y_test) - y_test.sum()),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "f1": f1_score(y_test, prediction, zero_division=0),
                    "precision": precision_score(y_test, prediction, zero_division=0),
                    "recall": recall_score(y_test, prediction, zero_division=0),
                    "specificity": tn / (tn + fp) if (tn + fp) else math.nan,
                    "accuracy": (tp + tn) / len(y_test),
                    "roc_auc": fold_auc,
                    "average_precision": fold_ap,
                    "time_threshold_fit_sec": threshold_wall_sec,
                    "cpu_threshold_fit_sec": threshold_cpu_sec,
                    "time_fold_scoring_sec": time.perf_counter() - scoring_started,
                    "cpu_fold_scoring_sec": time.process_time() - scoring_cpu_started,
                }
            )
        fold_rows.extend(metric_fold_rows)
        tp, fp, tn, fn = _classification_counts(labels, oof_prediction)
        try:
            auc = float(roc_auc_score(labels, oriented))
        except ValueError:
            auc = math.nan
        try:
            avg_precision = float(average_precision_score(labels, oriented))
        except ValueError:
            avg_precision = math.nan
        fold_f1 = np.asarray([float(row["f1"]) for row in metric_fold_rows])
        summary_rows.append(
            {
                "support": support,
                "dataset": dataset,
                "metric": metric,
                "classifier": "scalar_threshold",
                "cv_folds": n_splits,
                "threshold_train": float(np.mean(thresholds)),
                "threshold_train_std": float(np.std(thresholds, ddof=1)),
                "threshold_train_min": float(np.min(thresholds)),
                "threshold_train_max": float(np.max(thresholds)),
                "roc_auc": auc,
                "average_precision": avg_precision,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "f1": f1_score(labels, oof_prediction, zero_division=0),
                "mean_fold_f1": float(fold_f1.mean()),
                "std_fold_f1": float(fold_f1.std(ddof=1)),
                "precision": precision_score(labels, oof_prediction, zero_division=0),
                "recall": recall_score(labels, oof_prediction, zero_division=0),
                "specificity": tn / (tn + fp) if (tn + fp) else math.nan,
                "accuracy": (tp + tn) / len(labels),
                "n_pairs": len(df),
                "n_test": len(df),
                "n_test_positive": int(labels.sum()),
                "n_test_negative": int(len(labels) - labels.sum()),
                "time_cv_evaluation_total_sec": time.perf_counter() - metric_cv_started,
                "cpu_cv_evaluation_total_sec": time.process_time() - metric_cv_cpu_started,
                "time_threshold_fit_total_sec": float(
                    sum(float(row["time_threshold_fit_sec"]) for row in metric_fold_rows)
                ),
                "cpu_threshold_fit_total_sec": float(
                    sum(float(row["cpu_threshold_fit_sec"]) for row in metric_fold_rows)
                ),
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), None


def build_global_test_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    required = {"tp", "fp", "tn", "fn", "n_test"}
    if not required.issubset(summary.columns):
        return pd.DataFrame()
    rows = []
    for metric, g in summary.groupby("metric", sort=False):
        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        tn = int(g["tn"].sum())
        fn = int(g["fn"].sum())
        n_test = int(g["n_test"].sum())
        precision = tp / (tp + fp) if (tp + fp) else math.nan
        recall = tp / (tp + fn) if (tp + fn) else math.nan
        specificity = tn / (tn + fp) if (tn + fp) else math.nan
        accuracy = (tp + tn) / n_test if n_test else math.nan
        f1_denom = 2 * tp + fp + fn
        f1 = 2 * tp / f1_denom if f1_denom else 0.0
        rows.append(
            {
                "metric": metric,
                "n_datasets": int(g["dataset"].nunique()),
                "n_test": n_test,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "accuracy": accuracy,
                "f1": f1,
                "macro_f1_by_dataset": float(g["f1"].mean()),
                "weighted_f1_by_n_test": float(np.average(g["f1"], weights=g["n_test"])),
            }
        )
    return pd.DataFrame(rows).sort_values("f1", ascending=False)


def rank_metrics(summary: pd.DataFrame, score_col: str = "f1") -> pd.DataFrame:
    parts = []
    for (_, dataset), g in summary.groupby(["support", "dataset"], sort=False):
        tmp = g.copy()
        tmp["rank"] = tmp[score_col].rank(ascending=False, method="min")
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_metric_efficiency(pair_scores: pd.DataFrame) -> pd.DataFrame:
    timing_cols = [
        "time_overlap_sec",
        "time_overlap_cosine_sec",
        "time_overlap_cosine_plus_pooled_cosine_sec",
        "time_overlap_euclidean_plus_pooled_euclidean_sec",
        "time_pooled_sec",
        "time_cosine_sec",
        "time_euclidean_sec",
        *[f"time_{metric}_sec" for metric in METRIC_FUNCTIONS],
        "time_alg_grid_sec",
        "time_tuned_baseline_grid_sec",
        "time_alg_frozen_cold_sec",
        "time_alg_frozen_cached_sec",
        "time_alg_frozen_cache_prepare_sec",
        "time_score_pair_total_sec",
        *[f"time_alg_{geometry}_family_sec" for geometry in ("euclidean", "manhattan", "chebyshev", "cosine")],
        "cpu_score_pair_total_sec",
    ]
    for col in timing_cols:
        if col not in pair_scores.columns:
            pair_scores[col] = np.nan
    rows = []
    for (support, dataset), g in pair_scores.groupby(["support", "dataset"], sort=False):
        n_pairs = len(g)
        metric_time_series = {
            "ALL_metric_suite_total": g["time_score_pair_total_sec"],
            "overlap_sym": g["time_overlap_sec"],
            "overlap_f1": g["time_overlap_sec"],
            "overlap_cosine_sym": g["time_overlap_cosine_sec"],
            "overlap_cosine_plus_pooled_cosine": g["time_overlap_cosine_plus_pooled_cosine_sec"],
            "overlap_euclidean_plus_pooled_euclidean": g["time_overlap_euclidean_plus_pooled_euclidean_sec"],
            "cosine_pooled": g["time_pooled_sec"] + g["time_cosine_sec"],
            "neg_euclidean_pooled": g["time_pooled_sec"] + g["time_euclidean_sec"],
            **{
                metric: g[f"time_{metric}_sec"]
                for metric in METRIC_FUNCTIONS
            },
        }
        if any(metric in pair_scores.columns for metric in ALG_GRID_SCORE_COLS):
            metric_time_series["ALG_grid_all_1320"] = g["time_alg_grid_sec"]
            for geometry in ("euclidean", "manhattan", "chebyshev", "cosine"):
                metric_time_series[f"ALG_{geometry}_family_330"] = g[f"time_alg_{geometry}_family_sec"]
        if FROZEN_ALG_RUNTIME_METRIC in pair_scores.columns:
            metric_time_series[f"{FROZEN_ALG_RUNTIME_METRIC}__cold"] = g["time_alg_frozen_cold_sec"]
            metric_time_series[f"{FROZEN_ALG_RUNTIME_METRIC}__cached"] = g["time_alg_frozen_cached_sec"]
            metric_time_series[f"{FROZEN_ALG_RUNTIME_METRIC}__cache_prepare"] = g[
                "time_alg_frozen_cache_prepare_sec"
            ]
        for metric, timing in metric_time_series.items():
            valid = pd.to_numeric(timing, errors="coerce")
            valid = valid[np.isfinite(valid)].to_numpy(dtype=float)
            total_sec = float(valid.sum()) if len(valid) else math.nan
            cpu_valid = np.asarray([], dtype=float)
            if metric == "ALL_metric_suite_total":
                cpu_series = pd.to_numeric(g["cpu_score_pair_total_sec"], errors="coerce")
                cpu_valid = cpu_series[np.isfinite(cpu_series)].to_numpy(dtype=float)
            retained_cache_mb = math.nan
            if metric.startswith(FROZEN_ALG_RUNTIME_METRIC) and "alg_frozen_pair_retained_cache_bytes" in g:
                retained = pd.to_numeric(
                    g["alg_frozen_pair_retained_cache_bytes"], errors="coerce"
                )
                retained = retained[np.isfinite(retained)].to_numpy(dtype=float)
                if len(retained):
                    retained_cache_mb = float(np.median(retained) / (1024.0 * 1024.0))
            rows.append(
                {
                    "support": support,
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": n_pairs,
                    "n_timed_pairs": len(valid),
                    "total_metric_sec": float(total_sec),
                    "mean_metric_ms_per_pair": float(np.mean(valid) * 1000.0) if len(valid) else math.nan,
                    "std_metric_ms_per_pair": float(np.std(valid, ddof=1) * 1000.0) if len(valid) > 1 else math.nan,
                    "min_metric_ms_per_pair": float(np.min(valid) * 1000.0) if len(valid) else math.nan,
                    "p50_metric_ms_per_pair": float(np.quantile(valid, 0.50) * 1000.0) if len(valid) else math.nan,
                    "p90_metric_ms_per_pair": float(np.quantile(valid, 0.90) * 1000.0) if len(valid) else math.nan,
                    "p95_metric_ms_per_pair": float(np.quantile(valid, 0.95) * 1000.0) if len(valid) else math.nan,
                    "p99_metric_ms_per_pair": float(np.quantile(valid, 0.99) * 1000.0) if len(valid) else math.nan,
                    "max_metric_ms_per_pair": float(np.max(valid) * 1000.0) if len(valid) else math.nan,
                    "total_cpu_sec": float(cpu_valid.sum()) if len(cpu_valid) else math.nan,
                    "mean_cpu_ms_per_pair": float(cpu_valid.mean() * 1000.0) if len(cpu_valid) else math.nan,
                    "p50_cpu_ms_per_pair": float(np.quantile(cpu_valid, 0.50) * 1000.0) if len(cpu_valid) else math.nan,
                    "p95_cpu_ms_per_pair": float(np.quantile(cpu_valid, 0.95) * 1000.0) if len(cpu_valid) else math.nan,
                    "max_cpu_ms_per_pair": float(cpu_valid.max() * 1000.0) if len(cpu_valid) else math.nan,
                    "median_retained_cache_mb_per_pair": retained_cache_mb,
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    out_dir: Path,
    pair_scores: pd.DataFrame,
    errors: list[dict],
    dataset_runtime: pd.DataFrame,
    pair_selection: str,
    evaluation_folds: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated = [
        evaluate_dataset_pairs_cv(g, n_splits=evaluation_folds)
        for _, g in pair_scores.groupby(["support", "dataset"])
    ]
    summary_frames = [summary for summary, _, _ in evaluated]
    fold_frames = [folds for _, folds, _ in evaluated]
    cv_errors = [
        {
            "support": str(group["support"].iloc[0]),
            "dataset": str(group["dataset"].iloc[0]),
            "error": error,
        }
        for (_, group), (_, _, error) in zip(
            pair_scores.groupby(["support", "dataset"]), evaluated
        )
        if error is not None
    ]
    summary_frames = [frame for frame in summary_frames if not frame.empty]
    fold_frames = [frame for frame in fold_frames if not frame.empty]
    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    fold_summary = pd.concat(fold_frames, ignore_index=True) if fold_frames else pd.DataFrame()
    global_test_summary = build_global_test_summary(summary)
    metric_efficiency = build_metric_efficiency(pair_scores)
    availability_rows = []
    for (support, dataset), group in pair_scores.groupby(["support", "dataset"], sort=False):
        for metric in RANKING_SCORE_COLS:
            values = pd.to_numeric(group.get(metric, pd.Series(np.nan, index=group.index)), errors="coerce")
            availability_rows.append(
                {
                    "support": support,
                    "dataset": dataset,
                    "metric": metric,
                    "n_pairs": len(group),
                    "n_finite": int(np.isfinite(values).sum()),
                    "n_missing": int((~np.isfinite(values)).sum()),
                    "complete": bool(np.isfinite(values).all()),
                }
            )
    metric_availability = pd.DataFrame(availability_rows)
    ranked = rank_metrics(summary) if not summary.empty else pd.DataFrame()
    winner_parts = []
    for _, group in summary.groupby(["support", "dataset"], sort=False):
        best = group["f1"].max()
        co_winners = group[np.isclose(group["f1"], best, rtol=0.0, atol=1e-12)].copy()
        co_winners["n_co_winners"] = len(co_winners)
        co_winners["fractional_win"] = 1.0 / len(co_winners)
        co_winners["is_strict_win"] = len(co_winners) == 1
        winner_parts.append(co_winners)
    winners = pd.concat(winner_parts, ignore_index=True) if winner_parts else pd.DataFrame()
    support_wins = (
        winners.groupby(["support", "metric"], as_index=False)
        .agg(
            n_dataset_wins=("fractional_win", "sum"),
            n_strict_wins=("is_strict_win", "sum"),
            n_tied_wins=("is_strict_win", lambda x: int((~x).sum())),
        )
        .sort_values(["support", "n_dataset_wins"], ascending=[True, False])
    ) if not winners.empty else pd.DataFrame()
    overall_wins = (
        winners.groupby("metric", as_index=False)
        .agg(
            n_dataset_wins=("fractional_win", "sum"),
            n_strict_wins=("is_strict_win", "sum"),
            n_tied_wins=("is_strict_win", lambda x: int((~x).sum())),
        )
        .sort_values("n_dataset_wins", ascending=False)
    ) if not winners.empty else pd.DataFrame()
    mean_rank = (
        ranked.groupby("metric", as_index=False)
        .agg(mean_rank=("rank", "mean"), mean_f1=("f1", "mean"), mean_auc=("roc_auc", "mean"))
        .sort_values("mean_rank")
    ) if not ranked.empty else pd.DataFrame()

    pair_scores.to_csv(out_dir / "pair_scores.csv", index=False)
    dataset_runtime.to_csv(out_dir / "dataset_runtime_summary.csv", index=False)
    metric_efficiency.to_csv(out_dir / "metric_efficiency_summary.csv", index=False)
    metric_availability.to_csv(out_dir / "metric_availability.csv", index=False)
    summary.to_csv(out_dir / "metric_summary.csv", index=False)
    fold_summary.to_csv(out_dir / "metric_summary_folds.csv", index=False)
    pd.DataFrame(cv_errors).to_csv(out_dir / "cv_unavailable_datasets.csv", index=False)
    global_test_summary.to_csv(out_dir / "metric_summary_global_test.csv", index=False)
    ranked.to_csv(out_dir / "metric_summary_ranked.csv", index=False)
    winners.to_csv(out_dir / "winner_by_dataset.csv", index=False)
    support_wins.to_csv(out_dir / "winner_counts_by_support.csv", index=False)
    overall_wins.to_csv(out_dir / "winner_counts_overall.csv", index=False)
    mean_rank.to_csv(out_dir / "mean_rank_overall.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "skipped_or_failed_datasets.csv", index=False)

    payload = {
        "n_pair_scores": int(len(pair_scores)),
        "n_datasets": int(summary["dataset"].nunique()) if not summary.empty else 0,
        "n_supports": int(summary["support"].nunique()) if not summary.empty else 0,
        "score_cols": RANKING_SCORE_COLS,
        "ranking_score_cols": RANKING_SCORE_COLS,
        "pair_score_cols": PAIR_SCORE_COLS,
        "pair_selection": pair_selection,
        "evaluation_protocol": "stratified_5fold_out_of_fold_scalar_threshold_v1",
        "evaluation_folds": evaluation_folds,
        "classifier": "none; one scalar threshold per metric and training fold",
        "pair_protocol": (
            "fixed_disjoint_cloud_cartesian_v2"
            if pair_selection == "all"
            else "balanced_sampled_pairs_v1"
        ),
        "pointset_metric_config": {
            "manifold_k": POINTSET_CONFIG.manifold_k,
            "prc_k": POINTSET_CONFIG.prc_k,
            "prc_c": POINTSET_CONFIG.prc_c,
            "point_fscore_threshold": POINTSET_CONFIG.point_fscore_threshold,
            "max_points": POINTSET_CONFIG.max_points,
            "sliced_projections": POINTSET_CONFIG.sliced_projections,
            "dcd_alpha": POINTSET_CONFIG.dcd_alpha,
            "sinkhorn_reg": POINTSET_CONFIG.sinkhorn_reg,
            "prd_clusters": POINTSET_CONFIG.prd_clusters,
            "prd_angles": POINTSET_CONFIG.prd_angles,
            "c2st_folds": POINTSET_CONFIG.c2st_folds,
            "alpha_beta_steps": POINTSET_CONFIG.alpha_beta_steps,
            "geometry_landmarks": POINTSET_CONFIG.geometry_landmarks,
            "geometry_repeats": POINTSET_CONFIG.geometry_repeats,
            "geometry_i_max": POINTSET_CONFIG.geometry_i_max,
            "geometry_gamma": POINTSET_CONFIG.geometry_gamma,
        },
        "files": [
            "pair_scores.csv",
            "dataset_runtime_summary.csv",
            "metric_efficiency_summary.csv",
            "metric_availability.csv",
            "metric_summary.csv",
            "metric_summary_folds.csv",
            "cv_unavailable_datasets.csv",
            "metric_summary_global_test.csv",
            "metric_summary_ranked.csv",
            "winner_by_dataset.csv",
            "winner_counts_by_support.csv",
            "winner_counts_overall.csv",
            "mean_rank_overall.csv",
            "skipped_or_failed_datasets.csv",
        ],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2))


def write_pair_scores_only(
    out_dir: Path,
    pair_scores: pd.DataFrame,
    errors: list[dict],
    dataset_runtime: pd.DataFrame,
    pair_selection: str,
    evaluation_folds: int,
) -> None:
    """Persist raw pair scores/checkpoints without running metric-level CV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_scores.to_csv(out_dir / "pair_scores.csv", index=False)
    dataset_runtime.to_csv(out_dir / "dataset_runtime_summary.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "skipped_or_failed_datasets.csv", index=False)
    payload = {
        "pair_selection": pair_selection,
        "evaluation_folds_reserved_for_downstream": evaluation_folds,
        "evaluation_deferred": True,
        "n_pair_rows": len(pair_scores),
        "n_datasets": int(pair_scores["dataset"].nunique()) if "dataset" in pair_scores else 0,
        "n_score_columns": len(RANKING_SCORE_COLS),
        "files": ["pair_scores.csv", "dataset_runtime_summary.csv", "skipped_or_failed_datasets.csv"],
    }
    (out_dir / "run_summary.json").write_text(json.dumps(payload, indent=2))


def safe_run(
    name: str,
    func: Callable[[], DatasetResult],
    strict: bool,
    errors: list[dict],
) -> DatasetResult | None:
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        result = func()
        elapsed = time.perf_counter() - t0
        cpu_elapsed = time.process_time() - cpu0
        raw_rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        peak_rss_mb = raw_rss / (1024.0 * 1024.0) if sys.platform == "darwin" else raw_rss / 1024.0
        result.elapsed_sec = elapsed
        result.cpu_sec = cpu_elapsed
        result.peak_rss_mb = peak_rss_mb
        print(
            f"done {result.support}/{result.dataset}: {len(result.pairs)} pairs | "
            f"wall={elapsed:.2f}s cpu={cpu_elapsed:.2f}s peak_process_rss={peak_rss_mb:.1f}MB"
        )
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        cpu_elapsed = time.process_time() - cpu0
        msg = f"{type(exc).__name__}: {exc}"
        errors.append(
            {"dataset": name, "error": msg, "elapsed_sec": elapsed, "cpu_sec": cpu_elapsed}
        )
        print(f"skip {name}: {msg}")
        if strict:
            raise
        return None


def append_checkpoint(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    """Replace a checkpoint only after its complete successor was written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_csv_or_empty(path: Path, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()


def load_seed_catalog(seed_dir: Path | None, datasets: set[str]) -> dict[str, tuple[Path, int]]:
    """Resolve historical partitions without loading their potentially huge CSVs."""
    if seed_dir is None or not datasets:
        return {}
    manifest_path = seed_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing seed manifest: {manifest_path}")
    manifest = read_csv_or_empty(manifest_path)
    required = {"dataset", "relative_path"}
    if manifest.empty or not required.issubset(manifest.columns):
        raise ValueError(f"Invalid seed manifest: {manifest_path}")
    selected = manifest[manifest["dataset"].astype(str).isin(datasets)]
    catalog = {}
    for _, row in selected.iterrows():
        path = seed_dir / str(row["relative_path"])
        if path.exists():
            catalog[str(row["dataset"])] = (path, int(row.get("n_rows", 0)))
    return catalog


def canonical_pair_frame(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["object_a", "object_b", "label"]
    out = frame[keys].copy()
    for key in keys:
        out[key] = out[key].map(canonical_pair_value)
    return out


def filter_seed_to_manifest(
    seed_path: Path,
    manifest_pairs: pd.DataFrame,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Stream a huge historical partition and retain only current pair identities."""
    if manifest_pairs.empty:
        return pd.DataFrame()
    target = pd.MultiIndex.from_frame(canonical_pair_frame(manifest_pairs)).drop_duplicates()
    matches = []
    for chunk in pd.read_csv(seed_path, chunksize=chunksize, low_memory=False):
        if not {"object_a", "object_b", "label"}.issubset(chunk.columns):
            raise ValueError(f"Seed partition lacks pair identity columns: {seed_path}")
        chunk_index = pd.MultiIndex.from_frame(canonical_pair_frame(chunk))
        selected = chunk.loc[chunk_index.isin(target)]
        if not selected.empty:
            matches.append(selected)
    if not matches:
        return pd.DataFrame()
    return pd.concat(matches, ignore_index=True, sort=False)


def missing_metrics_by_dataset(pair_scores: pd.DataFrame) -> dict[str, set[str]]:
    if pair_scores.empty or "dataset" not in pair_scores.columns:
        return {}
    missing: dict[str, set[str]] = {}
    for dataset, group in pair_scores.groupby("dataset", sort=False):
        dataset_missing = set()
        for metric in RANKING_SCORE_COLS:
            if metric not in group.columns:
                dataset_missing.add(metric)
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            if not np.isfinite(values).all():
                dataset_missing.add(metric)
        missing[str(dataset)] = dataset_missing
    return missing


def completed_dataset_names(pair_scores: pd.DataFrame) -> set[str]:
    return {dataset for dataset, missing in missing_metrics_by_dataset(pair_scores).items() if not missing}


def dataset_names_from_output_dir(out_dir: Path) -> set[str]:
    """Only trust pair-level outputs containing every requested metric."""
    final_path = out_dir / "pair_scores.csv"
    checkpoint_path = out_dir / "_checkpoint_pair_scores.csv"
    path = final_path if final_path.exists() else checkpoint_path
    if not path.exists():
        return set()
    return completed_dataset_names(read_csv_or_empty(path, low_memory=False))


def reset_rng_for_dataset(rng: np.random.Generator, seed: int, dataset: str) -> None:
    """Make sampled pairs independent of job order, failures, and resume skips."""
    digest = hashlib.blake2b(dataset.encode("utf-8"), digest_size=8).digest()
    dataset_seed = (int.from_bytes(digest, "little") + int(seed)) % (2**63 - 1)
    rng.bit_generator.state = np.random.default_rng(dataset_seed).bit_generator.state


def merge_pair_score_updates(existing: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    """Fill current-manifest rows from old scores, then overlay new values.

    For every dataset present in ``updates``, its rows are authoritative. This
    lets an all-pairs historical file seed a smaller sampled protocol without
    leaking millions of out-of-protocol rows into the new output.
    """
    if existing.empty:
        return updates.copy()
    if updates.empty:
        return existing.copy()
    keys = [c for c in ["support", "dataset", "object_a", "object_b", "label"] if c in existing.columns and c in updates.columns]
    if not keys:
        return pd.concat([existing, updates], ignore_index=True)

    replaced = set(updates["dataset"].astype(str).unique()) if "dataset" in updates else set()
    kept = (
        existing[~existing["dataset"].astype(str).isin(replaced)].copy()
        if replaced and "dataset" in existing
        else pd.DataFrame()
    )
    old = (
        existing[existing["dataset"].astype(str).isin(replaced)].copy()
        if replaced and "dataset" in existing
        else existing.copy()
    )
    new = updates.copy()
    for key in keys:
        old[key] = old[key].map(canonical_pair_value)
        new[key] = new[key].map(canonical_pair_value)
    old["_pair_occurrence"] = old.groupby(keys, dropna=False).cumcount()
    new["_pair_occurrence"] = new.groupby(keys, dropna=False).cumcount()
    index_cols = keys + ["_pair_occurrence"]
    old = old.set_index(index_cols)
    new = new.set_index(index_cols)

    # ``combine_first`` performs the same new-value-first overlay in one
    # vectorized operation. In particular, it avoids inserting roughly 1,320
    # ALG columns one at a time when resuming a full-grid shard.
    old_on_new = old.reindex(new.index)
    current = new.combine_first(old_on_new).reset_index().drop(columns=["_pair_occurrence"])
    return pd.concat([kept, current], ignore_index=True, sort=False)


def pair_manifest_match_fraction(existing: pd.DataFrame, updates: pd.DataFrame) -> float:
    keys = [c for c in ["support", "dataset", "object_a", "object_b", "label"] if c in existing.columns and c in updates.columns]
    if not keys or existing.empty or updates.empty:
        return 0.0
    old = existing[keys].copy()
    new = updates[keys].copy()
    for key in keys:
        old[key] = old[key].map(canonical_pair_value)
        new[key] = new[key].map(canonical_pair_value)
    old["_pair_occurrence"] = old.groupby(keys, dropna=False).cumcount()
    new["_pair_occurrence"] = new.groupby(keys, dropna=False).cumcount()
    old_index = pd.MultiIndex.from_frame(old)
    new_index = pd.MultiIndex.from_frame(new)
    return len(old_index.intersection(new_index)) / max(len(new_index), 1)


def parse_hidden_sizes(text: str) -> list[tuple[int, ...]]:
    sizes = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        sizes.append(tuple(int(x) for x in chunk.split("-")))
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/overlap_metric_large_scale"),
    )
    parser.add_argument("--preset", choices=["offline", "medium", "full"], default="offline")
    parser.add_argument(
        "--n-pairs",
        type=str,
        default="800",
        help="Maximum sampled pairs per dataset, or 'all' for all available pairs.",
    )
    parser.add_argument("--cloud-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--evaluation-folds",
        type=int,
        default=5,
        help="Stratified out-of-fold threshold evaluation; the ALG campaign uses 5.",
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing pair-score cells and compute only missing (pair, metric) cells.",
    )
    parser.add_argument(
        "--pair-scores-only",
        action="store_true",
        help="Save raw pair scores and defer cross-validation to a downstream evaluator.",
    )
    parser.add_argument(
        "--seed-results-dir",
        type=Path,
        default=None,
        help="Historical pair-score partitions prepared by prepare_pair_metric_seed.py.",
    )
    parser.add_argument("--ucr-root", type=Path, default=None)
    parser.add_argument(
        "--ucr-max-datasets",
        type=int,
        default=25,
        help="Maximum UCR/extra time-series datasets. Use 0 for all datasets.",
    )
    parser.add_argument("--ucr-window", type=int, default=24)
    parser.add_argument("--ucr-stride", type=int, default=4)
    parser.add_argument(
        "--ts-roots",
        type=str,
        default="",
        help="Extra time-series roots as PREFIX:/path, comma-separated, e.g. UEA:/data/UEAArchive_2018.",
    )
    parser.add_argument("--glue-tasks", type=str, default="stsb,mrpc")
    parser.add_argument("--text-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--tudatasets", type=str, default="MUTAG,PROTEINS,ENZYMES,NCI1")
    parser.add_argument("--tudataset-max-items", type=int, default=5000)
    parser.add_argument("--torchvision-datasets", type=str, default="")
    parser.add_argument("--torchvision-max-items", type=int, default=5000)
    parser.add_argument("--torchvision-root", type=Path, default=Path("/tmp/torchvision"))
    parser.add_argument("--manifold-k", type=int, default=5)
    parser.add_argument("--prc-k", type=int, default=3)
    parser.add_argument("--prc-c", type=int, default=3)
    parser.add_argument(
        "--point-fscore-threshold",
        type=float,
        default=0.10,
        help="Global distance threshold after dataset-level cloud scaling.",
    )
    parser.add_argument(
        "--metric-max-points",
        type=int,
        default=64,
        help="Deterministic cap per cloud for expensive OT/topological metrics; 0 disables the cap.",
    )
    parser.add_argument("--sliced-wasserstein-projections", type=int, default=32)
    parser.add_argument("--dcd-alpha", type=float, default=100.0)
    parser.add_argument("--sinkhorn-reg", type=float, default=0.10)
    parser.add_argument("--prd-clusters", type=int, default=20)
    parser.add_argument("--prd-angles", type=int, default=1001)
    parser.add_argument("--c2st-folds", type=int, default=5)
    parser.add_argument("--alpha-beta-steps", type=int, default=30)
    parser.add_argument("--geometry-landmarks", type=int, default=32)
    parser.add_argument("--geometry-repeats", type=int, default=16)
    parser.add_argument("--geometry-i-max", type=int, default=10)
    parser.add_argument("--geometry-gamma", type=float, default=0.125)
    parser.add_argument(
        "--include-alg-grid",
        action="store_true",
        help="Evaluate all 1,320 ALG variants (large wide pair-score CSV).",
    )
    parser.add_argument(
        "--include-tuned-baseline-grid",
        action="store_true",
        help="Evaluate 39 development-selectable variants of seven leading baselines.",
    )
    parser.add_argument(
        "--include-alg-framework-components",
        action="store_true",
        help="Persist the 64 local and 240 global components used by the 168,960-framework grid.",
    )
    parser.add_argument(
        "--alg-framework-components-only",
        action="store_true",
        help="Compute only the 300 reusable ALG framework components (no baseline scores).",
    )
    parser.add_argument(
        "--benchmark-frozen-alg",
        action="store_true",
        help="Measure the selected cosine-rational ALG alone in cold and cached modes.",
    )
    parser.add_argument(
        "--alg-star-runtime-only",
        action="store_true",
        help="Measure only frozen ALG* and Initial ALG end-to-end from raw clouds.",
    )
    parser.add_argument(
        "--alg-runtime-repetitions",
        type=int,
        default=3,
        help="Independent cache-free repetitions per method and pair; median is persisted.",
    )
    parser.add_argument("--openml", type=str, default="")
    parser.add_argument(
        "--openml-suites",
        type=str,
        default="",
        help="OpenML benchmark suites, comma-separated. Use 99 for OpenML-CC18.",
    )
    parser.add_argument("--openml-suite-limit", type=int, default=0)
    parser.add_argument("--openml-max-rows", type=int, default=5000)
    parser.add_argument("--neural-seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--neural-hidden-sizes", type=str, default="32,64,128,64-32")
    parser.add_argument(
        "--skip-datasets-from",
        type=str,
        default="",
        help="Comma-separated output directories whose completed datasets should be skipped.",
    )
    parser.add_argument(
        "--dataset-allowlist",
        type=Path,
        default=None,
        help="Optional CSV containing a dataset column; only listed dataset jobs are run.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--list-jobs",
        action="store_true",
        help="Print the resolved dataset list and assigned shard, then exit without computing pairs.",
    )
    parser.add_argument(
        "--job-manifest-out",
        type=Path,
        help="With --list-jobs, persist the exact scheduled dataset/shard manifest as CSV.",
    )
    args = parser.parse_args()
    if args.include_alg_grid:
        enable_alg_grid()
    if args.include_tuned_baseline_grid:
        enable_tuned_baseline_grid()
    if args.include_alg_framework_components or args.alg_framework_components_only:
        enable_alg_framework_components()
    if args.alg_framework_components_only:
        PAIR_SCORE_COLS[:] = list(ALG_FRAMEWORK_COMPONENT_SCORE_COLS)
        RANKING_SCORE_COLS[:] = list(ALG_FRAMEWORK_COMPONENT_SCORE_COLS)
    if args.benchmark_frozen_alg:
        enable_frozen_alg_runtime()
    if args.alg_star_runtime_only:
        enable_alg_star_end_to_end_runtime()
        PAIR_SCORE_COLS[:] = [ALG_STAR_METRIC, INITIAL_ALG_METRIC]
        RANKING_SCORE_COLS[:] = [ALG_STAR_METRIC, INITIAL_ALG_METRIC]
        configure_alg_star_runtime(args.alg_runtime_repetitions)
    args.n_pairs = parse_n_pairs(args.n_pairs)
    pair_selection = "all" if args.n_pairs is None else str(args.n_pairs)
    if args.shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if args.evaluation_folds < 2:
        raise ValueError("--evaluation-folds must be >= 2")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard-index must satisfy 0 <= index < shard-count")
    if args.manifold_k < 1:
        raise ValueError("--manifold-k must be >= 1")
    if args.prc_k < 1 or args.prc_c < 1:
        raise ValueError("--prc-k and --prc-c must be >= 1")
    if args.point_fscore_threshold <= 0:
        raise ValueError("--point-fscore-threshold must be > 0")
    if args.sliced_wasserstein_projections < 1:
        raise ValueError("--sliced-wasserstein-projections must be >= 1")
    configure_metrics(
        manifold_k=args.manifold_k,
        prc_k=args.prc_k,
        prc_c=args.prc_c,
        point_fscore_threshold=args.point_fscore_threshold,
        max_points=args.metric_max_points,
        sliced_projections=args.sliced_wasserstein_projections,
        dcd_alpha=args.dcd_alpha,
        sinkhorn_reg=args.sinkhorn_reg,
        prd_clusters=args.prd_clusters,
        prd_angles=args.prd_angles,
        c2st_folds=args.c2st_folds,
        alpha_beta_steps=args.alpha_beta_steps,
        geometry_landmarks=args.geometry_landmarks,
        geometry_repeats=args.geometry_repeats,
        geometry_i_max=args.geometry_i_max,
        geometry_gamma=args.geometry_gamma,
    )

    rng = np.random.default_rng(args.seed)
    results: list[DatasetResult] = []
    errors: list[dict] = []

    jobs: list[tuple[str, Callable[[], DatasetResult]]] = [
        ("sklearn_digits_patch_clouds", lambda: run_digits_images(args.n_pairs, rng)),
        (
            "sklearn_iris_class_clouds",
            lambda: run_tabular_class_cloud(load_iris, "sklearn_iris_class_clouds", args.n_pairs, args.cloud_size, rng),
        ),
        (
            "sklearn_wine_class_clouds",
            lambda: run_tabular_class_cloud(load_wine, "sklearn_wine_class_clouds", args.n_pairs, args.cloud_size, rng),
        ),
        (
            "sklearn_breast_cancer_class_clouds",
            lambda: run_tabular_class_cloud(
                load_breast_cancer,
                "sklearn_breast_cancer_class_clouds",
                args.n_pairs,
                args.cloud_size,
                rng,
            ),
        ),
    ]

    if args.preset in {"medium", "full"}:
        seeds = [int(x) for x in args.neural_seeds.split(",") if x.strip()]
        hidden_sizes = parse_hidden_sizes(args.neural_hidden_sizes)
        jobs.append(
            (
                "digits_mlp_activation_clouds",
                lambda: run_neural_mlp_models(args.n_pairs, rng, seeds, hidden_sizes),
            )
        )

    if args.ucr_root is not None and args.ucr_root.exists():
        ucr_dirs = [p for p in sorted(args.ucr_root.iterdir()) if p.is_dir()]
        for dataset_dir in maybe_limit(ucr_dirs, args.ucr_max_datasets):
            jobs.append(
                (
                    f"UCR_{dataset_dir.name}",
                    lambda dataset_dir=dataset_dir: run_ucr_dataset(
                        dataset_dir,
                        args.n_pairs,
                        rng,
                        args.ucr_window,
                        args.ucr_stride,
                        "UCR",
                    ),
                )
            )

    for spec in [x.strip() for x in args.ts_roots.split(",") if x.strip()]:
        if ":" not in spec:
            errors.append({"dataset": spec, "error": "Invalid --ts-roots spec, expected PREFIX:/path", "elapsed_sec": 0.0})
            continue
        prefix, root_text = spec.split(":", 1)
        root = Path(root_text).expanduser()
        if not root.exists():
            errors.append({"dataset": f"{prefix}:{root}", "error": "time-series root does not exist", "elapsed_sec": 0.0})
            continue
        ts_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
        for dataset_dir in maybe_limit(ts_dirs, args.ucr_max_datasets):
            jobs.append(
                (
                    f"{prefix}_{dataset_dir.name}",
                    lambda dataset_dir=dataset_dir, prefix=prefix: run_ucr_dataset(
                        dataset_dir,
                        args.n_pairs,
                        rng,
                        args.ucr_window,
                        args.ucr_stride,
                        prefix,
                    ),
                )
            )

    if args.preset == "full":
        glue_tasks = MAX_GLUE_PAIR_TASKS if args.glue_tasks.strip().lower() == "max" else [
            t.strip() for t in args.glue_tasks.split(",") if t.strip()
        ]
        for task in glue_tasks:
            jobs.append(
                (
                    f"GLUE_{task}",
                    lambda task=task: run_text_glue(task, args.n_pairs, rng, args.text_model),
                )
            )

        tudataset_names = MAX_TUDATASETS if args.tudatasets.strip().lower() == "max" else [
            t.strip() for t in args.tudatasets.split(",") if t.strip()
        ]
        for name in tudataset_names:
            jobs.append(
                (
                    f"TUDataset_{name}",
                    lambda name=name: run_tudataset(name, args.n_pairs, rng, args.tudataset_max_items),
                )
            )

        torchvision_names = (
            MAX_TORCHVISION_DATASETS
            if args.torchvision_datasets.strip().lower() == "max"
            else [t.strip() for t in args.torchvision_datasets.split(",") if t.strip()]
        )
        for name in torchvision_names:
            jobs.append(
                (
                    f"TorchVision_{name}",
                    lambda name=name: run_torchvision_images(
                        name,
                        args.n_pairs,
                        rng,
                        args.torchvision_max_items,
                        args.torchvision_root,
                    ),
                )
            )

        openml_specs: list[tuple[int, str]] = []
        for spec in [x.strip() for x in args.openml.split(",") if x.strip()]:
            if ":" in spec:
                openml_id_text, openml_name = spec.split(":", 1)
            else:
                openml_id_text, openml_name = spec, f"openml_{spec}"
            openml_specs.append((int(openml_id_text), f"OpenML_{openml_name}"))

        for suite_text in [x.strip() for x in args.openml_suites.split(",") if x.strip()]:
            try:
                openml_specs.extend(resolve_openml_suite_specs(suite_text, args.openml_suite_limit))
            except Exception as exc:
                errors.append(
                    {
                        "dataset": f"OpenML_suite_{suite_text}",
                        "error": f"{type(exc).__name__}: {exc}",
                        "elapsed_sec": 0.0,
                    }
                )

        seen_openml = set()
        for openml_id, openml_name in openml_specs:
            if openml_id in seen_openml:
                continue
            seen_openml.add(openml_id)
            jobs.append(
                (
                    openml_name,
                    lambda openml_id=openml_id, openml_name=openml_name: run_openml_dataset(
                        openml_id,
                        openml_name,
                        args.n_pairs,
                        args.cloud_size,
                        rng,
                        args.openml_max_rows,
                    ),
                )
            )

    skip_datasets: set[str] = set()
    for out_dir_text in [x.strip() for x in args.skip_datasets_from.split(",") if x.strip()]:
        skip_datasets.update(dataset_names_from_output_dir(Path(out_dir_text).expanduser()))
    if skip_datasets:
        before = len(jobs)
        jobs = [(name, job) for name, job in jobs if name not in skip_datasets]
        print(f"skip-datasets-from: {len(skip_datasets)} completed datasets found; {before - len(jobs)} jobs skipped")

    if args.dataset_allowlist is not None:
        allowlist = pd.read_csv(args.dataset_allowlist)
        if "dataset" not in allowlist.columns:
            raise ValueError("--dataset-allowlist CSV must contain a 'dataset' column")
        allowed = set(allowlist["dataset"].dropna().astype(str))
        before = len(jobs)
        jobs = [(name, job) for name, job in jobs if name in allowed]
        unresolved = allowed - {name for name, _ in jobs}
        if unresolved:
            sample = ", ".join(sorted(unresolved)[:10])
            raise ValueError(
                f"dataset allowlist contains {len(unresolved)} unresolved jobs; first: {sample}"
            )
        print(f"dataset-allowlist: retained {len(jobs)} of {before} resolved jobs")

    if args.list_jobs:
        manifest = pd.DataFrame(
            [
                {"job_index": index, "shard_index": index % args.shard_count, "dataset": name}
                for index, (name, _) in enumerate(jobs)
            ]
        )
        if args.job_manifest_out:
            args.job_manifest_out.parent.mkdir(parents=True, exist_ok=True)
            manifest.to_csv(args.job_manifest_out, index=False)
        print(f"Resolved dataset jobs: {len(jobs)}")
        print(f"Pair budget per dataset: {pair_selection}")
        print(f"Shard count: {args.shard_count}")
        for row in manifest.itertuples(index=False):
            print(f"{row.job_index:4d}\tshard_{row.shard_index}\t{row.dataset}")
        return

    if args.shard_count > 1:
        before = len(jobs)
        jobs = jobs[args.shard_index :: args.shard_count]
        print(f"shard {args.shard_index + 1}/{args.shard_count}: {len(jobs)} of {before} remaining jobs")
        if not jobs:
            print("This shard has no assigned datasets; exiting successfully.")
            return

    seed_catalog = load_seed_catalog(
        args.seed_results_dir.expanduser() if args.seed_results_dir else None,
        {name for name, _ in jobs},
    )
    if seed_catalog:
        print(
            f"seed: found {len(seed_catalog)} historical dataset partitions; "
            "large files will be filtered by streaming"
        )

    existing_pair_scores = pd.DataFrame()
    existing_runtime = pd.DataFrame()
    existing_errors = pd.DataFrame()
    existing_datasets: set[str] = set()
    existing_missing_metrics: dict[str, set[str]] = {}
    if args.resume:
        pair_scores_path = args.out_dir / "pair_scores.csv"
        checkpoint_pair_scores_path = args.out_dir / "_checkpoint_pair_scores.csv"
        runtime_path = args.out_dir / "dataset_runtime_summary.csv"
        checkpoint_runtime_path = args.out_dir / "_checkpoint_dataset_runtime.csv"
        errors_path = args.out_dir / "skipped_or_failed_datasets.csv"
        existing_pair_frames = []
        if pair_scores_path.exists():
            existing_pair_frames.append(read_csv_or_empty(pair_scores_path, low_memory=False))
        elif checkpoint_pair_scores_path.exists():
            existing_pair_frames.append(read_csv_or_empty(checkpoint_pair_scores_path, low_memory=False))
        existing_pair_frames = [df for df in existing_pair_frames if not df.empty]
        if existing_pair_frames:
            existing_pair_scores = pd.concat(existing_pair_frames, ignore_index=True)
            existing_missing_metrics = missing_metrics_by_dataset(existing_pair_scores)
            existing_datasets = {
                dataset for dataset, missing in existing_missing_metrics.items() if not missing
            }
        existing_runtime_frames = []
        if runtime_path.exists():
            existing_runtime_frames.append(read_csv_or_empty(runtime_path))
        elif checkpoint_runtime_path.exists():
            existing_runtime_frames.append(read_csv_or_empty(checkpoint_runtime_path))
        existing_runtime_frames = [df for df in existing_runtime_frames if not df.empty]
        if existing_runtime_frames:
            existing_runtime = pd.concat(existing_runtime_frames, ignore_index=True)
        if errors_path.exists():
            existing_errors = read_csv_or_empty(errors_path)

    if existing_datasets:
        before = len(jobs)
        jobs = [(name, job) for name, job in jobs if name not in existing_datasets]
        print(f"resume: {len(existing_datasets)} complete datasets found; {before - len(jobs)} jobs skipped")

    for name, job in jobs:
        reset_rng_for_dataset(rng, args.seed, name)
        existing_for_dataset = (
            existing_pair_scores[existing_pair_scores["dataset"].astype(str) == name]
            if not existing_pair_scores.empty and "dataset" in existing_pair_scores.columns
            else pd.DataFrame()
        )
        seed_for_dataset = pd.DataFrame()
        current_manifest = pd.DataFrame()
        if existing_for_dataset.empty and name in seed_catalog:
            seed_path, historical_rows = seed_catalog[name]
            print(
                f"seed {name}: building current pair manifest before scanning "
                f"{historical_rows} historical rows"
            )
            set_active_ranking_metrics(set())
            set_active_pair_resume_plan(None)
            manifest_result = safe_run(f"{name} [manifest]", job, args.strict, errors)
            if manifest_result is None or manifest_result.pairs.empty:
                continue
            current_manifest = manifest_result.pairs
            seed_for_dataset = filter_seed_to_manifest(seed_path, current_manifest)
            match_fraction = pair_manifest_match_fraction(seed_for_dataset, current_manifest)
            print(
                f"seed {name}: retained {len(seed_for_dataset)} historical rows; "
                f"current-pair coverage={match_fraction:.1%}"
            )
            reset_rng_for_dataset(rng, args.seed, name)
        resume_source = existing_for_dataset if not existing_for_dataset.empty else seed_for_dataset
        source_missing = missing_metrics_by_dataset(resume_source).get(name, set(RANKING_SCORE_COLS))
        missing_for_dataset = source_missing
        # The per-pair plan is authoritative. Keeping the dataset-level filter
        # open ensures pairs absent from a partial historical seed compute all
        # metrics instead of inheriting the union of matched rows.
        set_active_ranking_metrics(None)
        set_active_pair_resume_plan(resume_source)
        if not resume_source.empty:
            missing_cells = sum(
                int(
                    (~np.isfinite(
                        pd.to_numeric(
                            resume_source.get(metric, pd.Series(np.nan, index=resume_source.index)),
                            errors="coerce",
                        ).to_numpy(dtype=float)
                    )).sum()
                )
                for metric in RANKING_SCORE_COLS
            )
            print(
                f"resume {name}: {missing_cells} missing pair-metric cells "
                f"across {len(missing_for_dataset)} metrics"
            )
        result = safe_run(name, job, args.strict, errors)
        if result is not None and not result.pairs.empty:
            results.append(result)
            completed_rows = merge_pair_score_updates(resume_source, result.pairs)
            completed_rows = add_rank_overlap_scores(completed_rows)
            existing_pair_scores = merge_pair_score_updates(existing_pair_scores, completed_rows)
            runtime_row = pd.DataFrame(
                [{
                    "support": result.support,
                    "dataset": result.dataset,
                    "n_pairs": len(result.pairs),
                    "elapsed_sec": result.elapsed_sec,
                    "elapsed_min": result.elapsed_sec / 60.0,
                    "cpu_sec": result.cpu_sec,
                    "cpu_to_wall_ratio": result.cpu_sec / result.elapsed_sec if result.elapsed_sec > 0 else math.nan,
                    "process_peak_rss_mb_after_dataset": result.peak_rss_mb,
                }]
            )
            existing_runtime = pd.concat([existing_runtime, runtime_row], ignore_index=True)
            existing_runtime = existing_runtime.drop_duplicates(subset=["dataset"], keep="last")
            write_csv_atomic(args.out_dir / "_checkpoint_pair_scores.csv", existing_pair_scores)
            write_csv_atomic(args.out_dir / "_checkpoint_dataset_runtime.csv", existing_runtime)
        set_active_pair_resume_plan(None)
    set_active_ranking_metrics(None)

    if not results and existing_pair_scores.empty:
        raise RuntimeError("No dataset produced pair scores.")

    pair_scores = existing_pair_scores
    dataset_runtime = existing_runtime
    if not dataset_runtime.empty and "dataset" in dataset_runtime.columns:
        dataset_runtime = dataset_runtime.drop_duplicates(subset=["dataset"], keep="last")

    errors_df = pd.concat([existing_errors, pd.DataFrame(errors)], ignore_index=True)
    if not errors_df.empty and "dataset" in errors_df.columns:
        generated = set(pair_scores["dataset"].astype(str).unique())
        errors_df = errors_df[~errors_df["dataset"].astype(str).isin(generated)]
        errors_df = errors_df.drop_duplicates(subset=["dataset", "error"], keep="last")
    errors = errors_df.to_dict("records") if not errors_df.empty else []

    output_writer = write_pair_scores_only if args.pair_scores_only else write_outputs
    output_writer(
        args.out_dir,
        pair_scores,
        errors,
        dataset_runtime,
        pair_selection,
        args.evaluation_folds,
    )

    print(f"\nSaved outputs to: {args.out_dir.resolve()}")
    if not args.pair_scores_only:
        print("Top 30 metrics by dataset wins:")
        print(pd.read_csv(args.out_dir / "winner_counts_overall.csv").head(30).to_string(index=False))
    global_path = args.out_dir / "metric_summary_global_test.csv"
    if global_path.exists():
        print("\nTop 30 global test metrics:")
        print(pd.read_csv(global_path).head(30).to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# Expanded ALG framework campaign.  The only search-space change relative to
# the historical 158,400 run is Manifold PR-F1 at k=1, yielding:
# 64 local x 240 global x 11 lambda = 168,960 configurations.

export SHARD_COUNT="${SHARD_COUNT:-128}"
export ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-128}"
export COMPONENT_ROOT="${COMPONENT_ROOT:-outputs/alg_framework_components_168960_243}"
export RESULT_ROOT="${RESULT_ROOT:-outputs/alg_framework_results_168960_243}"
export AGGREGATE_ROOT="${AGGREGATE_ROOT:-outputs/alg_framework_results_168960_243_aggregate}"

exec bash slurm/submit_alg_framework_158400_campaign.sh

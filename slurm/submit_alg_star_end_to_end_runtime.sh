#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARD_COUNT="${SHARD_COUNT:-128}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-128}"
SOURCE_PARTITION="${SOURCE_PARTITION:-outputs/alg_framework_results_168960_243_aggregate/eligible_partition_60dev_183test.csv}"
RUNTIME_ROOT="${RUNTIME_ROOT:-outputs/alg_star_end_to_end_runtime_test183}"
AGGREGATE_ROOT="${AGGREGATE_ROOT:-outputs/alg_star_end_to_end_runtime_test183_aggregate}"
TEST_PARTITION="${TEST_PARTITION:-${RUNTIME_ROOT}/test183_allowlist.csv}"

if [[ -n "${ALG_PARTITION:-}" ]]; then
  partition_args=(--partition="$ALG_PARTITION")
else
  partition_args=()
fi

mkdir -p "$RUNTIME_ROOT" "$AGGREGATE_ROOT" logs

prepare_job="$(sbatch --parsable "${partition_args[@]}" \
  --output="logs/algstar_runtime_prepare_%j.out" \
  --error="logs/algstar_runtime_prepare_%j.err" \
  --export="ALL,ALG_STAR_SOURCE_PARTITION=$SOURCE_PARTITION,ALG_STAR_TEST_PARTITION=$TEST_PARTITION" \
  slurm/submit_alg_star_runtime_prepare.slurm | cut -d';' -f1)"

runtime_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${prepare_job}" \
  --array="0-$((SHARD_COUNT - 1))%${ARRAY_PARALLELISM}" \
  --output="logs/algstar_runtime_%A_%a.out" \
  --error="logs/algstar_runtime_%A_%a.err" \
  --export="ALL,ALG_STAR_TEST_PARTITION=$TEST_PARTITION,ALG_STAR_RUNTIME_ROOT=$RUNTIME_ROOT,ALG_STAR_SHARD_COUNT=$SHARD_COUNT" \
  slurm/submit_alg_star_runtime_task.slurm | cut -d';' -f1)"

aggregate_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${runtime_job}" \
  --output="logs/algstar_runtime_aggregate_%j.out" \
  --error="logs/algstar_runtime_aggregate_%j.err" \
  --export="ALL,ALG_STAR_RUNTIME_ROOT=$RUNTIME_ROOT,ALG_STAR_RUNTIME_AGGREGATE=$AGGREGATE_ROOT,ALG_STAR_RUNTIME_JOB_ID=$runtime_job" \
  slurm/submit_alg_star_runtime_aggregate.slurm | cut -d';' -f1)"

printf 'Prepare test183 job: %s\nRuntime array job: %s\nAggregate job: %s\n' \
  "$prepare_job" "$runtime_job" "$aggregate_job"
printf 'ALG* and Initial ALG are each recomputed end-to-end %s times per pair.\n' \
  "${ALG_RUNTIME_REPETITIONS:-3}"


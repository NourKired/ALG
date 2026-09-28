#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${OUTPUT_DIR:-outputs/alg_literature_synthetic_128seeds}"
AGGREGATE="${AGGREGATE:-outputs/alg_literature_synthetic_128seeds_aggregate/all_scores.csv}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-128}"
if [[ -n "${ALG_PARTITION:-}" ]]; then
  PARTITION_ARGS=(--partition="$ALG_PARTITION")
else
  PARTITION_ARGS=()
fi
mkdir -p "$OUTPUT_DIR" "$(dirname "$AGGREGATE")" logs

ARRAY_JOB_ID="$(sbatch \
  --parsable \
  "${PARTITION_ARGS[@]}" \
  --array="0-127%${ARRAY_PARALLELISM}" \
  --export="ALL,ALG_SYNTHETIC_OUTPUT_DIR=$OUTPUT_DIR" \
  slurm/submit_literature_synthetic_array.slurm | cut -d';' -f1)"

AGGREGATE_JOB_ID="$(sbatch \
  --parsable \
  "${PARTITION_ARGS[@]}" \
  --dependency="afterok:${ARRAY_JOB_ID}" \
  --output="logs/alg_synagg_%j.out" \
  --error="logs/alg_synagg_%j.err" \
  --export="ALL,ALG_SYNTHETIC_OUTPUT_DIR=$OUTPUT_DIR,ALG_SYNTHETIC_AGGREGATE=$AGGREGATE,ALG_ARRAY_JOB_ID=$ARRAY_JOB_ID" \
  slurm/submit_literature_synthetic_aggregate.slurm | cut -d';' -f1)"

echo "Literature synthetic array job: $ARRAY_JOB_ID"
echo "Literature synthetic aggregate job: $AGGREGATE_JOB_ID"

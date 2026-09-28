#!/usr/bin/env bash
set -euo pipefail

ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-3}"
COMPONENT_ROOT="${COMPONENT_ROOT:-outputs/alg_neural_external_components_168960}"
RESULT_ROOT="${RESULT_ROOT:-outputs/alg_neural_external_results_168960}"
AGGREGATE_ROOT="${AGGREGATE_ROOT:-outputs/alg_neural_external_results_168960_aggregate}"

if [[ -z "${ALG_TORCHVISION_ROOT:-}" ]]; then
  echo "ALG_TORCHVISION_ROOT must point to the existing TorchVision data directory." >&2
  exit 1
fi

if [[ -n "${ALG_PARTITION:-}" ]]; then
  partition_args=(--partition="$ALG_PARTITION")
else
  partition_args=()
fi

mkdir -p "$COMPONENT_ROOT" "$RESULT_ROOT" "$AGGREGATE_ROOT" logs

component_job="$(sbatch --parsable "${partition_args[@]}" \
  --array="0-2%${ARRAY_PARALLELISM}" \
  --output="logs/alg_neural_components_%A_%a.out" \
  --error="logs/alg_neural_components_%A_%a.err" \
  --export="ALL,ALG_NEURAL_COMPONENT_ROOT=$COMPONENT_ROOT" \
  scripts/submit_neural_representation_component.slurm | cut -d';' -f1)"

evaluation_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${component_job}" \
  --array="0-2%${ARRAY_PARALLELISM}" \
  --output="logs/alg_neural_evaluation_%A_%a.out" \
  --error="logs/alg_neural_evaluation_%A_%a.err" \
  --export="ALL,ALGFW_COMPONENT_ROOT=$COMPONENT_ROOT,ALGFW_RESULT_ROOT=$RESULT_ROOT,ALGFW_SHARD_COUNT=3" \
  scripts/submit_alg_framework_evaluation_task.slurm | cut -d';' -f1)"

aggregate_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${evaluation_job}" \
  --job-name="alg_neural_agg" \
  --cpus-per-task=2 --mem=32G --time=04:00:00 \
  --output="logs/alg_neural_aggregate_%j.out" \
  --error="logs/alg_neural_aggregate_%j.err" \
  --wrap="cd '$PWD' && python3 scripts/aggregate_neural_representation_validation.py --results-root '$RESULT_ROOT' --components-root '$COMPONENT_ROOT' --output-dir '$AGGREGATE_ROOT'" \
  | cut -d';' -f1)"

printf 'Neural component job: %s\nNeural evaluation job: %s\nNeural aggregate job: %s\n' \
  "$component_job" "$evaluation_job" "$aggregate_job"

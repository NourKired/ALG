#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SHARD_COUNT="${SHARD_COUNT:-128}"
ARRAY_PARALLELISM="${ARRAY_PARALLELISM:-128}"
GRID_SIZE="$(python3 -c 'import sys; sys.path.insert(0,"src"); from alg_similarity.alg_framework_grid import framework_grid_size; print(framework_grid_size())')"
COMPONENT_ROOT="${COMPONENT_ROOT:-outputs/alg_framework_components_${GRID_SIZE}_243}"
RESULT_ROOT="${RESULT_ROOT:-outputs/alg_framework_results_${GRID_SIZE}_243}"
AGGREGATE_ROOT="${AGGREGATE_ROOT:-outputs/alg_framework_results_${GRID_SIZE}_243_aggregate}"
SOURCE_AGGREGATE="${SOURCE_AGGREGATE:-outputs/alg_full_literature_extended_grid1320_aggregate}"
PARTITION="${PARTITION:-${AGGREGATE_ROOT}/eligible_partition_60dev_183test.csv}"
BASELINE_SUMMARY="${BASELINE_SUMMARY:-${SOURCE_AGGREGATE}/metric_summary_all_datasets.csv}"
BASELINE_EFFICIENCY="${BASELINE_EFFICIENCY:-${SOURCE_AGGREGATE}/metric_efficiency_all_datasets.csv}"

if [[ -n "${ALG_PARTITION:-}" ]]; then
  partition_args=(--partition="$ALG_PARTITION")
else
  partition_args=()
fi

mkdir -p "$COMPONENT_ROOT" "$RESULT_ROOT" "$AGGREGATE_ROOT" logs

python3 scripts/build_alg_framework_partition.py \
  --aggregate-root "$SOURCE_AGGREGATE" \
  --output "$PARTITION" \
  --expected-development 60 \
  --expected-test 183

# Freeze OpenML IDs from the already frozen dataset partition. This avoids
# querying the OpenML suite API independently in every array task.
OPENML_SPECS="$(python3 -c '
import pandas as pd, sys
datasets = pd.read_csv(sys.argv[1])["dataset"].astype(str)
specs = []
for name in datasets:
    if name.startswith("OpenML_suite_99_"):
        data_id = name.rsplit("_", 1)[-1]
        specs.append(f"{data_id}:{name[7:]}")
print(",".join(specs))
' "$PARTITION")"

manifest_args=(
  scripts/run_overlap_metric_large_scale.py
  --preset full
  --n-pairs "${ALG_N_PAIRS:-1000}"
  --cloud-size "${ALG_CLOUD_SIZE:-12}"
  --seed "${ALG_SEED:-42}"
  --evaluation-folds 5
  --shard-count "$SHARD_COUNT"
  --alg-framework-components-only
  --dataset-allowlist "$PARTITION"
  --ucr-max-datasets "${ALG_UCR_MAX_DATASETS:-0}"
  --glue-tasks "${ALG_GLUE_TASKS:-max}"
  --tudatasets "${ALG_TUDATASETS:-max}"
  --torchvision-datasets "${ALG_TORCHVISION_DATASETS:-max}"
  --torchvision-root "${ALG_TORCHVISION_ROOT:-data/torchvision}"
  --openml "$OPENML_SPECS"
  --openml-suites ""
  --list-jobs
  --job-manifest-out "$COMPONENT_ROOT/expected_datasets.csv"
)
[[ -n "${ALG_UCR_ROOT:-}" ]] && manifest_args+=(--ucr-root "$ALG_UCR_ROOT")
[[ -n "${ALG_TS_ROOTS:-}" ]] && manifest_args+=(--ts-roots "$ALG_TS_ROOTS")
python3 "${manifest_args[@]}" > "$COMPONENT_ROOT/dataset_resolution.log"

resolved="$(python3 -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$COMPONENT_ROOT/expected_datasets.csv")"
if [[ "$resolved" -ne 243 ]]; then
  echo "Expected exactly 243 ALG-eligible jobs (60+183), resolved $resolved" >&2
  exit 1
fi
python3 -c '
import pandas as pd, sys
resolved = pd.read_csv(sys.argv[1]).sort_values("job_index")["dataset"].astype(str).tolist()
partition = pd.read_csv(sys.argv[2]).sort_values("job_order")["dataset"].astype(str).tolist()
if resolved != partition:
    for index, (left, right) in enumerate(zip(resolved, partition)):
        if left != right:
            raise SystemExit(f"Deterministic job-order mismatch at {index}: {left} != {right}")
    raise SystemExit("Deterministic job-order mismatch")
print("Validated deterministic 243-dataset job order.")
' "$COMPONENT_ROOT/expected_datasets.csv" "$PARTITION"

component_job="$(sbatch --parsable "${partition_args[@]}" \
  --array="0-$((SHARD_COUNT - 1))%${ARRAY_PARALLELISM}" \
  --output="logs/algfw_components_%A_%a.out" \
  --error="logs/algfw_components_%A_%a.err" \
  --export="ALL,ALGFW_COMPONENT_ROOT=$COMPONENT_ROOT,ALGFW_SHARD_COUNT=$SHARD_COUNT,ALGFW_PARTITION=$PARTITION" \
  slurm/submit_alg_framework_components_task.slurm | cut -d';' -f1)"

evaluation_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${component_job}" \
  --array="0-$((SHARD_COUNT - 1))%${ARRAY_PARALLELISM}" \
  --output="logs/algfw_evaluation_%A_%a.out" \
  --error="logs/algfw_evaluation_%A_%a.err" \
  --export="ALL,ALGFW_COMPONENT_ROOT=$COMPONENT_ROOT,ALGFW_RESULT_ROOT=$RESULT_ROOT,ALGFW_SHARD_COUNT=$SHARD_COUNT" \
  slurm/submit_alg_framework_evaluation_task.slurm | cut -d';' -f1)"

aggregate_job="$(sbatch --parsable "${partition_args[@]}" \
  --dependency="afterok:${evaluation_job}" \
  --output="logs/algfw_aggregate_%j.out" \
  --error="logs/algfw_aggregate_%j.err" \
  --export="ALL,ALGFW_RESULT_ROOT=$RESULT_ROOT,ALGFW_PARTITION=$PARTITION,ALGFW_BASELINE_SUMMARY=$BASELINE_SUMMARY,ALGFW_BASELINE_EFFICIENCY=$BASELINE_EFFICIENCY,ALGFW_AGGREGATE_ROOT=$AGGREGATE_ROOT,ALGFW_COMPONENT_JOB_ID=$component_job,ALGFW_EVALUATION_JOB_ID=$evaluation_job" \
  slurm/submit_alg_framework_aggregate.slurm | cut -d';' -f1)"

printf 'Component array job: %s\nEvaluation array job: %s\nAggregate job: %s\n' \
  "$component_job" "$evaluation_job" "$aggregate_job"
printf 'Requested parallelism: %s; the cluster QOS remains authoritative.\n' "$ARRAY_PARALLELISM"

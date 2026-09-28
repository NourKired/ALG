#!/usr/bin/env bash
#SBATCH --job-name=algfw_comp
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

extra_args=()
[[ -n "${ALG_UCR_ROOT:-}" ]] && extra_args+=(--ucr-root "$ALG_UCR_ROOT")
[[ -n "${ALG_TS_ROOTS:-}" ]] && extra_args+=(--ts-roots "$ALG_TS_ROOTS")
openml_specs="$(python3 -c '
import pandas as pd, sys
datasets = pd.read_csv(sys.argv[1])["dataset"].astype(str)
specs = []
for name in datasets:
    if name.startswith("OpenML_suite_99_"):
        data_id = name.rsplit("_", 1)[-1]
        specs.append(f"{data_id}:{name[7:]}")
print(",".join(specs))
' "${ALGFW_PARTITION:?}")"

shard_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}"
printf -v shard_dir '%s/shard_%03d' "${ALGFW_COMPONENT_ROOT:?}" "$shard_id"

python3 scripts/run_overlap_metric_large_scale.py \
  --out-dir "$shard_dir" \
  --preset full \
  --n-pairs "${ALG_N_PAIRS:-1000}" \
  --cloud-size "${ALG_CLOUD_SIZE:-12}" \
  --seed "${ALG_SEED:-42}" \
  --evaluation-folds 5 \
  --shard-count "${ALGFW_SHARD_COUNT:-128}" \
  --shard-index "$shard_id" \
  --resume \
  --strict \
  --pair-scores-only \
  --alg-framework-components-only \
  --dataset-allowlist "${ALGFW_PARTITION:?}" \
  --ucr-max-datasets "${ALG_UCR_MAX_DATASETS:-0}" \
  --glue-tasks "${ALG_GLUE_TASKS:-max}" \
  --tudatasets "${ALG_TUDATASETS:-max}" \
  --torchvision-datasets "${ALG_TORCHVISION_DATASETS:-max}" \
  --torchvision-root "${ALG_TORCHVISION_ROOT:-data/torchvision}" \
  --torchvision-max-items "${ALG_TORCHVISION_MAX_ITEMS:-5000}" \
  --openml "$openml_specs" \
  --openml-suites "" \
  --metric-max-points "${ALG_METRIC_MAX_POINTS:-256}" \
  "${extra_args[@]}"

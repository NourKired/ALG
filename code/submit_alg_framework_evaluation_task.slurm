#!/usr/bin/env bash
#SBATCH --job-name=algfw_eval
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=72:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

python3 scripts/evaluate_alg_framework_158400.py \
  --components-root "${ALGFW_COMPONENT_ROOT:?}" \
  --output-dir "${ALGFW_RESULT_ROOT:?}" \
  --shard-count "${ALGFW_SHARD_COUNT:-128}" \
  --shard-index "${SLURM_ARRAY_TASK_ID:?}" \
  --chunk-size "${ALGFW_CHUNK_SIZE:-256}" \
  --cv-folds 5 \
  --cv-seed "${ALGFW_CV_SEED:-42}"

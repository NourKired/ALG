#!/usr/bin/env bash
#SBATCH --job-name=algstar_agg
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

python3 scripts/aggregate_alg_star_end_to_end_runtime.py \
  --input-root "${ALG_STAR_RUNTIME_ROOT:?}" \
  --output-dir "${ALG_STAR_RUNTIME_AGGREGATE:?}" \
  --expected-datasets 183

python3 scripts/collect_slurm_accounting.py \
  --job-id "${ALG_STAR_RUNTIME_JOB_ID:?}" \
  --output "${ALG_STAR_RUNTIME_AGGREGATE:?}/slurm_accounting.csv"


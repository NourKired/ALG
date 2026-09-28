#!/usr/bin/env bash
#SBATCH --job-name=alg_synagg
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 scripts/collect_slurm_accounting.py \
  --job-id "${ALG_ARRAY_JOB_ID:?ALG_ARRAY_JOB_ID is not set}" \
  --output "$(dirname "${ALG_SYNTHETIC_AGGREGATE:?ALG_SYNTHETIC_AGGREGATE is not set}")/slurm_accounting.csv"

python3 scripts/aggregate_literature_synthetic.py \
  --input-dir "${ALG_SYNTHETIC_OUTPUT_DIR:?ALG_SYNTHETIC_OUTPUT_DIR is not set}" \
  --output "${ALG_SYNTHETIC_AGGREGATE:?ALG_SYNTHETIC_AGGREGATE is not set}"

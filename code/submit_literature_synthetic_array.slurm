#!/usr/bin/env bash
#SBATCH --job-name=alg_synlit
#SBATCH --array=0-127%128
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=logs/alg_synlit_%A_%a.out
#SBATCH --error=logs/alg_synlit_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
mkdir -p "${ALG_SYNTHETIC_OUTPUT_DIR:-outputs/alg_literature_synthetic_128seeds}" logs

python3 scripts/run_literature_synthetic_protocols.py \
  --output-dir "${ALG_SYNTHETIC_OUTPUT_DIR:-outputs/alg_literature_synthetic_128seeds}" \
  --seed "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is not set}" \
  --samples "${ALG_SYNTHETIC_SAMPLES:-1000}" \
  --protocol "${ALG_SYNTHETIC_PROTOCOL:-all}" \
  --metric-max-points "${ALG_METRIC_MAX_POINTS:-256}" \
  --geometry-repeats "${ALG_GEOMETRY_REPEATS:-16}"

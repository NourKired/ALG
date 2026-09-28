#!/usr/bin/env bash
#SBATCH --job-name=algfw_agg
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

python3 scripts/aggregate_alg_framework_158400.py \
  --results-root "${ALGFW_RESULT_ROOT:?}" \
  --partition "${ALGFW_PARTITION:?}" \
  --baseline-summary "${ALGFW_BASELINE_SUMMARY:?}" \
  --baseline-efficiency "${ALGFW_BASELINE_EFFICIENCY:?}" \
  --output-dir "${ALGFW_AGGREGATE_ROOT:?}" \
  --expected-development 60 \
  --expected-test 183

python3 scripts/collect_slurm_accounting.py \
  --job-id "${ALGFW_COMPONENT_JOB_ID:?}" \
  --output "${ALGFW_AGGREGATE_ROOT}/slurm_components_accounting.csv"

python3 scripts/collect_slurm_accounting.py \
  --job-id "${ALGFW_EVALUATION_JOB_ID:?}" \
  --output "${ALGFW_AGGREGATE_ROOT}/slurm_evaluation_accounting.csv"

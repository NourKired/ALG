#!/usr/bin/env bash
#SBATCH --job-name=alg_neural_comp
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python3 scripts/generate_neural_representation_components.py \
  --torchvision-root "${ALG_TORCHVISION_ROOT:?}" \
  --output-root "${ALG_NEURAL_COMPONENT_ROOT:?}" \
  --dataset-index "${SLURM_ARRAY_TASK_ID:?}" \
  --train-samples "${ALG_NEURAL_TRAIN_SAMPLES:-5000}" \
  --probe-samples "${ALG_NEURAL_PROBE_SAMPLES:-1000}" \
  --model-seeds "${ALG_NEURAL_MODEL_SEEDS:-5}" \
  --max-iter "${ALG_NEURAL_MAX_ITER:-80}" \
  --n-pairs "${ALG_NEURAL_N_PAIRS:-80}" \
  --seed "${ALG_NEURAL_SEED:-42}"

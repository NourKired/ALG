# Adaptive Local--Global Similarity (ALG)

This is the companion repository for **Adaptive Local--Global Similarity for Comparing Heterogeneous Unordered Embedding Sets**.

ALG combines a bounded local agreement and a bounded global agreement,

```text
ALG(A,B) = lambda L(A,B) + (1-lambda) G(A,B).
```

The complete search evaluates 168,960 configurations on 60 training datasets. The selected configuration is frozen before evaluation on 183 benchmark test datasets and three additional neural-representation datasets:

```text
ALG* = 0.7 LocalScaling(Manhattan, k=3)
     + 0.3 exp(-CosineDistance(centroid(A), centroid(B))/2).
```

## Repository contents

- `src/alg_similarity/`: installable ALG implementation and metric components.
- `scripts/`: dataset preparation, evaluation, and aggregation entry points.
- `slurm/`: cluster launchers and array-job definitions.
- `tests/`: deterministic unit and invariant tests.
- `notebooks/`: executable notebook reconstructing the article tables and figures.
- `configs/`: public protocol manifests and examples.
- `results/benchmark/`: 168,960-configuration search, frozen selection, baseline results, partitions, and efficiency records.
- `results/neural/`: results on three neural-representation datasets.
- `results/controlled/`: controlled-perturbation results aggregated over 128 seeds.
- `results/runtime/`: measured end-to-end ALG* and Initial ALG timings over 155,393 pairs.
- `results/article/`: paper-level derived tables.
- `paper/`: manuscript, references, tables, and final figures.
- `data/README.md`: dataset provenance and retrieval policy.

Large CSV files are stored as `.csv.gz`; pandas reads them directly with `pd.read_csv(path)`.

## Main protocol

- 60 training datasets for configuration selection.
- 183 held-out benchmark test datasets.
- 3 additional neural-representation test datasets.
- 40 baselines.
- Stratified five-fold threshold evaluation; no classifier is trained for the metric comparison.
- Pairwise F1 uses positive label 1.
- The selected configuration is frozen before test evaluation.

## Installation and checks

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev,figures]'
make test
make lint
```

For all optional literature baselines and external dataset adapters:

```bash
python3 -m pip install -e '.[full,figures,dev]'
```

## Reproducing the figures

```bash
jupyter lab notebooks/generate_all_article_figures.ipynb
```

The notebook expects the directory structure shipped in this repository.

## Reproducing the benchmark

After configuring the public dataset roots, submit the complete 168,960-grid
campaign from the repository root:

```bash
export ALG_UCR_ROOT=/path/to/UCRArchive_2018
export ALG_TORCHVISION_ROOT=/path/to/torchvision
export ARRAY_PARALLELISM=128
bash slurm/submit_alg_framework_168960_campaign.sh
```

The scheduler and cluster quality-of-service policy determine the actual number
of concurrent jobs. Controlled protocols, neural-representation validation, and
runtime measurement have dedicated launchers in `slurm/`.

## Data availability

Raw third-party corpora are not redistributed. Dataset identifiers, partitions, derived results, and retrieval instructions are released here. The complete working-output archive is too large for GitHub and will be deposited separately; its persistent URL will be added to `data/README.md`.

## Integrity

`SHA256SUMS` contains a SHA-256 checksum for every released artifact.

## Contributing and licensing

Contribution instructions and community expectations are documented in
`CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`. Software is distributed under the
BSD 3-Clause License (`LICENSE-CODE`); original documentation and research
artifacts are distributed under CC BY-SA 4.0 (`LICENSE`). Third-party datasets
retain their original terms and are not redistributed.

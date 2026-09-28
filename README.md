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

- `code/`: metric implementations, evaluation pipeline, Slurm launchers, and aggregation scripts.
- `notebooks/`: the executable notebook that reconstructs the article tables and figures.
- `results/benchmark/`: the 168,960-configuration search, frozen selection, baseline results, partitions, and efficiency records.
- `results/neural/`: results on three neural-representation datasets.
- `results/controlled/`: aggregated controlled-perturbation results over 128 seeds.
- `results/runtime/`: real end-to-end ALG* and Initial ALG timings over 155,393 pairs.
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

## Reproducing the figures

```bash
python3 -m pip install numpy pandas scipy scikit-learn matplotlib jupyter
jupyter lab notebooks/generate_all_article_figures.ipynb
```

The notebook expects the directory structure shipped in this repository. Cluster-scale experiments can be relaunched with the scripts under `code/` after configuring the public dataset roots.

## Data availability

Raw third-party corpora are not redistributed. Dataset identifiers, partitions, derived results, and retrieval instructions are released here. The complete working-output archive is too large for GitHub and will be deposited separately; its persistent URL will be added to `data/README.md`.

## Integrity

`SHA256SUMS` contains a SHA-256 checksum for every released artifact.

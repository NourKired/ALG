#!/usr/bin/env bash
#SBATCH --job-name=algstar_prep
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd

source = Path(os.environ["ALG_STAR_SOURCE_PARTITION"])
target = Path(os.environ["ALG_STAR_TEST_PARTITION"])
table = pd.read_csv(source)
required = {"dataset", "partition"}
if not required.issubset(table.columns):
    raise SystemExit(f"Partition lacks columns: {sorted(required - set(table.columns))}")
test = table.loc[table["partition"].astype(str).eq("test")].copy()
if test["dataset"].nunique() != 183 or len(test) != 183:
    raise SystemExit(f"Expected exactly 183 unique test datasets, found {len(test)} rows")
target.parent.mkdir(parents=True, exist_ok=True)
test.to_csv(target, index=False)
print(f"Saved {len(test)} test datasets to {target}")
PY


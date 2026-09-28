#!/usr/bin/env python3
"""Fail-fast dependency check for the full ALG campaign."""
from __future__ import annotations

import importlib


REQUIRED = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "ot": "POT",
    "ripser": "ripser",
    "persim": "persim",
    "gudhi": "gudhi",
    "openml": "openml",
    "datasets": "datasets",
    "sentence_transformers": "sentence-transformers",
    "torchvision": "torchvision",
    "torch_geometric": "torch-geometric",
}


def main() -> None:
    missing = []
    for module, package in REQUIRED.items():
        try:
            importlib.import_module(module)
            print(f"OK      {module}")
        except Exception as exc:
            missing.append(package)
            print(f"MISSING {module}: {type(exc).__name__}: {exc}")
    if missing:
        print("\nInstall the full campaign dependencies with:")
        print(
            "  python3 -m pip install --user "
            "'gudhi>=3.9' 'openml>=0.14' 'datasets>=2.18' "
            "'torchvision>=0.16' 'torch-geometric>=2.5'"
        )
        raise SystemExit(f"Missing/incompatible packages: {', '.join(missing)}")
    print("All full-campaign Python dependencies are importable.")


if __name__ == "__main__":
    main()

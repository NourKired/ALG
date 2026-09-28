# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
make test
make lint
```

Keep scientific changes reproducible: state the dataset identifiers, random seeds,
evaluation protocol, and commands used. Do not commit third-party raw datasets or
credentials. New metric implementations require tests and an explicit literature
reference where applicable.

Commits should be focused and use an imperative summary. By contributing, you
agree that your code contribution is licensed under `LICENSE-CODE` and your
documentation contribution under `LICENSE`.


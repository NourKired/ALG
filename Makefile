.PHONY: install install-full test lint figures checksums

install:
	python3 -m pip install -e .

install-full:
	python3 -m pip install -e '.[full,figures,dev]'

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check src tests

figures:
	jupyter nbconvert --to notebook --execute notebooks/generate_all_article_figures.ipynb --output /tmp/alg-figures.ipynb

checksums:
	@git ls-files -co --exclude-standard | LC_ALL=C sort | while IFS= read -r file; do \
		if [ "$$file" != "SHA256SUMS" ]; then shasum -a 256 "$$file"; fi; \
	done > SHA256SUMS

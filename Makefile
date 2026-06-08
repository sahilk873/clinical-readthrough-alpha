.PHONY: install install-dev test lint typecheck clean run-pipeline

install:
	pip install -e ".[dev]"

install-dev:
	pip install -e ".[dev,polygon]"

test:
	pytest tests/ -v --tb=short --cov=clinical_alpha

test-full:
	pytest tests/ -v --tb=short --cov=clinical_alpha -m "" -k ""

lint:
	ruff check src/clinical_alpha/ tests/ scripts/

typecheck:
	mypy src/clinical_alpha/ scripts/

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

run-pipeline:
	python scripts/run_pipeline.py

fetch-all:
	python scripts/fetch_all.py

generate-report:
	python scripts/generate_report.py

sample-output:
	python scripts/generate_sample_output.py

quality:
	ruff check src/clinical_alpha/ tests/ scripts/
	mypy src/clinical_alpha/ scripts/ --ignore-missing-imports
	pytest tests/ -v --tb=short --cov=clinical_alpha --cov-report=term-missing

data-dirs:
	mkdir -p data/raw data/processed data/samples reports/tables reports/figures

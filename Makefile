.PHONY: setup preprocess train evaluate test lint

setup:
	pip install -e ".[dev]"
	python scripts/download.py

preprocess:
	python scripts/preprocess.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

test:
	pytest tests/ --cov=src --cov-report=term -v

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

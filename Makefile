PYTHON ?= python3

.PHONY: setup preprocess train evaluate test lint

setup:
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) scripts/download.py

preprocess:
	$(PYTHON) scripts/preprocess.py

train:
	$(PYTHON) scripts/train.py

evaluate:
	$(PYTHON) scripts/evaluate.py

test:
	pytest tests/ --cov=src --cov-report=term -v

lint:
	ruff check src/ tests/ scripts/
	black --check src/ tests/ scripts/

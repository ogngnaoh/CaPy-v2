PYTHON ?= python3

.PHONY: setup preprocess train evaluate test lint format coverage clean

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

format:
	ruff check src/ tests/ scripts/ --fix
	black src/ tests/ scripts/

coverage:
	pytest tests/ --cov=src --cov-report=term --cov-report=html:htmlcov -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov *.egg-info build/ dist/

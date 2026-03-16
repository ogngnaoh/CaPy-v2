FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY tests/ tests/
COPY scripts/ scripts/
COPY results/ results/
COPY Makefile .
RUN pip install --no-cache-dir -e ".[dev]"
CMD ["pytest", "tests/", "-v", "--tb=short"]

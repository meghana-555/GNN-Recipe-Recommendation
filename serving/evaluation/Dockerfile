FROM python:3.11-slim

LABEL maintainer="gnn-recipe-rec evaluation"
LABEL description="Evaluation and load-testing tools for the GNN recipe recommendation API"

# No GPU needed — CPU-only image
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy evaluation scripts
COPY locustfile.py .
COPY benchmark.py .
COPY run_evaluation.py .

# Configurable via env vars at runtime
ENV BASELINE_URL=""
ENV OPTIMIZED_URL=""
ENV FURTHER_URL=""
ENV BENCHMARK_DURATION="30"
ENV USER_POOL_SIZE="1000"
ENV OUTPUT_DIR="/results"
ENV MODEL_VERSION="v1.0"
ENV CODE_VERSION="HEAD"
ENV HARDWARE="cpu"
ENV COMPUTE_INSTANCE="chameleon-m1.medium"

# Mount point for persisting results outside the container
VOLUME ["/results"]

# Default: run the full orchestration
CMD ["python", "run_evaluation.py"]

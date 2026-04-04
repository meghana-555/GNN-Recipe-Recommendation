# -------------------------------------------------------------------
# Dockerfile for the batch pre-computation pipeline
#
# Runs as a one-shot job: trains (if needed), scores all users,
# writes recommendations to Redis / JSON and recipe metadata to disk.
# -------------------------------------------------------------------
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="gnn-recipe-recommendation"
LABEL description="Batch scoring pipeline for heterogeneous GraphSAGE recipe recommender"

# Avoid interactive prompts during package installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies needed by torch_geometric and pandas
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch_geometric and its companion libraries pinned to the
# CUDA 12.1 / PyTorch 2.1.0 wheel index for compatibility.
RUN pip install --no-cache-dir \
    torch_scatter \
    torch_sparse \
    torch_cluster \
    torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html \
    && pip install --no-cache-dir torch_geometric

# Copy application code
COPY config.py model.py batch_score.py ./

# Default environment variables (can be overridden at runtime)
ENV MODEL_PATH=/models/recipe_gnn_model.pt \
    DATA_DIR=/data \
    REDIS_HOST=localhost \
    REDIS_PORT=6379 \
    CACHE_BACKEND=memory \
    TOP_N=10 \
    BATCH_SIZE=128 \
    RECIPE_METADATA_OUTPUT=/data/recipe_metadata.json \
    RECOMMENDATIONS_OUTPUT=/data/recommendations.json

# Create expected directories
RUN mkdir -p /models /data

# Run the batch scoring pipeline as a one-shot job
CMD ["python", "batch_score.py"]

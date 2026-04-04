# ---------------------------------------------------------------------------
# CPU-only serving container for the GNN recipe recommendation API
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install OS-level deps (none beyond what slim provides, but keep a layer
# in case we need build tools later)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps — cached layer unless requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application source
COPY . .

# Create default directories that the app expects
RUN mkdir -p /data /logs

# Expose the port the API listens on
EXPOSE 8000

# Health check — uses the /health endpoint
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn.  Settings (port, workers, etc.) can be overridden via
# environment variables at runtime.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]

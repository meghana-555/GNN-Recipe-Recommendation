# Assisted by Claude
FROM python:3.11-slim

# curl is needed by the HEALTHCHECK instruction below; the slim image
# doesn't ship it by default.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first so dependency install layers are cached when
# only source files change.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 9090

# /metrics always returns 200 once the app is up — it's the cheapest
# real-work endpoint to probe and exercises the aggregator path.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:9090/metrics || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9090"]

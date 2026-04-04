# Serving Layer — GNN Recipe Recommendation

Serving infrastructure for the heterogeneous GraphSAGE recipe recommendation model. Designed to run on Chameleon cloud infrastructure inside Docker containers.

## Architecture

```
                                  +-----------------+
                                  |   Mealie App    |
                                  +--------+--------+
                                           |
                                    POST /recommend
                                           |
                                  +--------v--------+
                                  |   FastAPI API    |
                                  |  (CPU, <50ms)   |
                                  +--------+--------+
                                           |
                              cache lookup (user_id)
                                           |
                        +------------------+------------------+
                        |                                     |
               +--------v--------+               +-----------v-----------+
               |      Redis      |               | In-memory dict cache  |
               | (optimized)     |               | (baseline)            |
               +--------+--------+               +-----------------------+
                        |
                 populated by
                        |
               +--------v--------+
               |  Batch Scoring  |
               |  (GPU, weekly)  |
               +-----------------+
```

## Directory Structure

```
serving/
  api/                   # FastAPI serving endpoint
    main.py              # Application with /health and /recommend endpoints
    cache.py             # Swappable cache backend (Redis / in-memory)
    config.py            # Environment variable configuration
    Dockerfile           # CPU-only container
    requirements.txt

  batch/                 # Batch pre-computation pipeline
    batch_score.py       # Main script: train (if needed) -> score -> write cache
    model.py             # GNN model definitions (extracted from notebook)
    config.py            # Environment variable configuration
    Dockerfile           # CUDA-capable container (pytorch base)
    requirements.txt

  evaluation/            # Load testing & benchmarking
    benchmark.py         # Async httpx benchmark (p50/p95/p99, throughput, errors)
    locustfile.py        # Locust load test definition
    run_evaluation.py    # Orchestration script for full evaluation suite
    Dockerfile           # CPU-only container
    requirements.txt

  docker-compose.yml     # Full stack: Redis + batch + API + evaluation
  README.md              # This file
```

## Prerequisites

1. **Data files** — Place the following CSVs in `serving/data/`:
   - `PP_recipes.csv`
   - `PP_users.csv`
   - `interactions_train.csv`
   - `interactions_validation.csv`
   - `interactions_test.csv`
   - `RAW_recipes.csv`

2. **Docker** and **Docker Compose** installed.

## Quick Start

### 1. Prepare data directory

```bash
mkdir -p serving/data serving/models serving/logs serving/results

# Copy your CSV data files into serving/data/
cp /path/to/your/csvs/*.csv serving/data/
```

### 2. Run the batch pipeline (trains model if needed + pre-computes recommendations)

```bash
cd serving
docker compose up batch
```

This will:
- Check if a trained model exists at `models/recipe_gnn_model.pt`
- If not, train the GraphSAGE model from scratch (~10-15 minutes on GPU)
- Compute embeddings for all users and recipes
- Score every user against the full recipe catalog
- Write top-10 recommendations per user to Redis
- Write `recipe_metadata.json` to `data/`

### 3. Start the serving API

```bash
# Start Redis + API (Redis-backed, optimized)
docker compose up -d redis api

# Or start the baseline (in-memory cache)
docker compose up -d api-memory
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user:42"}'
```

### 5. Run evaluation

```bash
# Full benchmark suite (tests baseline and optimized)
docker compose up evaluation

# Or use Locust directly for interactive load testing
docker compose run evaluation \
  locust -f locustfile.py --host http://api:8000 \
  --headless -u 50 -r 10 --run-time 120s
```

## Running on Chameleon

### GPU node (for batch scoring)

```bash
docker build -t recipe-batch ./batch
docker run --gpus all \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  -e CACHE_BACKEND=redis \
  -e REDIS_HOST=<redis-host> \
  recipe-batch
```

### CPU node (for serving API)

```bash
docker build -t recipe-api ./api
docker run -d -p 8000:8000 \
  -v /path/to/data:/data \
  -e CACHE_BACKEND=redis \
  -e REDIS_HOST=<redis-host> \
  recipe-api
```

### Evaluation

```bash
docker build -t recipe-eval ./evaluation
docker run \
  -v /path/to/results:/results \
  -e BASELINE_URL=http://<baseline-host>:8001 \
  -e OPTIMIZED_URL=http://<optimized-host>:8000 \
  recipe-eval
```

## Configuration

All configuration is via environment variables:

### API (`serving/api/`)

| Variable | Default | Description |
|---|---|---|
| `CACHE_BACKEND` | `memory` | `redis` or `memory` |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `RECIPE_METADATA_PATH` | `/data/recipe_metadata.json` | Recipe metadata file |
| `RECOMMENDATIONS_PATH` | `/data/recommendations.json` | Pre-computed recs (memory cache) |
| `PORT` | `8000` | Uvicorn listen port |
| `LOG_DIR` | `/logs` | Serving audit log directory |

### Batch (`serving/batch/`)

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/models/recipe_gnn_model.pt` | Model checkpoint path |
| `DATA_DIR` | `/data` | Directory containing CSV files |
| `CACHE_BACKEND` | `redis` | `redis` or `memory` |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `TOP_N` | `10` | Recommendations per user |
| `BATCH_SIZE` | `128` | Scoring batch size |
| `HIDDEN_CHANNELS` | `64` | GNN hidden dimension |
| `TRAIN_EPOCHS` | `5` | Training epochs |

### Evaluation (`serving/evaluation/`)

| Variable | Default | Description |
|---|---|---|
| `BASELINE_URL` | — | Baseline endpoint URL |
| `OPTIMIZED_URL` | — | Optimized endpoint URL |
| `FURTHER_URL` | — | Further-optimized endpoint URL |
| `BENCHMARK_DURATION` | `30` | Seconds per concurrency level |
| `USER_POOL_SIZE` | `1000` | Number of synthetic user IDs |
| `OUTPUT_DIR` | `./results` | Output directory for results |

## API Contract

**Request:**
```json
{"user_id": "user:42"}
```

**Response:**
```json
{
  "user_id": "user:42",
  "recommendations": [
    {"recipe_id": "40893", "predicted_score": 4.8, "name": "arriba baked winter squash"},
    {"recipe_id": "85009", "predicted_score": 4.6, "name": "breakfast pizza"},
    {"recipe_id": "44394", "predicted_score": 4.4, "name": "chicken tikka masala"},
    {"recipe_id": "12345", "predicted_score": 4.2, "name": "lemon herb pasta"},
    {"recipe_id": "67890", "predicted_score": 4.0, "name": "spicy black bean tacos"}
  ],
  "served_at": "2026-04-03T08:32:00Z"
}
```

## Evaluation Results Format

Results are output as console table, CSV, and JSON matching this schema:

| Option | Endpoint URL | Model version | Code version | Hardware | p50/p95 latency | Throughput | Error rate | Concurrency tested | Compute instance type | Notes |
|--------|-------------|---------------|--------------|----------|-----------------|------------|------------|-------------------|----------------------|-------|

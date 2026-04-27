# Mealie GNN Recipe Recommender — Full System

End-to-end recipe recommendation system built on Mealie. Three role-owned components operate independently via sub-composes and stitch together via this root compose file.

## Quick start

```bash
cp .env.example .env
# edit .env with real credentials (AWS keys, Kaggle token, Mealie admin creds)
docker compose up -d
```

Brings up Postgres + Mealie + MLflow + data pipeline + serving API + monitor on a shared network (`mealie-ml-network`).

Training-heavy services (`training`, `traffic-simulator`) are gated behind Compose profiles so they don't run by default:

```bash
docker compose --profile training up training        # run a training job
docker compose --profile simulation up traffic-simulator
docker compose --profile tools run --rm rollback     # manual rollback
```

## What runs where

| Service | Host port | Stack | Purpose |
|---|---|---|---|
| postgres | 5432 | data_pipeline | Mealie backend DB |
| mealie | 9000 | data_pipeline | Mealie UI + API |
| data-pipeline | — | data_pipeline | Ingest + seed + traffic generator + per-user tag injection |
| dashboard | 8501 | data_pipeline | Streamlit recommendation dashboard (per-user) |
| mlflow | 5000 | training | Experiment tracking + model registry |
| training | — | training | Training job (profile: training) |
| feedback-capture | — | training | Pulls Mealie ratings into training data |
| inference | 8002 | training | Live FastAPI inference service |
| retrain-orchestrator | 8080 | training | Watch for triggers, promote models |
| prometheus | 9091 | training | Metrics scraping |
| grafana | 3000 | training | Dashboards |
| traffic-simulator | — | training | Synthetic traffic (profile: simulation) |
| redis | 6379 | serving | Pre-computed recs cache |
| batch | — | serving | GNN batch scorer (on-demand) |
| api | 8000 | serving | FastAPI serving (Redis-backed) |
| api-memory | 8001 | serving | FastAPI serving (in-memory) |
| monitor | 9090 | serving | Metrics + promote/rollback decisions |
| evaluation | — | serving | Load-testing runner |
| rollback | — | serving | One-shot rollback tool (profile: tools) |

## Environment file

Root `.env` (copied from `.env.example`) supplies secrets and shared config to all three stacks. Each sub-compose reads its own slice via `env_file: - .env` or explicit `${VAR}` interpolation.

The three stacks reuse some credentials under different names:
- `AWS_ACCESS_KEY_ID` / `CHAMELEON_ACCESS_KEY` / `S3_ACCESS_KEY` are all the same secret — set them to the same value.

## Running sub-stacks standalone

Each sub-compose works independently if you only need one role:

```bash
docker compose -f data_pipeline/docker-compose.yml up
docker compose -f training/docker-compose.yml --profile training up
docker compose -f serving/docker-compose.yml up -d api monitor redis
```

Standalone mode uses each sub-compose's own network name (`mealie-ml-network` via the `default:` entry we added); services from other stacks won't be reachable.

## Feedback loop (end-to-end)

1. Users interact with recipes in Mealie UI (localhost:9000) — ratings, favorites, and meal plans.
2. `data_pipeline/generator.py` simulates 5 user personas with distinct behavior profiles.
3. `data_pipeline/ingest_mealie_traffic.py` polls Mealie's Postgres for all three signal types and writes to S3.
4. `data_pipeline/batch_pipeline.py` compiles versioned training data with temporal splits (no leakage).
5. Every 12 hours, a `retrain_trigger_*.json` is automatically created on S3.
6. `training/retrain-orchestrator` detects the trigger, runs a new training job.
7. MLflow registry gets the new version, promote/rollback decision is made.
8. `data_pipeline/serve_recommendations.py` refreshes per-user `🤖 For {name}` tags in Mealie.
9. `data_pipeline/dashboard.py` auto-reloads the model (30-min TTL) and displays per-user recommendations at localhost:8501.
10. `serving/api` serves the new recs at localhost:8000.

## Rollback paths

See `serving/README.md` → "Rollback paths" for the file-swap vs. MLflow-registry rollback split.

## Directory map

```
data_pipeline/    Junhao — Mealie + Postgres + ingestion + traffic simulation + dashboard
training/         Meghana — GNN training + MLflow + monitoring
serving/          Zayed — FastAPI + Redis + decisions
docker-compose.yml   root orchestrator (this file's sibling)
.env.example      consolidated env vars
```

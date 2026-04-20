# Serving Layer

This is the online serving stack for the GNN recipe recommender: a heterogeneous GraphSAGE model trained on Mealie ratings, served behind a sub-50ms FastAPI. The key insight is that the expensive GPU work happens once a week in a batch job - the job scores every user against every recipe and writes the top-10 per user into Redis, so the online API reduces to a single cache lookup and a JSON serialization. A separate monitor service observes request telemetry and user ratings, joins them to compute Precision@10 and engagement signals, and decides whether each weekly retrained model should be promoted or rolled back. No inference happens on the hot path.

## Directory layout

```
serving/
  api/              FastAPI service (Redis-backed + in-memory variants)
  batch/            One-shot job: trains if needed, scores all users to Redis
  evaluation/       Load-testing and latency benchmark runner
  monitoring/       Metrics service, /track, /feedback forwarding, promote-decision
  rollback/         One-shot tool: swaps .pt.backup -> .pt
  sample_io/        Example /recommend request and response payloads
  docker-compose.yml
  README.md
  SAFEGUARDING.md
```

## Quick start

```bash
mkdir -p data models logs results
# put CSVs into ./data/
docker compose up -d redis
docker compose run --rm batch            # trains if needed, then scores
docker compose up -d api monitor
curl http://localhost:8000/health
curl -XPOST http://localhost:8000/recommend -H 'Content-Type: application/json' -d '{"user_id":"user:0"}'
```

The `api-memory` variant (port 8001) is the unoptimised in-memory-cache baseline used by `evaluation/` for the Redis vs. dict head-to-head.

## Feedback loop

1. User calls `POST /recommend`. The API looks up the pre-computed top-10 in Redis and returns them.
2. The API writes a line to `/logs/serving.log` (the audit trail) and fires-and-forgets a `POST http://monitor:9090/track` with `{user_id, recipe_ids, predicted_scores, latency_ms, status_code}`. The telemetry has a 500ms timeout and does not block the response.
3. When the user rates a recommended recipe in Mealie, Mealie calls `POST /feedback` on the API with `{user_id, recipe_id, rating}`. The API appends to `/logs/feedback.jsonl` and forwards the event to monitor.
4. Monitor joins `serving.log` x `feedback.jsonl` on `(user_id, recipe_id)` to compute `Precision@10` (was the rated recipe in the top-10 we served?) and `avg_rating_on_recommended`.
5. Weekly, a host cron job runs `docker compose run --rm batch`. Monitor's `/promote-decision` endpoint evaluates the new model's metrics against the thresholds below and emits a PROMOTE, ROLLBACK, or HOLD.

## Monitoring and promote/rollback thresholds

| Signal | Threshold | Action | Why |
|---|---|---|---|
| `error_rate` | >5% over last 1000 requests | ROLLBACK | Operational failure, user-visible |
| `avg_rating_on_recommended` drop | >0.2 vs 30d baseline | ROLLBACK | 0.2 stars ~= 10% of scale, user-detectable |
| `feedback_rate` drop | >20% vs prior 7d | ROLLBACK | Users disengaging |
| Precision@10 improvement + `avg_rating_on_recommended` up >0.1 over 7d | both true | PROMOTE | Conservative - both quality and engagement |
| `min_feedback_count` not met | <50 rated in window | HOLD | Insufficient sample size |

Every decision is written to `/logs/decisions.jsonl` with the metrics snapshot that drove it, so promotions and rollbacks are fully auditable.

## Reproducing batch scoring

The batch container trains if no checkpoint is found at `MODEL_PATH`, then runs inference for every user and writes top-10 into Redis. GPU is auto-detected (CUDA or ROCm); the job falls back to CPU if neither is present, which is significantly slower but functional.

- `--dry-run` runs the full pipeline but writes to a scratch key prefix so you can canary-test a new training config without clobbering the live recommendations.
- `MODEL_PATH=/models/recipe_gnn.pt` skips retraining when a checkpoint already exists at that path.
- If Redis is unreachable the job falls back to writing JSON files under `results/`, so pre-computation still completes.

## API contract

`POST /recommend` takes a `user_id` and returns the pre-computed top-10 recipes with predicted scores.

- Request: see [`sample_io/input_sample.json`](./sample_io/input_sample.json).
- Response: see [`sample_io/output_sample.json`](./sample_io/output_sample.json).

`POST /feedback` accepts a user's rating on a recommended recipe:

```json
// request
{"user_id": "user:13751", "recipe_id": "170078", "rating": 4}
// response
{"status": "ok", "logged_at": "2026-04-19T12:34:56Z"}
```

If Redis is unavailable, `/recommend` returns 404 with a clear error rather than crashing. `/feedback` persists to disk even if monitor is down; the forward to monitor is best-effort.

## Weekly retraining

Retraining is a host-level cron job, not a long-running container - we did not want a scheduler daemon in compose just to fire once a week. Example crontab line:

```
0 3 * * 0 cd /path/to/serving && docker compose run --rm batch
```

After the batch run, call `GET http://localhost:9090/promote-decision` to emit the PROMOTE, ROLLBACK, or HOLD for the fresh model. If the decision is ROLLBACK, run `docker compose --profile tools run --rm rollback` to swap `.pt.backup` back to `.pt` and bounce `api`.

## Rollback paths

There are two rollback mechanisms in this repository; which one you want depends on whether the failing model was registered in MLflow and how much audit trail you need.

- Fast local rollback — `docker compose --profile tools run --rm rollback`. Swaps `.pt.backup` → `.pt` on the shared models volume. Takes effect for `api` on the next request that reads the file; if the API caches the model in memory, a restart of the `api` service is also required. Use when: operational failure, a quick revert is needed, or MLflow is unreachable.
- Full registry rollback — `python training/rollback_manager.py --action rollback --reason "<reason>"`. Transitions the MLflow model registry Production stage back to the prior version, writes audit entries to `rollback/` in object storage, and (if configured) triggers a Kubernetes deploy rollback. Use when: a tracked model version was registered in MLflow and you want the full audit trail.
- Combined — when `MLFLOW_TRACKING_URI` is set AND the `rollback` container has `training/rollback_manager.py` mounted at `/training/`, the `rollback.sh` script runs both steps in order: the file swap first, then the MLflow stage change. The file swap is authoritative — if the MLflow step fails, the rollback is still considered applied locally and a non-fatal warning is logged.

## Right-sizing on Chameleon

- `api` container: ~200MB resident, <10% CPU under 20 concurrent users. A single `m1.medium` node handles the full Mealie population with headroom.
- `monitor` container: ~100MB, negligible CPU. Co-locates fine with `api`.
- `batch` container: requires a GPU node (we tested on AMD MI100 and NVIDIA RTX6000). Training plus scoring for 25K users completes in minutes; the node can be released immediately afterwards.
- `redis`: <50MB for 25K users at top-10 with payload ~1.5KB per user. No persistence tuning required.

See [SAFEGUARDING.md](./SAFEGUARDING.md) for the fairness, privacy, and robustness posture of this stack.

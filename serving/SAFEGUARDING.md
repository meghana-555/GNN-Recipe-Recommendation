# Safeguarding

This doc covers the responsible-deployment posture of the GNN recipe serving stack. Each subsection names a concrete concern, what we do about it today, and what is deferred.

## Fairness

Collaborative filtering amplifies popularity bias - users who rated many items drive the recipe embedding space, so long-tail recipes are systematically under-recommended. Mitigation is diversity re-ranking: cache top-50 in the batch step, then at serve time apply MMR (Maximal Marginal Relevance) on recipe tag Jaccard similarity, trimming to the shown 10 with a diversity floor of >=3 distinct cuisine tags. **Status: documented as future work.** Current serving returns raw top-10 to preserve p99 < 50ms; re-ranking would add ~5-15ms per request depending on tag-set sizes. The roadmap keeps ranking in the hot path bounded (O(50 * 50 * |tags|)) so the SLA holds.

## Explainability

Every `/recommend` response includes `predicted_score` for each of the 10 recipes - downstream UIs can surface confidence to users. The `/feedback` endpoint gives users a direct signal channel to disagree with the model; those disagreements feed the weekly retraining. No post-hoc attribution is exposed today, but the dot-product scoring is trivially linear in the user-recipe embedding, so nearest-neighbor tag rationales are feasible.

## Transparency

Model version is written to `/models/current_version.txt` on every batch run and surfaced via `GET /model-health`. The README documents that this is ML-driven (no hidden rules). All promote/rollback decisions are written to `/logs/decisions.jsonl` with justification - they are auditable after the fact.

## Privacy

`user_id` in Redis and logs is an opaque graph index (e.g. `user:13751`), never a Mealie username or email - the mapping lives in the upstream data pipeline, out of scope for serving. Redis stores only `{recipe_id, predicted_score, name}`; no PII. Feedback logs contain `rating` (integer) only, never free-text. Both `serving.log` and `feedback.jsonl` are local to the serving volume; rotation is a known follow-up to prevent unbounded growth.

## Accountability

Every decision to promote or roll back a model is written to `/logs/decisions.jsonl` with timestamp, decision, justification, and the metrics snapshot that drove it. The `current_version.txt` file records which model version is live. The batch Dockerfile is reproducible (pinned torch + PyG wheels), so a given version can be re-trained from the same CSVs if needed.

## Robustness

If Redis is unavailable, `GET /recommend` returns 404 with a clear error rather than crashing (the cache's `get` method returns `None`, which maps to 404). The batch job has a fallback to JSON file output when Redis is unreachable, so pre-computation can still complete. Automated rollback fires if `error_rate > 5%` - a separate signal from engagement, to cover ops failures distinctly from model-quality regressions. Fire-and-forget telemetry to the monitor service has a 500ms timeout and logs failures; monitor downtime never blocks user requests.

---

See [README.md](./README.md) for system architecture and the thresholds that drive these safeguards.

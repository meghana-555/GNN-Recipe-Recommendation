#!/usr/bin/env bash
# One-shot rollback: swap /models/recipe_gnn_model.pt.backup -> /models/recipe_gnn_model.pt
# Exit codes:
#   0  rollback applied
#   1  no backup present (nothing to restore)
#   2  filesystem error
set -euo pipefail

MODEL="${MODEL_PATH:-/models/recipe_gnn_model.pt}"
BACKUP="${MODEL}.backup"

if [[ ! -f "$BACKUP" ]]; then
    echo "rollback: no backup found at $BACKUP — nothing to roll back" >&2
    exit 1
fi

# Move the current model aside (to .rolled-back) so the operation is reversible
if [[ -f "$MODEL" ]]; then
    mv "$MODEL" "${MODEL}.rolled-back"
fi

cp "$BACKUP" "$MODEL"
echo "rollback: restored $MODEL from $BACKUP"

# ---------------------------------------------------------------------------
# Second stage: MLflow model-registry stage change.
#
# The file swap above is authoritative — if this stage fails for any reason,
# the rollback is still considered applied locally and the script must exit 0.
# rollback_manager.py comes from a volume mount provided by the root compose
# file; when running the sub-compose standalone it will not be present, and
# we skip this stage with a log line.
# ---------------------------------------------------------------------------

ROLLBACK_MANAGER="/training/rollback_manager.py"

mlflow_set="unset"
if [[ -n "${MLFLOW_TRACKING_URI:-}" ]]; then
    mlflow_set="set"
fi

manager_present="absent"
if [[ -f "$ROLLBACK_MANAGER" ]]; then
    manager_present="present"
fi

if [[ "$mlflow_set" == "set" && "$manager_present" == "present" ]]; then
    REASON="serving rollback script invoked at $(date -u +%FT%TZ)"
    python "$ROLLBACK_MANAGER" --action rollback --reason "$REASON" \
        || echo "rollback: MLflow stage change failed (non-fatal) — file swap still applied"
else
    echo "rollback: MLflow integration skipped (MLFLOW_TRACKING_URI=${mlflow_set}, rollback_manager=${manager_present})"
fi

exit 0

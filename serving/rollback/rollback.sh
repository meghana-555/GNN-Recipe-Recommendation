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

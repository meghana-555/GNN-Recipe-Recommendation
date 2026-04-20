#!/bin/bash
INFERENCE_URL=http://localhost:8000
REQUESTS=0
ERRORS=0
echo "Starting production traffic simulation against inference service..."
echo "Target: $INFERENCE_URL"
echo "---"
for i in $(seq 1 60); do
  USER_ID=$((RANDOM % 500 + 1))
  CODE=$(curl -s -o /dev/null -w "%{http_code}" "$INFERENCE_URL/api/recommendations?user_id=$USER_ID&top_k=5")
  REQUESTS=$((REQUESTS+1))
  if [ "$CODE" = "200" ]; then
    RECIPE_ID=$((RANDOM % 1000 + 1))
    RATING=$((RANDOM % 3 + 3))
    curl -s -X POST $INFERENCE_URL/api/feedback \
      -H 'Content-Type: application/json' \
      -d "{\"user_id\":$USER_ID,\"recipe_id\":$RECIPE_ID,\"rating\":$RATING,\"action\":\"rated\"}" > /dev/null
    echo "[OK] user=$USER_ID  rec_code=$CODE  feedback=rated $RATING stars"
  else
    ERRORS=$((ERRORS+1))
    echo "[ERR] user=$USER_ID  code=$CODE"
  fi
  sleep 1
done
echo "---"
echo "Done: $REQUESTS requests, $ERRORS errors"

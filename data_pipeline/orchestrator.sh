#!/bin/bash
echo "=================================================="
echo "🚀 GNN Recommend Pipeline Orchestrator Starting..."
echo "=================================================="

# 1. Wait for Mealie API to be healthy
echo "⏳ Waiting for Mealie Backend to boot sequence..."
until python -c "import urllib.request; urllib.request.urlopen('http://mealie-frontend:9000/api/app/about', timeout=2)" 2>/dev/null; do
  echo "Still waiting for Mealie (mealie-frontend:9000)... sleeping 5s"
  sleep 5
done
echo "✅ Mealie is online!"

# 2. Dynamic Autonomous Authentication
echo ""
echo "🔑 Handshaking with Mealie to acquire security tokens..."
export MEALIE_API_TOKEN=$(python auto_auth.py)
if [ $? -ne 0 ] || [ -z "$MEALIE_API_TOKEN" ]; then
  echo "❌ CRITICAL FAILURE: Could not acquire API Token! Check MEALIE_ADMIN_EMAIL and PASSWORD in .env!"
  exit 1
fi
echo "✅ Security Handshake Complete!"

# 3. Sequential Bootstrap Procedures (Run once on startup)
echo ""
echo "📦 [STAGE 1] Synchronizing Local Kaggle Foundation (Ingest Baseline)"
python ingest_baseline.py

echo ""
echo "🌱 [STAGE 2] Seeding Master User & Recipes into Mealie Platform"
python seed_mealie_recipes.py

echo ""
echo "🧠 [STAGE 3] Running GNN Prediction Engine & Tagging Recipes in UI"
python serve_recommendations.py

# 4. Launch Traffic Generator as background daemon
echo ""
echo "🤖 [STAGE 4] Launching Continuous Traffic Generator (Background)..."
python generator.py --duration -1 --interval 90 --users 5 &
GENERATOR_PID=$!
echo "✅ Generator running in background (PID: $GENERATOR_PID, 5 personas, interval: 90s)"

# 5. Enter Background Continuous Loop (Daemon Mode)
echo ""
echo "🎯 Orchestration Complete! System is fully primed."
echo "🔄 Entering Background Daemon Mode (Traffic Polling & Batch Generation)..."

while true; do 
  echo "-- $(date) -- Waking up Daemons --"
  
  echo "1. Polling Mealie for New User Traffic..."
  python ingest_mealie_traffic.py
  
  echo "2. Compiling Latest Daily Batch for Model Training..."
  python batch_pipeline.py
  
  echo "3. Checking if Retrain Trigger is Due (every 12 hours)..."
  python -c "
import os, json, time, boto3
from botocore.client import Config
from datetime import datetime

s3 = boto3.client('s3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    endpoint_url=os.getenv('AWS_ENDPOINT_URL','https://chi.tacc.chameleoncloud.org:7480'),
    config=Config(signature_version='s3v4'), region_name='us-east-1')
bucket = os.getenv('S3_BUCKET_NAME','ObjStore_proj14')

# Check last trigger time from marker file
marker_key = 'train/last_trigger_timestamp.txt'
try:
    obj = s3.get_object(Bucket=bucket, Key=marker_key)
    last_ts = float(obj['Body'].read().decode())
except:
    last_ts = 0

hours_since = (time.time() - last_ts) / 3600
if hours_since >= 12:
    trigger = {
        'timestamp': datetime.utcnow().isoformat(),
        'reason': 'scheduled_12h_cycle',
        'source': 'data_pipeline_orchestrator',
        'new_data_location': 'train/'
    }
    key = f'train/retrain_trigger_{datetime.utcnow().strftime(\"%Y%m%d_%H%M%S\")}.json'
    s3.put_object(Bucket=bucket, Key=key, Body=json.dumps(trigger))
    s3.put_object(Bucket=bucket, Key=marker_key, Body=str(time.time()))
    print(f'✅ Retrain trigger created: {key}')
else:
    print(f'⏳ Next trigger in {12 - hours_since:.1f} hours. Skipping.')
" 2>/dev/null || echo "⚠️ Trigger check skipped (non-critical)"
  
  echo "4. Refreshing Mealie AI Tags (Per-User Personalized)..."
  python serve_recommendations.py 2>/dev/null || echo "⚠️ Tag refresh skipped (non-critical)"
  
  echo "Daemons sleeping for 5 minutes..."
  sleep 300
done

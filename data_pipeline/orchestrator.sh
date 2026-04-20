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

# 4. Enter Background Continuous Loop (Daemon Mode)
echo ""
echo "🎯 Orchestration Complete! System is fully primed."
echo "🔄 Entering Background Daemon Mode (Traffic Polling & Batch Generation)..."

while true; do 
  echo "-- $(date) -- Waking up Daemons --"
  
  echo "1. Polling Mealie for New User Traffic..."
  python ingest_mealie_traffic.py
  
  echo "2. Compiling Latest Daily Batch for Model Training..."
  python batch_pipeline.py
  
  echo "Daemons sleeping for 5 minutes..."
  sleep 300
done

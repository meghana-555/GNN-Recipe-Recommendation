# Mealie Recipe Recommendation ML System

An end-to-end machine learning system for personalized recipe recommendations in Mealie, featuring automated training, deployment, monitoring, and safeguarding.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PRODUCTION TRAFFIC                                │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │      Istio Gateway          │                         │
│                    │    (Traffic Splitting)      │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                         │
│              ┌────────────────────┼────────────────────┐                   │
│              │                    │                    │                   │
│              ▼                    ▼                    ▼                   │
│    ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐          │
│    │    Staging      │  │   Production    │  │     Canary      │          │
│    │  (0% traffic)   │  │  (90% traffic)  │  │  (10% traffic)  │          │
│    └────────┬────────┘  └────────┬────────┘  └────────┬────────┘          │
│             │                    │                    │                    │
│             └────────────────────┼────────────────────┘                    │
│                                  │                                         │
│                    ┌─────────────▼─────────────┐                           │
│                    │     MLflow Model          │                           │
│                    │      Registry             │                           │
│                    └─────────────┬─────────────┘                           │
│                                  │                                         │
└──────────────────────────────────┼─────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼─────────────────────────────────────────┐
│                    TRAINING & FEEDBACK LOOP                                 │
│                                  │                                         │
│    ┌─────────────────┐  ┌───────▼───────┐  ┌─────────────────┐            │
│    │   Mealie App    │──│   Feedback    │──│  Object Storage │            │
│    │  (User Actions) │  │   Capture     │  │   (Training     │            │
│    └─────────────────┘  └───────────────┘  │     Data)       │            │
│                                  │          └────────┬────────┘            │
│                                  │                   │                     │
│                         ┌────────▼────────┐          │                     │
│                         │    Retrain      │◄─────────┘                     │
│                         │  Orchestrator   │                                │
│                         └────────┬────────┘                                │
│                                  │                                         │
│                         ┌────────▼────────┐                                │
│                         │   Training      │                                │
│                         │   Pipeline      │                                │
│                         └────────┬────────┘                                │
│                                  │                                         │
│                         ┌────────▼────────┐                                │
│                         │  Quality Gates  │──── Reject if AUC < 0.70       │
│                         └────────┬────────┘                                │
│                                  │ Pass                                    │
│                         ┌────────▼────────┐                                │
│                         │ Model Registry  │                                │
│                         └─────────────────┘                                │
└────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Training Pipeline (`train.py`)
- GraphSAGE-based GNN for user-recipe link prediction
- Loads training data from Chameleon Cloud object storage
- Tracks experiments in MLflow
- **Quality gates**: Models must achieve AUC ≥ 0.70 and AP ≥ 0.65 to be registered

### Inference Service (`inference_service.py`)
- FastAPI REST API serving personalized recommendations
- Loads trained `best_model.pt` and precomputes user/recipe embeddings
- Endpoints: `/api/recommendations`, `/api/feedback`, `/health`, `/metrics`
- Prometheus metrics: request count, latency, CTR, error rate
- Docker: `Dockerfile.inference`, runs on port 8000

### Feedback Capture (`feedback_capture.py`)
- Collects user interactions from Mealie (ratings, meal plans, views)
- Processes and uploads training-ready data to object storage
- Triggers retraining when sufficient new data is collected

### Production Traffic Simulation (`simulate_production_traffic.py`)
- Simulates realistic user behavior for testing
- Creates users, browses recipes, rates items, requests recommendations
- Generates interaction logs for the feedback pipeline

### Retraining Orchestrator (`retrain_orchestrator.py`)
- Monitors for retrain triggers
- Executes training with quality evaluation
- Manages model promotion: Staging → Canary → Production
- Handles automatic rollback on quality degradation

### Rollback Manager (`rollback_manager.py`)
- Health check monitoring
- Automatic rollback when error rate exceeds thresholds
- Kubernetes deployment rollback
- MLflow model version management
- Audit logging of all rollback events

## Deployment

### Local (without Docker)
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 2. Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html
pip install pandas numpy scikit-learn mlflow tqdm boto3 kaggle

# 3. Run training (auto-downloads data from Kaggle if missing)
python train.py
```

### Docker Compose (local infrastructure)
```bash
# Start MLflow + monitoring
docker-compose up -d mlflow prometheus grafana

# Run training
docker-compose --profile training up --build training

# Start traffic simulation
docker-compose --profile simulation up traffic-simulator
```

### Deploy to Chameleon Cloud

#### Automated (one command)
```powershell
# From Windows PowerShell:
.\deploy_chameleon.ps1
```

This script:
1. Creates remote directory on Chameleon VM
2. Uploads code files (`train.py`, `Dockerfile`, etc.)
3. Uploads data CSVs (~800 MB total)
4. Installs Docker on Chameleon VM (if not present)
5. Builds the Docker training image
6. Runs training in a container

#### Manual SSH deployment
```bash
# SSH into Chameleon VM
ssh -i ~/.ssh/id_rsa_chameleon cc@<CHAMELEON_IP>

# On the VM:
cd /home/cc/mealie_training

# Option A: Run with Docker
sudo docker build -t mealie-training .
sudo docker run --rm \
    -v /home/cc/mealie_training/data:/app/data \
    -v /home/cc/mealie_training/output:/app/output \
    --name mealie-training \
    mealie-training

# Option B: Run directly with Python
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html
pip install pandas numpy scikit-learn mlflow tqdm boto3 kaggle
python train.py
```

#### Post-deployment commands
```bash
# Check container status
ssh -i ~/.ssh/id_rsa_chameleon cc@<CHAMELEON_IP> "sudo docker ps"

# View training logs
ssh -i ~/.ssh/id_rsa_chameleon cc@<CHAMELEON_IP> "sudo docker logs mealie-training"

# Download trained model
scp -i ~/.ssh/id_rsa_chameleon cc@<CHAMELEON_IP>:/home/cc/mealie_training/output/best_model.pt .
```

### Kubernetes (4-person teams)
```bash
# Apply all manifests
kubectl apply -f k8s/deployment.yaml

# Check deployment status
kubectl get pods -n mealie-production
kubectl get pods -n mealie-staging

# View traffic split
kubectl get virtualservice mealie-recommender -n mealie-production -o yaml
```

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ml-pipeline.yml`) automates:

1. **Validate**: Lint and test code
2. **Build**: Create Docker images
3. **Check Data**: Determine if new training data exists
4. **Train**: Run training if new data available
5. **Deploy Staging**: Deploy to staging environment
6. **Deploy Canary**: Route 10% traffic to new model
7. **Deploy Production**: Promote to 100% traffic after canary success
8. **Rollback**: Automatic rollback on failure

## Monitoring & Alerting

### Prometheus Metrics
- `http_requests_total` - Request count by status
- `http_request_duration_seconds` - Request latency histogram
- `recommendations_served_total` - Recommendations served
- `recommendations_clicked_total` - Recommendations clicked (CTR)
- `model_prediction_latency_seconds` - Model inference latency

### Key Alerts
| Alert | Threshold | Action |
|-------|-----------|--------|
| HighErrorRate | > 5% for 5m | Auto-rollback |
| HighLatency | P99 > 2s for 5m | Warning |
| LowCTR | < 5% for 30m | Investigation |
| TrainingFailed | Job failed | Notification |

## Safeguarding

See [docs/SAFEGUARDING_PLAN.md](docs/SAFEGUARDING_PLAN.md) for complete details on:
- **Fairness**: Bias detection, exposure fairness
- **Explainability**: Feature attribution, recommendation explanations
- **Transparency**: Model cards, user communication
- **Privacy**: Data minimization, differential privacy options
- **Accountability**: Audit logging, version control
- **Robustness**: Quality gates, canary deployments, auto-rollback

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `CHAMELEON_ACCESS_KEY` | Chameleon Cloud S3 access key | For object storage |
| `CHAMELEON_SECRET_KEY` | Chameleon Cloud S3 secret key | For object storage |
| `KAGGLE_USERNAME` | Kaggle API username | For auto data download |
| `KAGGLE_KEY` | Kaggle API key | For auto data download |
| `MLFLOW_TRACKING_URI` | MLflow server URL | Yes |
| `MEALIE_URL` | Mealie service URL | For feedback capture |
| `MEALIE_ADMIN_TOKEN` | Mealie admin API token | For feedback capture |
| `PROMETHEUS_URL` | Prometheus server URL | For monitoring |

## Quick Start

```bash
# 1. Set environment variables
export CHAMELEON_ACCESS_KEY=your_access_key
export CHAMELEON_SECRET_KEY=your_secret_key
export MLFLOW_TRACKING_URI=http://localhost:5000

# 2. Start infrastructure
docker-compose up -d mlflow prometheus grafana

# 3. Run training
python train.py

# 4. Start feedback capture (background)
python feedback_capture.py --continuous &

# 5. Start retrain orchestrator (background)
python retrain_orchestrator.py --mode watch &

# 6. Simulate traffic (optional, for testing)
python simulate_production_traffic.py --duration 3600
```

## Repository Structure

```
mealie_training/
├── train.py                    # Main training script with quality gates
├── inference_service.py        # FastAPI recommendation serving API
├── feedback_capture.py         # Feedback collection pipeline
├── retrain_orchestrator.py     # Automated retraining orchestration
├── rollback_manager.py         # Model rollback management
├── simulate_production_traffic.py  # Traffic simulation for testing
├── deploy_chameleon.ps1        # One-command deploy to Chameleon Cloud
├── Dockerfile                  # Training container
├── Dockerfile.inference        # Inference service container
├── docker-compose.yml          # Local infrastructure setup
├── k8s/
│   └── deployment.yaml         # Kubernetes manifests
├── monitoring/
│   ├── prometheus.yml          # Prometheus configuration
│   └── alerts.yml              # Alerting rules
├── docs/
│   └── SAFEGUARDING_PLAN.md    # Safeguarding documentation
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml     # CI/CD pipeline
└── data/                       # Static training data
    ├── PP_recipes.csv
    ├── PP_users.csv
    ├── interactions_validation.csv
    └── interactions_test.csv
```

## Team Responsibilities

| Component | Owner |
|-----------|-------|
| Training pipeline & quality gates | Training role |
| Data collection & feature engineering | Data role |
| Inference service & deployment | Serving role |
| Monitoring & observability | All (joint) |
| Safeguarding implementation | All (joint) |

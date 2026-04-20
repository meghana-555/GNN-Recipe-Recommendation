# Deploy Mealie Training to Chameleon Cloud VM
# Usage: .\deploy_chameleon.ps1

$SSH_KEY = "$env:USERPROFILE\.ssh\id_rsa_chameleon"
$REMOTE_USER = "cc"
$REMOTE_HOST = "192.5.87.101"
$REMOTE_DIR = "/home/cc/mealie_training"
$SSH_CMD = "ssh -i $SSH_KEY -o StrictHostKeyChecking=no $REMOTE_USER@$REMOTE_HOST"
$SCP_CMD = "scp -i $SSH_KEY -o StrictHostKeyChecking=no"

Write-Host "=== Deploying Mealie Training to Chameleon Cloud ===" -ForegroundColor Cyan
Write-Host "Target: $REMOTE_USER@$REMOTE_HOST" -ForegroundColor Yellow

# Step 1: Create remote directory
Write-Host "`n[1/5] Creating remote directory..." -ForegroundColor Green
Invoke-Expression "$SSH_CMD 'mkdir -p $REMOTE_DIR/data $REMOTE_DIR/monitoring/grafana/dashboards $REMOTE_DIR/monitoring/grafana/provisioning'"

# Step 2: Copy project files (exclude venv, large cached files)
Write-Host "`n[2/5] Copying project files to Chameleon VM..." -ForegroundColor Green

# Copy essential files
$files = @(
    "train.py",
    "inference_service.py",
    "Dockerfile",
    "Dockerfile.inference",
    "docker-compose.yml",
    "requirements.txt",
    ".env.example"
)

foreach ($file in $files) {
    if (Test-Path $file) {
        Write-Host "  Copying $file..."
        Invoke-Expression "$SCP_CMD `"$file`" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
    }
}

# Copy data files
Write-Host "`n[3/5] Copying data files..." -ForegroundColor Green
$dataFiles = Get-ChildItem -Path ".\data" -Filter "*.csv" -ErrorAction SilentlyContinue
if ($dataFiles) {
    foreach ($df in $dataFiles) {
        Write-Host "  Copying data/$($df.Name) ($([math]::Round($df.Length / 1MB, 1)) MB)..."
        Invoke-Expression "$SCP_CMD `"data/$($df.Name)`" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/data/"
    }
} else {
    Write-Host "  No data files found locally - will download on remote via Kaggle" -ForegroundColor Yellow
}

# Copy monitoring configs
Write-Host "  Copying monitoring configs..."
$monFiles = @(
    "monitoring/prometheus.yml",
    "monitoring/alerts.yml",
    "monitoring/grafana/dashboards/recommendations.json",
    "monitoring/grafana/dashboards/training.json",
    "monitoring/grafana/provisioning/dashboards.yml",
    "monitoring/grafana/provisioning/datasources.yml"
)
foreach ($mf in $monFiles) {
    if (Test-Path $mf) {
        $remotePath = "$REMOTE_DIR/$($mf -replace '\\','/')"
        Invoke-Expression "$SCP_CMD `"$mf`" ${REMOTE_USER}@${REMOTE_HOST}:${remotePath}"
    }
}

# Copy trained model if available
if (Test-Path "best_model.pt") {
    Write-Host "  Copying best_model.pt..."
    Invoke-Expression "$SCP_CMD `"best_model.pt`" ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"
}

# Step 4: Setup and build on remote
Write-Host "`n[4/5] Setting up Docker on Chameleon VM..." -ForegroundColor Green
$setupCmd = "cd /home/cc/mealie_training && (command -v docker > /dev/null 2>&1 || (sudo apt-get update && sudo apt-get install -y docker.io docker-compose-plugin && sudo usermod -aG docker cc && sudo systemctl enable --now docker)) && echo 'Building Docker images...' && sudo docker build -t mealie-training . && sudo docker build -t mealie-inference -f Dockerfile.inference ."
Invoke-Expression "$SSH_CMD '$setupCmd'"

# Step 5: Run training
Write-Host "`n[5/5] Starting training on Chameleon VM..." -ForegroundColor Green
$runCmd = "cd /home/cc/mealie_training && echo 'Starting training container...' && sudo docker run --rm -v /home/cc/mealie_training/data:/app/data -v /home/cc/mealie_training/output:/app/output --name mealie-training mealie-training"
Invoke-Expression "$SSH_CMD '$runCmd'"

Write-Host "`n=== Deployment complete ===" -ForegroundColor Cyan
Write-Host "To check status:  $SSH_CMD 'sudo docker ps'" -ForegroundColor Yellow
Write-Host "To view logs:     $SSH_CMD 'sudo docker logs mealie-training'" -ForegroundColor Yellow
Write-Host "To get model:     $SCP_CMD ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/output/best_model.pt ." -ForegroundColor Yellow

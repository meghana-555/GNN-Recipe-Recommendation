FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch first
RUN pip install --no-cache-dir torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu

# Install torch-scatter and torch-sparse
RUN pip install --no-cache-dir \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.17 \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html

# Install remaining dependencies
RUN pip install --no-cache-dir \
    torch-geometric==2.3.1 \
    pandas==1.5.3 \
    numpy==1.24.3 \
    scikit-learn==1.2.2 \
    mlflow==2.3.1 \
    tqdm \
    boto3

# Copy training script
COPY train.py .

# Copy data folder (static files: PP_recipes, PP_users, val/test interactions)
COPY data/ ./data/

# Run training
CMD ["python", "train.py"]
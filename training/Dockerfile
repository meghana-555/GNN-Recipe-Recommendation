FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CPU)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install PyG and pyg-lib
RUN pip install --no-cache-dir torch-geometric \
    && pip install --no-cache-dir pyg-lib -f https://data.pyg.org/whl/torch-$(python -c "import torch; print(torch.__version__.split('+')[0])")+cpu.html

# Install remaining dependencies
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    mlflow \
    tqdm \
    boto3 \
    kaggle

# Copy training script
COPY train.py .

# Copy data folder if present
COPY data/ ./data/

# Run training
CMD ["python", "train.py"]
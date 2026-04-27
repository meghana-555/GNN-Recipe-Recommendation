import pandas as pd
import numpy as np
import os
import boto3
import time
import zipfile
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
import sys
import botocore

# Try to import pydantic for evaluation, fallback to manual if not available
try:
    from pydantic import BaseModel, Field, ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# --- Configuration (Pulled strictly from ENV for security) ---
ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

# Pydantic Schema for evaluation
if HAS_PYDANTIC:
    class InteractionSchema(BaseModel):
        user_id: int
        recipe_id: int
        date: str
        rating: float = Field(ge=0.0, le=5.0)

def check_s3_already_exists():
    print("--- 0. Checking S3 for Existing Baseline ---")
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=ENDPOINT_URL, config=Config(signature_version='s3v4'), region_name='us-east-1')
    try:
        s3.head_object(Bucket=BUCKET_NAME, Key="dataset/historical_baseline/RAW_interactions.csv")
        s3.head_object(Bucket=BUCKET_NAME, Key="dataset/historical_baseline/PP_recipes.csv")
        print("✅ Full Baseline already exists on S3.")
        
        # Critical: verify local files also exist. If not, pull from S3.
        local_files = [
            "RAW_interactions.csv", "RAW_recipes.csv",
            "PP_recipes.csv", "PP_users.csv",
            "interactions_train.csv", "interactions_validation.csv", "interactions_test.csv"
        ]
        os.makedirs("local_data", exist_ok=True)
        missing = [f for f in local_files if not os.path.exists(f"local_data/{f}")]
        
        if missing:
            print(f"⚠️ Local files missing ({len(missing)} files). Downloading from S3...")
            for f in missing:
                s3_key = f"dataset/historical_baseline/{f}"
                local_path = f"local_data/{f}"
                try:
                    s3.download_file(BUCKET_NAME, s3_key, local_path)
                    print(f"  ✅ Downloaded {f}")
                except Exception as e:
                    print(f"  ⚠️ Could not download {f}: {e}")
            print("✅ Local files synced from S3.")
        else:
            print("✅ Local files also present. Skipping ingestion.")
        
        sys.exit(0)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            print("Baseline artifacts missing on S3. Proceeding with robust ingestion...")
        else:
            raise e

def download_from_url():
    print("--- 1. Downloading Dataset ---")
    dataset_url = os.getenv("DATASET_URL", "https://www.kaggle.com/api/v1/datasets/download/shuyangli94/food-com-recipes-and-user-interactions")
    dest_path = "local_data/"
    os.makedirs(dest_path, exist_ok=True)
    
    csv_file = f"{dest_path}/RAW_interactions.csv"
    recipes_file = f"{dest_path}/RAW_recipes.csv"
    zip_path = f"{dest_path}/dataset.zip"
    if os.path.exists(csv_file) and os.path.exists(recipes_file):
        print("Data already exists locally.")
        return dest_path

    import urllib.request
    print(f"Downloading from {dataset_url} ... (This may fail if Kaggle blocks unauthenticated API calls)")
    try:
        urllib.request.urlretrieve(dataset_url, zip_path)
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        return dest_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download the dataset zip and extract RAW_interactions.csv and RAW_recipes.csv to local_data/")
        return None

def evaluate_and_format(dest_path):
    print("--- 2. Evaluating and Formatting Data Schema ---")
    
    # --- Process Interactions ---
    file_path = f"{dest_path}/RAW_interactions.csv"
    df = pd.read_csv(file_path)
    
    # Validation 1: At Ingestion. Ensure required columns exist
    required_cols = ['user_id', 'recipe_id', 'rating', 'date']
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"
    
    # Validation 2: Ensure Ratings are valid
    assert df['rating'].between(0, 5).all(), "Found ratings outside 0-5 range"
    assert not df[required_cols].isnull().any().any(), "Found NULL values in critical columns"
    
    # Expansion Logic: If < 5GB, expand following best practices for synthetic data
    # Food.com dataset is ~500MB raw. Course requires expansion if < 5GB.
    # Strategy: Generate plausible NEW interactions (not duplicate existing ones)
    #   1. Sample users and recipes proportional to their activity/popularity
    #   2. Generate new (user, recipe) pairs that DON'T already exist
    #   3. Assign ratings with Gaussian noise around user/recipe means
    #   4. Assign dates within the original date range (no future leakage)
    original_len = len(df)
    if original_len < 5_000_000:
        print("Dataset is small. Expanding with synthetic data generation...")
        target_rows = int(original_len * 0.5)  # Add 50% more rows
        
        # Pre-compute statistics for realistic generation
        user_mean_rating = df.groupby('user_id')['rating'].mean().to_dict()
        recipe_mean_rating = df.groupby('recipe_id')['rating'].mean().to_dict()
        user_counts = df['user_id'].value_counts()
        recipe_counts = df['recipe_id'].value_counts()
        
        # Unique users and recipes with popularity weights
        users = user_counts.index.tolist()
        user_weights = (user_counts.values / user_counts.values.sum()).tolist()
        recipes = recipe_counts.index.tolist()
        recipe_weights = (recipe_counts.values / recipe_counts.values.sum()).tolist()
        
        # Build set of existing (user, recipe) pairs to avoid duplicates
        existing_pairs = set(zip(df['user_id'], df['recipe_id']))
        
        # Date range for synthetic entries (stay within original range, no future leak)
        try:
            dates = pd.to_datetime(df['date'])
            date_min, date_max = dates.min(), dates.max()
        except Exception:
            date_min = pd.Timestamp('2008-01-01')
            date_max = pd.Timestamp('2018-12-31')
        date_range_days = (date_max - date_min).days
        
        # Generate synthetic interactions
        synth_rows = []
        attempts = 0
        max_attempts = target_rows * 3  # Prevent infinite loop
        
        while len(synth_rows) < target_rows and attempts < max_attempts:
            attempts += 1
            # Sample user and recipe by popularity (realistic distribution)
            u = np.random.choice(users, p=user_weights)
            r = np.random.choice(recipes, p=recipe_weights)
            
            # Skip if this (user, recipe) pair already exists
            if (u, r) in existing_pairs:
                continue
            existing_pairs.add((u, r))
            
            # Generate rating: blend user and recipe means + Gaussian noise
            u_mean = user_mean_rating.get(u, 3.5)
            r_mean = recipe_mean_rating.get(r, 3.5)
            blended = 0.5 * u_mean + 0.5 * r_mean
            rating = np.clip(round(blended + np.random.normal(0, 0.8)), 0, 5)
            
            # Generate date: uniform within original range (no future leakage)
            random_day = np.random.randint(0, max(date_range_days, 1))
            synth_date = (date_min + pd.Timedelta(days=random_day)).strftime('%Y-%m-%d')
            
            synth_rows.append({
                'user_id': u, 'recipe_id': r,
                'rating': float(rating), 'date': synth_date,
                'review': ''  # Empty review for synthetic rows
            })
        
        synth_df = pd.DataFrame(synth_rows)
        df = pd.concat([df, synth_df], ignore_index=True)
        print(f"Expanded from {original_len} to {len(df)} rows "
              f"({len(synth_rows)} synthetic interactions generated).")

    # Mealie Alignment: Convert Kaggle IDs into Mealie String prefixed IDs
    df['user_id'] = 'user:' + df['user_id'].astype(str)
    df['recipe_id'] = 'recipe:' + df['recipe_id'].astype(str)

    output_path_interactions = f"{dest_path}/processed_baseline_interactions.csv"
    df.to_csv(output_path_interactions, index=False)
    
    # --- Process Recipes ---
    recipes_path = f"{dest_path}/RAW_recipes.csv"
    if os.path.exists(recipes_path):
        print("Processing Recipes features...")
        df_rec = pd.read_csv(recipes_path)
        df_rec['id'] = 'recipe:' + df_rec['id'].astype(str)
        if 'contributor_id' in df_rec.columns:
            df_rec['contributor_id'] = 'user:' + df_rec['contributor_id'].astype(str)
        output_path_recipes = f"{dest_path}/processed_baseline_recipes.csv"
        df_rec.to_csv(output_path_recipes, index=False)
    else:
        print("Warning: RAW_recipes.csv not found, skipping recipe metadata processing.")
        output_path_recipes = None

    return output_path_interactions, output_path_recipes

def upload_to_chameleon(interactions_path, recipes_path):
    print(f"--- 3. Uploading to Chameleon S3: {BUCKET_NAME} ---")
    transfer_config = TransferConfig(multipart_threshold=1024*25, max_concurrency=10, multipart_chunksize=1024*25, use_threads=True)
    
    s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=ENDPOINT_URL, config=Config(signature_version='s3v4'), region_name='us-east-1')
    
    # Store explicitly as CSV
    s3_key_interactions = "dataset/historical_baseline/RAW_interactions.csv"
    print(f"Uploading {interactions_path} to {s3_key_interactions}")
    
    try:
        s3.upload_file(interactions_path, BUCKET_NAME, s3_key_interactions, Config=transfer_config)
        if recipes_path and os.path.exists(recipes_path):
            s3_key_recipes = "dataset/historical_baseline/RAW_recipes.csv"
            print(f"Uploading {recipes_path} to {s3_key_recipes}")
            s3.upload_file(recipes_path, BUCKET_NAME, s3_key_recipes, Config=transfer_config)
            
        print("Uploading secondary pre-processed Kaggle artifacts...")
        extra_files = ["PP_users.csv", "PP_recipes.csv", "interactions_train.csv", "interactions_validation.csv", "interactions_test.csv"]
        for f in extra_files:
            f_path = f"local_data/{f}"
            if os.path.exists(f_path):
                key = f"dataset/historical_baseline/{f}"
                print(f"Uploading {f_path} to {key}")
                s3.upload_file(f_path, BUCKET_NAME, key, Config=transfer_config)
                
        print("✅ Baseline Ingestion COMPLETE.")
    except Exception as e:
        print(f"❌ Failed to upload: {e}")

if __name__ == "__main__":
    check_s3_already_exists()
    dest_path = download_from_url()
    if dest_path:
        int_path, rec_path = evaluate_and_format(dest_path)
        upload_to_chameleon(int_path, rec_path)


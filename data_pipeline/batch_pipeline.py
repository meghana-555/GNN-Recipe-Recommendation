import pandas as pd
import boto3
import io
import os
import json
import time
from botocore.client import Config

# --- Configuration ---
ACCESS_KEY = "af8beb8221454104b30fa21e3fad8a4c"
SECRET_KEY = "fa72afa48dd941cba3dde168382eabc8"
BUCKET_NAME = "ObjStore_proj14"
ENDPOINT_URL = "https://chi.tacc.chameleoncloud.org:7480"
VERSION = time.strftime("%Y%m%d_%H%M")

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def run_batch_pipeline():
    s3 = get_s3_client()
    print(f"--- [Mealie Batch Pipeline] Starting Version: {VERSION} ---")

    # 1. Fetch Historical Data (Searching deep in dataset/VERSION/...)
    print("Searching for historical data in dataset/ subfolders...")
    hist_df = None
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="dataset/"):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('RAW_interactions.csv'):
                print(f" Found Historical Data: {obj['Key']}")
                hist_obj = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                hist_df = pd.read_csv(io.BytesIO(hist_obj['Body'].read()))
                break
    
    if hist_df is None:
        print("Error: Could not find RAW_interactions.csv in any dataset/ subfolder.")
        return

    hist_df['timestamp'] = pd.to_datetime(hist_df['date'])

    # 2. Fetch Simulation Data (Searching deep in simulation/VERSION/...)
    print("\nSearching for simulation batches in simulation/ subfolders...")
    sim_records = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="simulation/"):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                print(f" Found Batch: {obj['Key']}")
                sim_obj = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                batch_data = json.load(io.BytesIO(sim_obj['Body'].read()))
                sim_records.extend(batch_data)

    # Merge Logic
    if sim_records:
        sim_df = pd.DataFrame(sim_records)
        sim_df['timestamp'] = pd.to_datetime(sim_df['timestamp'])
        # Standardize columns for merging
        full_df = pd.concat([hist_df, sim_df], ignore_index=True)
        print(f"\nSuccessfully merged {len(sim_df)} new production rows with history.")
    else:
        full_df = hist_df
        print("\nNo simulation data found. Proceeding with historical data only.")

    # 3. Temporal Split (80/20) - Ensuring no data leakage
    print("Sorting and splitting data by time...")
    full_df = full_df.sort_values(by='timestamp')
    split_idx = int(len(full_df) * 0.8)
    train_df = full_df.iloc[:split_idx]
    eval_df = full_df.iloc[split_idx:]

    # 4. Upload to NEW versioned folders: train/VERSION/ and evaluation/VERSION/
    def upload_to_s3(df, folder, filename):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        target_key = f"{folder}/{VERSION}/{filename}"
        s3.put_object(Bucket=BUCKET_NAME, Key=target_key, Body=csv_buffer.getvalue())
        print(f"✅ Created artifact: {target_key}")

    print(f"\nSaving results for training team...")
    upload_to_s3(train_df, "train", f"train_{VERSION}.csv")
    upload_to_s3(eval_df, "evaluation", f"eval_{VERSION}.csv")

    print(f"\n--- Batch Pipeline Successfully Completed (v{VERSION}) ---")

if __name__ == "__main__":
    run_batch_pipeline()
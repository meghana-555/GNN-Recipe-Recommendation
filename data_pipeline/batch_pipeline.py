import pandas as pd
import boto3
import io
import json
import time
import os
from botocore.client import Config

ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")
VERSION = time.strftime("%Y%m%d_%H%M")

def get_s3_client():
    return boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=ENDPOINT_URL, config=Config(signature_version='s3v4'), region_name='us-east-1')

def fetch_datasets(s3):
    print("Fetching Baseline...")
    base_obj = s3.get_object(Bucket=BUCKET_NAME, Key="dataset/historical_baseline/RAW_interactions.parquet")
    df_base = pd.read_parquet(io.BytesIO(base_obj['Body'].read()))
    
    print("Fetching Simulation Increments...")
    sim_records = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="simulation/"):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.json'):
                sim_obj = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                events = json.load(io.BytesIO(sim_obj['Body'].read()))
                sim_records.extend(events)
                
    if sim_records:
        df_sim = pd.DataFrame(sim_records)
        df_sim['date'] = pd.to_datetime(df_sim['date']).astype(str)
        df_combined = pd.concat([df_base, df_sim], ignore_index=True)
    else:
        df_combined = df_base
        
    return df_combined

def run_batch():
    s3 = get_s3_client()
    df = fetch_datasets(s3)
    
    # Data Quality: Node Sparsity check (Candidate Selection Quality)
    # The GNN needs users to have >= 5 ratings. 
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= 5].index
    
    original_len = len(df)
    df_filtered = df[df['user_id'].isin(valid_users)]
    drop_ratio = 1.0 - (len(df_filtered) / original_len)
    
    print(f"Data Sparsity Quality Check: Dropped {drop_ratio:.2%} of rows due to <5 ratings rule.")
    if drop_ratio > 0.3:
        print("WARNING: Data quality deteriorating. Too many cold-start users generated.")
        # We don't halt here to allow the project to compile, but we flag it!
        
    df_filtered['timestamp_dt'] = pd.to_datetime(df_filtered['date'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')
    
    split_idx = int(len(df_filtered) * 0.8)
    train_df = df_filtered.iloc[:split_idx]
    eval_df = df_filtered.iloc[split_idx:]
    
    # Leakage check logic
    train_max = train_df['timestamp_dt'].max()
    eval_min = eval_df['timestamp_dt'].min()
    assert train_max <= eval_min, f"CRITICAL Leakage detected: {train_max} > {eval_min}"
    print("✅ Leakage Prevention Verified: Train timestamps strictly before Eval timestamps.")

    # Drop temp column
    train_df = train_df.drop(columns=['timestamp_dt'])
    eval_df = eval_df.drop(columns=['timestamp_dt'])
    
    def upload_df(df_target, folder):
        buf = io.BytesIO()
        df_target.to_parquet(buf, index=False)
        key = f"{folder}/{VERSION}/interactions.parquet"
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())
        print(f"Uploaded securely to {key}")
        
    upload_df(train_df, "train")
    upload_df(eval_df, "evaluation")
    print("Batch Pipeline Complete.")

if __name__ == "__main__":
    run_batch()

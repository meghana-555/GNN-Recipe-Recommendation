import pandas as pd
import boto3
import io
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

def load_csv_from_s3(s3, key):
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def upload_csv_to_s3(s3, df, key):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())
    print(f"Uploaded securely to {key}")

def build_data_layer():
    s3 = get_s3_client()
    
    # 1. Fetch historical purely Integer-mapped baseline limits
    print("Fetching Pre-Processed Component Baselines from Local Mount...")
    df_base_interactions = pd.read_csv("local_data/interactions_train.csv")
    df_pp_users = pd.read_csv("local_data/PP_users.csv")
    df_pp_recipes = pd.read_csv("local_data/PP_recipes.csv")

    # Pure execution of dynamic bound fetch based on Kaggle CSV layout format constraints
    base_max_u = int(df_pp_users.iloc[-1]['u'])
    base_max_i = int(df_pp_recipes.iloc[-1]['i'])

    # 2. Extract incremental traffic
    print("Fetching Production Increments...")
    sim_records = []
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="production_traffic/"):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.parquet'):
                sim_obj = s3.get_object(Bucket=BUCKET_NAME, Key=obj['Key'])
                batch_df = pd.read_parquet(io.BytesIO(sim_obj['Body'].read()))
                
                # Check for successfully normalized u & i fields inside new batch
                if not batch_df.empty and 'u' in batch_df.columns:
                    clean_batch = batch_df[['u', 'i', 'rating', 'date']].copy()
                    sim_records.append(clean_batch)
                
    if sim_records:
        df_sim = pd.concat(sim_records, ignore_index=True)
        # Typecasting arrays
        df_sim['u'] = pd.to_numeric(df_sim['u'], errors='coerce')
        df_sim['i'] = pd.to_numeric(df_sim['i'], errors='coerce')
        df_sim = df_sim.dropna()
        df_sim['u'] = df_sim['u'].astype(int)
        df_sim['i'] = df_sim['i'].astype(int)
        df_combined = pd.concat([df_base_interactions, df_sim], ignore_index=True)
    else:
        df_combined = df_base_interactions

    print(f"Combined Interactions graph edges: {len(df_combined)} rows.")
    
    # 3. Dynamic Node Padding for exact Kaggle Graph replication constraints
    if not df_combined.empty:
        new_max_u = int(df_combined['u'].max())
        new_max_i = int(df_combined['i'].max())
        
        # Format padding for users missing features
        if new_max_u > base_max_u:
            print(f"Padding users array: Node [{base_max_u + 1}] to [{new_max_u}]...")
            target_str_zeroes_58 = str([0]*58)
            new_users_df = pd.DataFrame({
                'u': range(base_max_u + 1, new_max_u + 1),
                'techniques': [target_str_zeroes_58] * (new_max_u - base_max_u),
                'items': ["[]"] * (new_max_u - base_max_u)
            })
            df_pp_users = pd.concat([df_pp_users, new_users_df], ignore_index=True)

        if new_max_i > base_max_i:
            print(f"Padding recipes array: Node [{base_max_i + 1}] to [{new_max_i}]...")
            target_str_zeroes_58 = str([0]*58)
            new_recipes_df = pd.DataFrame({
                'i': range(base_max_i + 1, new_max_i + 1),
                'techniques': [target_str_zeroes_58] * (new_max_i - base_max_i),
                'calorie_level': [1] * (new_max_i - base_max_i), 
                'ingredient_ids': ["[]"] * (new_max_i - base_max_i)
                # Left empty or NaN for non-essential ML metrics per requested formatting
            })
            df_pp_recipes = pd.concat([df_pp_recipes, new_recipes_df], ignore_index=True)

    # 4. Temporal Split logic
    df_combined['timestamp_dt'] = pd.to_datetime(df_combined['date'], errors='coerce')
    df_combined = df_combined.dropna(subset=['timestamp_dt']).sort_values('timestamp_dt')
    split_idx = int(len(df_combined) * 0.8)
    
    train_df = df_combined.iloc[:split_idx].drop(columns=['timestamp_dt'])
    val_df = df_combined.iloc[split_idx:].drop(columns=['timestamp_dt'])
    
    # 5. Output natively structured .CSV streams strictly over the designated versioned bucket format.
    print(f"Writing natively-mapped GNN-ready datasets to train/{VERSION}/ ...")
    upload_csv_to_s3(s3, train_df, f"train/{VERSION}/interactions_train.csv")
    upload_csv_to_s3(s3, val_df, f"train/{VERSION}/interactions_validation.csv")
    upload_csv_to_s3(s3, df_pp_users, f"train/{VERSION}/PP_users.csv")
    upload_csv_to_s3(s3, df_pp_recipes, f"train/{VERSION}/PP_recipes.csv")
    
    print("✅ GNN Seamless Integration Pipeline Output Complete!")

if __name__ == "__main__":
    build_data_layer()

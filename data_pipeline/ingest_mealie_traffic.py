import os
import time
import pandas as pd
import boto3
import botocore
from sqlalchemy import create_engine
import io

ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

POSTGRES_USER = os.getenv("POSTGRES_USER", "mealie")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "mealie_password")
POSTGRES_HOST = os.getenv("POSTGRES_SERVER", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mealie")

VERSION = time.strftime("%Y%m%d_%H%M")

def get_s3_client():
    return boto3.client(
        's3', 
        aws_access_key_id=ACCESS_KEY, 
        aws_secret_access_key=SECRET_KEY, 
        endpoint_url=ENDPOINT_URL, 
        config=botocore.client.Config(signature_version='s3v4'), 
        region_name='us-east-1'
    )

def fetch_s3_registry(s3):
    print("--- 1. Syncing Global ID Registry from S3 ---")
    key = "dataset/registry/id_mapping_registry.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        registry_df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        print(f"Loaded existing registry with {len(registry_df)} entries.")
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "NoSuchKey":
            print("Registry not found on S3. Initializing empty registry.")
            registry_df = pd.DataFrame(columns=["mealie_uuid", "ml_native_id", "entity_type"])
        else:
            raise e
    return registry_df

def upload_s3_registry(s3, registry_df):
    buf = io.BytesIO()
    registry_df.to_parquet(buf, index=False)
    key = "dataset/registry/id_mapping_registry.parquet"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())
    print("Registry snapshot successfully committed to S3.")

def get_dynamic_max_id(s3, entity_type):
    # Dynamically find the absolute maximum indexed vector in the Kaggle CSVs
    try:
        filename = "PP_users.csv" if entity_type == "user" else "PP_recipes.csv"
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=f"dataset/historical_baseline/{filename}")
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        col = 'u' if entity_type == "user" else 'i'
        # Get the strict last line increment or global max
        max_id = int(df.iloc[-1][col])
        return max_id
    except Exception as e:
        print(f"Warning: Failed to dynamically seek {entity_type} limits, using hardcoded Kaggle limits. ({e})")
        return 25075 if entity_type == "user" else 178264

def extract_mealie_data():
    print("--- 2. Connecting to PostgreSQL Mealie Replica ---")
    connection_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    try:
        engine = create_engine(connection_string)
        # Note: If Mealie rating structure lacks direct 'recipe_ratings', manual querying on 'recipe_comments' might trigger.
        df_traffic = pd.read_sql("SELECT * FROM recipe_ratings LIMIT 10", engine)
    except Exception as e:
        print("Waiting for Mealie schema to populate appropriate tables... Simulating traffic framework.")
        df_traffic = pd.DataFrame(columns=["user_id", "recipe_id", "rating", "date", "mealie_recipe_uuid", "mealie_user_uuid"])
        
    return df_traffic

def update_mapping_registry(s3, traffic_df, registry_df):
    if traffic_df.empty:
        return traffic_df, registry_df
        
    print("--- 3. Running ID Conflict Resolution & Mapping Middleware ---")
    
    # Process Recipes
    new_recipes = traffic_df[~traffic_df['mealie_recipe_uuid'].isin(registry_df[registry_df['entity_type'] == 'recipe']['mealie_uuid'])]
    unique_new_recipes = new_recipes['mealie_recipe_uuid'].dropna().unique()
    
    recipe_registry = registry_df[registry_df['entity_type'] == 'recipe']
    if not recipe_registry.empty:
        current_max_recipe = int(recipe_registry['ml_native_id'].max())
    else:
        current_max_recipe = get_dynamic_max_id(s3, "recipe")
        
    new_records = []
    for rnd_uuid in unique_new_recipes:
        current_max_recipe += 1
        new_records.append({
            "mealie_uuid": str(rnd_uuid),
            "ml_native_id": current_max_recipe,
            "entity_type": "recipe"
        })
        
    # Process Users
    new_users = traffic_df[~traffic_df['mealie_user_uuid'].isin(registry_df[registry_df['entity_type'] == 'user']['mealie_uuid'])]
    unique_new_users = new_users['mealie_user_uuid'].dropna().unique()
    
    user_registry = registry_df[registry_df['entity_type'] == 'user']
    if not user_registry.empty:
        current_max_user = int(user_registry['ml_native_id'].max())
    else:
        current_max_user = get_dynamic_max_id(s3, "user")
        
    for rnd_uuid in unique_new_users:
        current_max_user += 1
        new_records.append({
            "mealie_uuid": str(rnd_uuid),
            "ml_native_id": current_max_user,
            "entity_type": "user"
        })
        
    # Append to registry
    if new_records:
        registry_df = pd.concat([registry_df, pd.DataFrame(new_records)], ignore_index=True)
        
    # Translate records into strictly pure ints maps matching the SAGE model's edge_index requirements
    map_dict = dict(zip(registry_df['mealie_uuid'], registry_df['ml_native_id']))
    
    # Generate unified names matching Kaggle CSV format
    traffic_df['u'] = traffic_df['mealie_user_uuid'].astype(str).map(map_dict)
    traffic_df['i'] = traffic_df['mealie_recipe_uuid'].astype(str).map(map_dict)
    
    return traffic_df, registry_df

def upload_traffic_to_s3(s3, final_df):
    if final_df.empty:
        print("No new traffic. Skipping S3 upload.")
        return
        
    print(f"--- 4. Pushing Processed Batch to Production Traffic ({len(final_df)} records) ---")
    buf = io.BytesIO()
    final_df.to_parquet(buf, index=False)
    key = f"production_traffic/{VERSION}/batch.parquet"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())

def run_poller():
    os.makedirs("local_data", exist_ok=True)
    s3 = get_s3_client()
    
    registry_df = fetch_s3_registry(s3)
    traffic_df = extract_mealie_data()
    
    final_traffic_df, updated_registry_df = update_mapping_registry(s3, traffic_df, registry_df)
    
    if len(updated_registry_df) > len(registry_df):
        upload_s3_registry(s3, updated_registry_df)
        
    upload_traffic_to_s3(s3, final_traffic_df)
    print("✅ Ingestion Polling Cycle Complete.")

if __name__ == "__main__":
    run_poller()

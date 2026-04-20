import os
import io
import ast
import requests
import pandas as pd
import boto3
from botocore.client import Config

# --- Environment Context ---
MEALIE_API_URL = os.getenv("MEALIE_BASE_URL", "http://localhost:9000")
MEALIE_TOKEN = os.getenv("MEALIE_API_TOKEN")

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

def get_s3_client():
    return boto3.client('s3', 
        aws_access_key_id=AWS_ACCESS_KEY, 
        aws_secret_access_key=AWS_SECRET_KEY, 
        endpoint_url=ENDPOINT_URL, 
        config=Config(signature_version='s3v4'), 
        region_name='us-east-1')

def load_kaggle_recipes():
    print("Loading Local Kaggle CSVs...")
    df_raw = pd.read_csv("local_data/RAW_recipes.csv")
    df_pp = pd.read_csv("local_data/PP_recipes.csv")
    # Join on Kaggle ID to align metadata with contiguous index `i`
    df_merged = pd.merge(df_raw, df_pp[['id', 'i']], on='id', how='inner')
    
    # Sort or pick top 500 highest ingredient counts / random
    # To keep the demo fast, we'll pick first 500
    df_subset = df_merged.head(500)
    return df_subset

def push_to_mealie_api(df_recipes):
    print(f"Connecting to Mealie at {MEALIE_API_URL} ...")
    headers = {
        "Authorization": f"Bearer {MEALIE_TOKEN}",
        "Content-Type": "application/json"
    }
    
    mapping_records = []
    
    for idx, row in df_recipes.iterrows():
        # Parse ingredients from stringified list "['sugar', 'salt']"
        try:
            ingredients = ast.literal_eval(row['ingredients'])
        except:
            ingredients = []
            
        recipe_payload = {
            "name": str(row['name']).title(),
            "description": str(row['description']) if pd.notna(row['description']) else "A classic food.com recipe.",
            "recipeIngredient": [{"note": ing} for ing in ingredients]
        }
        
        try:
            resp = requests.post(f"{MEALIE_API_URL}/api/recipes", json=recipe_payload, headers=headers)
            if resp.status_code in [200, 201]:
                # Mealie returns recipe ID or slug. In v1.0.0-beta.5 usually it returns the Recipe ID string.
                created_id = resp.json()
                if isinstance(created_id, dict):
                    created_id = created_id.get('id', str(resp.json()))
                print(f"✅ Seaded Recipe -> Mealie ID: {created_id} (Kaggle i: {row['i']})")
                
                mapping_records.append({
                    "mealie_uuid": str(created_id),
                    "ml_native_id": int(row['i']),
                    "entity_type": "recipe"
                })
            else:
                print(f"❌ Failed to push recipe: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"Connection failed: {e}")
            break
            
    return pd.DataFrame(mapping_records)

def save_registry_to_s3(registry_df, s3):
    if registry_df.empty:
        print("No records to save. Aborting S3 push.")
        return
        
    print(f"Saving Seed Registry with {len(registry_df)} nodes to S3...")
    buf = io.BytesIO()
    registry_df.to_parquet(buf, index=False)
    # Using the standard location from our architecture
    key = "dataset/registry/id_mapping_registry.parquet"
    s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=buf.getvalue())
    print("🚀 Initial Mealie Base Successfully Embedded into Object Store.")

if __name__ == "__main__":
    if not MEALIE_TOKEN:
        print("ERROR: MEALIE_API_TOKEN environment variable not set. Please set it via .env")
        exit(1)
        
    df_recipes = load_kaggle_recipes()
    registry = push_to_mealie_api(df_recipes)
    
    s3 = get_s3_client()
    save_registry_to_s3(registry, s3)

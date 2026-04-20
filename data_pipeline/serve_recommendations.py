import os
import io
import pandas as pd
import boto3
import requests
from botocore.client import Config
from sqlalchemy import create_engine, text

# --- Environment Context ---
def get_env_url(var_name, docker_default, host_default):
    val = os.getenv(var_name)
    if val: return val
    return docker_default if os.path.exists('/.dockerenv') else host_default

MEALIE_API_URL = get_env_url("MEALIE_BASE_URL", "http://mealie-frontend:9000", "http://localhost:9000")
MEALIE_TOKEN = os.getenv("MEALIE_API_TOKEN")

MEALIE_POSTGRES_URL = get_env_url("POSTGRES_URL", "postgresql://mealie:mealie_password@mealie-postgres:5432/mealie", "postgresql://mealie:mealie_password@localhost:5432/mealie")

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

def fetch_s3_registry(s3):
    print("--- 1. Pulling Global ID Registry from S3 ---")
    key = "dataset/registry/id_mapping_registry.parquet"
    try:
        obj = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        registry_df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        return registry_df
    except Exception as e:
        print(f"Failed to fetch registry: {e}")
        return pd.DataFrame()

def generate_mock_gnn_predictions(registry_df, num_recipes=7):
    print("--- 2. Simulating GNN Model Output (Top-K Int IDs) ---")
    recipe_registry = registry_df[registry_df['entity_type'] == 'recipe']
    user_registry = registry_df[registry_df['entity_type'] == 'user']
    
    if recipe_registry.empty or user_registry.empty:
        print("Registry is empty. Cannot generate predictions.")
        return [], None
        
    target_user_int = int(user_registry['ml_native_id'].max())
    target_user_uuid = user_registry[user_registry['ml_native_id'] == target_user_int]['mealie_uuid'].iloc[0]
    
    predicted_recipe_ints = recipe_registry['ml_native_id'].sample(n=num_recipes, replace=False).tolist()
    
    print(f"GNN Output for User {target_user_int}: Recipes {predicted_recipe_ints}")
    return predicted_recipe_ints, target_user_uuid

def map_predictions_to_uuid(registry_df, predicted_ints):
    print("--- 3. Inverse Mapping ML Ints -> Mealie UUIDs ---")
    recipe_registry = registry_df[registry_df['entity_type'] == 'recipe']
    
    mapped_uuids = []
    for p_int in predicted_ints:
        match = recipe_registry[recipe_registry['ml_native_id'] == p_int]
        if not match.empty:
            mapped_uuids.append(match.iloc[0]['mealie_uuid'])
            
    print(f"Successfully mapped {len(mapped_uuids)} UUIDs for Database Injection.")
    return mapped_uuids


def inject_tags_via_database(recipe_uuids):
    """
    Solves the Mealie REST API deduplication bug by writing the Tag associations 
    directly into the PostgreSQL relational table.
    """
    print("--- 4. Direct Database Tag Injection (Bypassing API Constraints) ---")
    headers = {
        "Authorization": f"Bearer {MEALIE_TOKEN}",  # We need the API token for initial creation
        "Content-Type": "application/json"
    }
    
    tag_name = "🤖 AI Recommended"
    
    try:
        engine = create_engine(MEALIE_POSTGRES_URL)
        
        with engine.begin() as conn:
            # 1. Fetch Tag ID directly from Postgres (guaranteed to exist from previous run)
            print("Locating the unified Tag ID directly via Database...")
            tag_query = text("SELECT id FROM tags WHERE name = :name LIMIT 1")
            result = conn.execute(tag_query, {"name": tag_name}).fetchone()
            
            if not result:
                print("❌ Master Tag not found in Database.")
                return
                
            tag_id = result[0]
            print(f"✅ Master Tag Ready: ID [{tag_id}]")
            
            # 2. Use Direct Database Connection to forcibly attach recipes
            print("Dropping into Postgres directly to execute forced linkage...")
            success_count = 0
            for r_slug in recipe_uuids:
                # 1. Resolve the API Slug into the Database Native UUID
                resolver_query = text("SELECT id FROM recipes WHERE slug = :slug OR id::text = :slug LIMIT 1")
                recipe_row = conn.execute(resolver_query, {"slug": str(r_slug)}).fetchone()
                
                if not recipe_row:
                    print(f"⚠️ Warning: Could not locate underlying DB UUID for recipe '{r_slug}'. Skipping.")
                    continue
                    
                db_recipe_uuid = recipe_row[0]
                
                # 2. Link using Strict DB UUIDs
                query = text("""
                    INSERT INTO recipes_to_tags (recipe_id, tag_id)
                    VALUES (:recipe_id, :tag_id)
                    ON CONFLICT DO NOTHING
                """)
                conn.execute(query, {"recipe_id": db_recipe_uuid, "tag_id": str(tag_id)})
                success_count += 1
                print(f"🌟 Forcibly Linked Recipe [ {r_slug[:40]} ] to AI Recommendations!")
                
        print(f"\n🎉 OPERATION COMPLETE: Successfully injected {success_count} recipes.")
        print("👉 Go to your Mealie UI, click 'Tags' -> '🤖 AI Recommended' to see all of them natively!")
    except Exception as e:
        print(f"\n❌ Database Connection Failed: {e}")

if __name__ == "__main__":
    if not MEALIE_TOKEN:
        print("ERROR: MEALIE_API_TOKEN environment variable not set. Please set it in .env")
        exit(1)
        
    s3 = get_s3_client()
    registry = fetch_s3_registry(s3)
    
    predicted_ints, target_user_uuid = generate_mock_gnn_predictions(registry, num_recipes=7)
    if predicted_ints:
        predicted_uuids = map_predictions_to_uuid(registry, predicted_ints)
        
        # Execute the invincible Database Injector
        inject_tags_via_database(predicted_uuids)

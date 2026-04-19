import time
import json
import random
import os
import boto3
from datetime import datetime
from botocore.client import Config

# --- Configuration ---
ACCESS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
SECRET_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

VERSION = time.strftime("%Y%m%d_%H%M")
BATCH_SIZE = 50
TICK_INTERVAL = 0.5

def get_s3_client():
    return boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=ENDPOINT_URL, config=Config(signature_version='s3v4'), region_name='us-east-1')

def generate_mealie_payload():
    # Schema alignment: Prefix IDs with user: and recipe:
    # We use high randomized values to represent "new active users" not in Kaggle
    user_id = f"user:{random.randint(500000, 600000)}"
    recipe_id = f"recipe:{random.randint(1, 10000)}"
    
    return {
        "event_id": f"event_{random.randint(10000, 99999)}",
        "user_id": user_id,
        "recipe_id": recipe_id,
        "rating": float(random.randint(3, 5)), # simulate primarily positive interactions
        "review": "Mock synthetic review",
        "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def run_generator(duration_minutes=1):
    s3 = get_s3_client()
    if duration_minutes <= 0:
        end_time = float('inf')
    else:
        end_time = time.time() + (duration_minutes * 60)
    
    buffer = []
    batch_count = 0

    print(f"--- [Mealie Simulator] Generating Synthetic Mealie Webhooks ---")
    
    try:
        while time.time() < end_time:
            buffer.append(generate_mealie_payload())
            
            if len(buffer) >= BATCH_SIZE:
                batch_count += 1
                local_path = f"local_data/batch_{VERSION}_{batch_count}.json"
                os.makedirs("local_data", exist_ok=True)
                
                with open(local_path, 'w') as f:
                    json.dump(buffer, f)
                
                s3_key = f"simulation/{VERSION}/batch_{batch_count}.json"
                print(f"Uploading simulation batch {batch_count} containing {len(buffer)} events...")
                s3.upload_file(local_path, BUCKET_NAME, s3_key)
                
                os.remove(local_path)
                buffer = []
            
            time.sleep(TICK_INTERVAL)
            
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    run_generator()

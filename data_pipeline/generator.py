import time
import json
import random
import os
import pandas as pd
import boto3
from datetime import datetime
from botocore.client import Config

# --- Configuration ---
ACCESS_KEY = "af8beb8221454104b30fa21e3fad8a4c"
SECRET_KEY = "fa72afa48dd941cba3dde168382eabc8"
BUCKET_NAME = "ObjStore_proj14" 
ENDPOINT_URL = "https://chi.tacc.chameleoncloud.org:7480"
VERSION = time.strftime("%Y%m%d_%H%M")

# Simulation Parameters
TOTAL_USERS = 5000       
TOTAL_RECIPES = 180000   
BATCH_SIZE = 100         # Buffer 100 rows before uploading
TICK_INTERVAL = 0.1      # Generate data every 0.1s for high volume

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def run_simulation(duration_minutes=3):
    s3 = get_s3_client()
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    buffer = []
    batch_count = 0
    total_generated = 0

    print(f"--- [Mealie Simulation] High-Volume Generator Started ---")
    print(f"--- Target Duration: {duration_minutes} minutes | Version: {VERSION} ---")

    try:
        while time.time() < end_time:
            # Generate synthetic interaction
            event = {
                "user_id": random.randint(1, TOTAL_USERS),
                "recipe_id": random.randint(1, TOTAL_RECIPES),
                "rating": random.randint(1, 5),
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
            }
            buffer.append(event)
            total_generated += 1

            # When buffer is full, upload to Object Storage
            if len(buffer) >= BATCH_SIZE:
                batch_count += 1
                batch_filename = f"sim_batch_{VERSION}_{batch_count}.json"
                local_path = f"data/{batch_filename}"
                
                # Save batch locally
                with open(local_path, 'w') as f:
                    json.dump(buffer, f)
                
                # Upload to simulation folder
                s3_key = f"simulation/{VERSION}/{batch_filename}"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Uploading Batch {batch_count} ({total_generated} total rows)...")
                
                s3.upload_file(local_path, BUCKET_NAME, s3_key)
                
                # Cleanup and reset buffer
                os.remove(local_path)
                buffer = []

            time.sleep(TICK_INTERVAL)

    except KeyboardInterrupt:
        print("\n--- Simulation Interrupted by User ---")
    
    print(f"\n--- Simulation Complete ---")
    print(f"Total Rows Generated: {total_generated}")
    print(f"Total Batches Uploaded: {batch_count}")
    print(f"Data stored in: {BUCKET_NAME}/simulation/{VERSION}/")

if __name__ == "__main__":
    # Ensure data directory exists for temporary batches
    os.makedirs("data", exist_ok=True)
    run_simulation(duration_minutes=3)
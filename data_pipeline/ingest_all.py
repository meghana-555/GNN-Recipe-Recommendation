import pandas as pd
import numpy as np
import os
import glob
import boto3
import time
from botocore.client import Config
from boto3.s3.transfer import TransferConfig


ACCESS_KEY = "af8beb8221454104b30fa21e3fad8a4c"
SECRET_KEY = "fa72afa48dd941cba3dde168382eabc8"
PROJECT_ID = "CHI-251409"
BUCKET_NAME = f"ObjStore_proj14"
ENDPOINT_URL = "https://chi.tacc.chameleoncloud.org:7480" 
VERSION = time.strftime("%Y%m%d_%H%M") 

def process_and_upload():
    transfer_config = TransferConfig(
        multipart_threshold=1024 * 1024 * 1024, 
        use_threads=False
    )
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

    csv_files = glob.glob("data/*.csv")
    print(f"Starting Ingestion for Version: {VERSION}")
    
    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        if file_name.startswith("processed_"): continue
        
        print(f"\n--- Handling: {file_name} ---")
        try:
            df = pd.read_csv(file_path)
            if len(df) < 50000:
                df = pd.concat([df, df.sample(n=50000, replace=True)])
            
            temp_path = f"data/processed_{file_name}"
            df.to_csv(temp_path, index=False)

            versioned_key = f"dataset/{VERSION}/{file_name}"
            print(f"Uploading to {BUCKET_NAME}/{versioned_key}...")
            
            s3.upload_file(
                Filename=temp_path, 
                Bucket=BUCKET_NAME, 
                Key=versioned_key,
                Config=transfer_config
            )
            print(f"✅ SUCCESS: {file_name} (Version: {VERSION})")
            if os.path.exists(temp_path): os.remove(temp_path)
        except Exception as e:
            print(f"❌ FAILED {file_name}: {e}")

if __name__ == "__main__":
    process_and_upload()
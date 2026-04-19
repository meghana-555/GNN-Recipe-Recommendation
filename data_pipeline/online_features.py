import json
import boto3
import os
import io
import time
import pandas as pd
from botocore.client import Config

ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "af8beb8221454104b30fa21e3fad8a4c")
SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "fa72afa48dd941cba3dde168382eabc8")
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

def mock_historic_fetch(user_id):
    """
    In a real system, this queries a fast Feature Store (Redis) or SQLite.
    For this code, we just return simulated realistic averages matching the node features.
    """
    return {
        "count": 14,
        "average_rating": 4.5,
        "reviews_written": 2,
        "recent_recipe_history": ["recipe:170078", "recipe:12345"]
    }

def log_inference_drift(user_id, stats):
    """Logs the queries properties back to S3 to monitor for drift."""
    try:
        s3 = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=ENDPOINT_URL, config=Config(signature_version='s3v4'), region_name='us-east-1')
        drift_payload = {
            "query_time": time.time(),
            "user_id": user_id,
            "stats": stats
        }
        key = f"monitoring/{time.strftime('%Y%m%d')}/drift_log_{int(time.time()*1000)}.json"
        s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps(drift_payload))
    except Exception as e:
        print(f"Warning: Failed to log drift metrics: {e}")

def compute_online_features(input_payload):
    """
    Expected input_payload matching serving/sample_io/input_sample.json:
    {"user_id": "user:13751"}
    """
    try:
        data = json.loads(input_payload)
        user_id = data.get("user_id")
    except Exception:
        return {"error": "Invalid JSON input"}
        
    if not user_id or not user_id.startswith("user:"):
        return {"error": "Invalid user_id format. Must be user:ID"}

    # Fetch features
    stats = mock_historic_fetch(user_id)
    
    # Production Validation / Evaluation 3:
    if stats["average_rating"] == 0 or len(stats["recent_recipe_history"]) == 0:
        print(f"[DRIFT ALARM] User {user_id} has alarming feature sparsity.")

    log_inference_drift(user_id, stats)

    # Output Schema matched exactly for Serving
    return {
        "user_id": user_id,
        "status": "active",
        "feature_payload": {
            "interaction_stats": stats,
            "inference_seeds": {
                "recent_recipe_history": stats["recent_recipe_history"]
            }
        }
    }

if __name__ == "__main__":
    # Test aligning exactly to sample_io format
    test_json = '{"user_id": "user:13751"}'
    print(f"Input: {test_json}")
    result = compute_online_features(test_json)
    print("Output Features:")
    print(json.dumps(result, indent=4))

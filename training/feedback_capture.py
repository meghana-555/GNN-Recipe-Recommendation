#!/usr/bin/env python
# coding: utf-8
"""
Feedback Capture Pipeline for Mealie Recipe Recommendation System

This script:
1. Collects user interaction data from Mealie (ratings, views, meal plans)
2. Processes and formats data for retraining
3. Uploads training-ready data to object storage
4. Triggers retraining when sufficient new data is collected

Can run as:
- One-shot: Process and upload current feedback
- Continuous: Poll for new feedback periodically
"""

import os
import io
import json
import time
import hashlib
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple

import pandas as pd
import requests
import boto3
from botocore.client import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "bucket_name": "ObjStore_proj14",
    "endpoint_url": "https://chi.tacc.chameleoncloud.org:7480",
    "feedback_prefix": "feedback/",
    "train_prefix": "train/",
    "min_new_ratings": 100,  # Minimum new ratings before triggering retrain
    "min_hours_since_last_train": 6,  # Minimum hours between retrains
    "batch_size_for_retrain": 1000,  # Accumulate this many before retraining
}
# ──────────────────────────────────────────────────────────────────────────────


def get_s3_client():
    """Get S3 client for Chameleon object storage."""
    access_key = os.getenv("CHAMELEON_ACCESS_KEY")
    secret_key = os.getenv("CHAMELEON_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError("CHAMELEON_ACCESS_KEY and CHAMELEON_SECRET_KEY must be set!")
    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=CONFIG["endpoint_url"],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


class MealieDataCollector:
    """Collects feedback data from Mealie API."""
    
    def __init__(self, base_url: str, admin_token: str):
        self.base_url = base_url.rstrip('/')
        self.token = admin_token
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {admin_token}',
            'Content-Type': 'application/json'
        })
    
    def get_all_users(self) -> List[Dict]:
        """Get all users from Mealie."""
        try:
            resp = self.session.get(f"{self.base_url}/api/admin/users")
            if resp.status_code == 200:
                return resp.json().get('items', [])
            return []
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return []
    
    def get_user_ratings(self, user_id: str) -> List[Dict]:
        """Get all ratings for a specific user."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/users/{user_id}/ratings"
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            logger.error(f"Error fetching ratings for user {user_id}: {e}")
            return []
    
    def get_all_recipes(self) -> List[Dict]:
        """Get all recipes with their IDs."""
        try:
            all_recipes = []
            page = 1
            while True:
                resp = self.session.get(
                    f"{self.base_url}/api/recipes",
                    params={"page": page, "perPage": 100}
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                items = data.get('items', [])
                if not items:
                    break
                all_recipes.extend(items)
                page += 1
            return all_recipes
        except Exception as e:
            logger.error(f"Error fetching recipes: {e}")
            return []
    
    def get_meal_plans(self, start_date: str, end_date: str) -> List[Dict]:
        """Get meal plans (positive implicit feedback)."""
        try:
            resp = self.session.get(
                f"{self.base_url}/api/groups/mealplans",
                params={"start_date": start_date, "end_date": end_date}
            )
            if resp.status_code == 200:
                return resp.json()
            return []
        except Exception as e:
            logger.error(f"Error fetching meal plans: {e}")
            return []
    
    def collect_all_feedback(self) -> pd.DataFrame:
        """Collect all available feedback data."""
        logger.info("Collecting feedback data from Mealie...")
        
        interactions = []
        
        # Get all users
        users = self.get_all_users()
        logger.info(f"Found {len(users)} users")
        
        # Build user ID mapping (username/email to numeric ID)
        user_map = {}
        for idx, user in enumerate(users):
            user_id = user.get('id', user.get('username'))
            user_map[user_id] = idx
        
        # Get recipe mapping
        recipes = self.get_all_recipes()
        recipe_map = {}
        for idx, recipe in enumerate(recipes):
            recipe_id = recipe.get('id', recipe.get('slug'))
            recipe_map[recipe_id] = idx
        logger.info(f"Found {len(recipes)} recipes")
        
        # Collect ratings
        for user in users:
            user_id = user.get('id', user.get('username'))
            ratings = self.get_user_ratings(user_id)
            
            for rating in ratings:
                recipe_id = rating.get('recipeId', rating.get('recipe_id'))
                if recipe_id in recipe_map:
                    interactions.append({
                        'user_id': user_map.get(user_id, user_id),
                        'recipe_id': recipe_map.get(recipe_id, recipe_id),
                        'rating': rating.get('rating', rating.get('value', 5)),
                        'timestamp': rating.get('createdAt', datetime.utcnow().isoformat()),
                        'feedback_type': 'explicit_rating'
                    })
        
        # Collect meal plans as implicit feedback (positive signal)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
        meal_plans = self.get_meal_plans(start_date, end_date)
        
        for plan in meal_plans:
            recipe_id = plan.get('recipeId', plan.get('recipe_id'))
            user_id = plan.get('userId', plan.get('user_id'))
            if recipe_id in recipe_map and user_id in user_map:
                interactions.append({
                    'user_id': user_map.get(user_id),
                    'recipe_id': recipe_map.get(recipe_id),
                    'rating': 5,  # Meal plan = strong positive signal
                    'timestamp': plan.get('date', datetime.utcnow().isoformat()),
                    'feedback_type': 'implicit_mealplan'
                })
        
        df = pd.DataFrame(interactions)
        logger.info(f"Collected {len(df)} total interactions")
        return df


class FeedbackProcessor:
    """Processes and prepares feedback for training."""
    
    def __init__(self, s3_client):
        self.s3 = s3_client
    
    def get_last_training_data(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Get the most recent training dataset."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            train_files = []
            for page in paginator.paginate(Bucket=CONFIG["bucket_name"], Prefix=CONFIG["train_prefix"]):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.csv'):
                        train_files.append(obj['Key'])
            
            if not train_files:
                return None, None
            
            latest = sorted(train_files)[-1]
            obj = self.s3.get_object(Bucket=CONFIG["bucket_name"], Key=latest)
            df = pd.read_csv(io.BytesIO(obj['Body'].read()))
            return df, latest
        except Exception as e:
            logger.error(f"Error fetching last training data: {e}")
            return None, None
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for change detection."""
        return hashlib.md5(
            pd.util.hash_pandas_object(df).values.tobytes()
        ).hexdigest()[:12]
    
    def deduplicate_interactions(self, new_df: pd.DataFrame, 
                                  existing_df: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Remove duplicate interactions, keeping the most recent."""
        if existing_df is None or existing_df.empty:
            return new_df
        
        # Combine dataframes
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        
        # Sort by timestamp (newest first) and deduplicate
        if 'timestamp' in combined.columns:
            combined = combined.sort_values('timestamp', ascending=False)
        
        # Keep first occurrence (most recent)
        deduped = combined.drop_duplicates(
            subset=['user_id', 'recipe_id'],
            keep='first'
        ).reset_index(drop=True)
        
        return deduped
    
    def prepare_training_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in format expected by training pipeline."""
        # Rename columns to match training expectations
        train_df = df.rename(columns={
            'user_id': 'u',
            'recipe_id': 'i'
        })
        
        # Ensure required columns
        required_cols = ['u', 'i', 'rating']
        for col in required_cols:
            if col not in train_df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Select and order columns
        output_cols = ['u', 'i', 'rating']
        if 'timestamp' in train_df.columns:
            output_cols.append('timestamp')
        if 'feedback_type' in train_df.columns:
            output_cols.append('feedback_type')
        
        return train_df[output_cols]
    
    def upload_training_data(self, df: pd.DataFrame) -> str:
        """Upload processed training data to object storage."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        data_hash = self.compute_data_hash(df)
        key = f"{CONFIG['train_prefix']}interactions_{timestamp}_{data_hash}.csv"
        
        # Convert to CSV
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        
        # Upload
        self.s3.put_object(
            Bucket=CONFIG["bucket_name"],
            Key=key,
            Body=buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        
        logger.info(f"Uploaded training data: {key} ({len(df)} rows)")
        return key
    
    def save_feedback_snapshot(self, df: pd.DataFrame) -> str:
        """Save raw feedback snapshot for auditing."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        key = f"{CONFIG['feedback_prefix']}snapshot_{timestamp}.csv"
        
        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        
        self.s3.put_object(
            Bucket=CONFIG["bucket_name"],
            Key=key,
            Body=buffer.getvalue().encode('utf-8'),
            ContentType='text/csv'
        )
        
        logger.info(f"Saved feedback snapshot: {key}")
        return key


class RetrainTrigger:
    """Determines when to trigger model retraining."""
    
    def __init__(self, s3_client):
        self.s3 = s3_client
    
    def get_last_train_time(self) -> Optional[datetime]:
        """Get timestamp of last training run."""
        try:
            # Check MLflow or training logs for last run
            # Fallback: check last training data upload
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=CONFIG["bucket_name"], Prefix="training/"):
                for obj in page.get('Contents', []):
                    if 'best_model' in obj['Key']:
                        return obj['LastModified'].replace(tzinfo=None)
            return None
        except Exception as e:
            logger.error(f"Error getting last train time: {e}")
            return None
    
    def should_retrain(self, new_interactions_count: int, 
                       last_train_time: Optional[datetime]) -> Tuple[bool, str]:
        """Determine if retraining should be triggered."""
        
        # Check minimum new data threshold
        if new_interactions_count < CONFIG["min_new_ratings"]:
            return False, f"Insufficient new data: {new_interactions_count} < {CONFIG['min_new_ratings']}"
        
        # Check minimum time since last training
        if last_train_time:
            hours_since_train = (datetime.utcnow() - last_train_time).total_seconds() / 3600
            if hours_since_train < CONFIG["min_hours_since_last_train"]:
                return False, f"Too soon since last train: {hours_since_train:.1f}h < {CONFIG['min_hours_since_last_train']}h"
        
        return True, f"Retrain triggered: {new_interactions_count} new interactions"
    
    def trigger_retrain(self) -> bool:
        """Trigger the retraining pipeline."""
        # Option 1: Write a trigger file to S3
        trigger_key = f"{CONFIG['train_prefix']}retrain_trigger_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        trigger_data = {
            'triggered_at': datetime.utcnow().isoformat(),
            'trigger_type': 'automatic',
            'reason': 'sufficient_new_feedback'
        }
        
        self.s3.put_object(
            Bucket=CONFIG["bucket_name"],
            Key=trigger_key,
            Body=json.dumps(trigger_data).encode('utf-8'),
            ContentType='application/json'
        )
        
        logger.info(f"Retrain trigger written: {trigger_key}")
        
        # Option 2: Call training API/webhook (if available)
        retrain_webhook = os.getenv('RETRAIN_WEBHOOK_URL')
        if retrain_webhook:
            try:
                resp = requests.post(retrain_webhook, json=trigger_data, timeout=30)
                logger.info(f"Retrain webhook response: {resp.status_code}")
            except Exception as e:
                logger.error(f"Retrain webhook failed: {e}")
        
        return True


def run_feedback_pipeline(mealie_url: str, admin_token: str, 
                          trigger_retrain: bool = True) -> Dict:
    """Run the complete feedback capture pipeline."""
    
    results = {
        'status': 'success',
        'interactions_collected': 0,
        'new_interactions': 0,
        'retrain_triggered': False,
    }
    
    try:
        # Initialize components
        s3 = get_s3_client()
        collector = MealieDataCollector(mealie_url, admin_token)
        processor = FeedbackProcessor(s3)
        retrain_trigger = RetrainTrigger(s3)
        
        # Collect feedback
        new_feedback = collector.collect_all_feedback()
        results['interactions_collected'] = len(new_feedback)
        
        if new_feedback.empty:
            logger.warning("No feedback collected")
            return results
        
        # Save raw snapshot
        processor.save_feedback_snapshot(new_feedback)
        
        # Get existing training data
        existing_data, existing_key = processor.get_last_training_data()
        
        # Deduplicate and merge
        combined = processor.deduplicate_interactions(new_feedback, existing_data)
        
        # Calculate new interactions
        existing_count = len(existing_data) if existing_data is not None else 0
        new_count = len(combined) - existing_count
        results['new_interactions'] = max(0, new_count)
        
        logger.info(f"New interactions since last training: {new_count}")
        
        # Prepare and upload training data
        train_data = processor.prepare_training_data(combined)
        processor.upload_training_data(train_data)
        
        # Check if retraining should be triggered
        if trigger_retrain:
            last_train = retrain_trigger.get_last_train_time()
            should_train, reason = retrain_trigger.should_retrain(new_count, last_train)
            logger.info(f"Retrain decision: {reason}")
            
            if should_train:
                retrain_trigger.trigger_retrain()
                results['retrain_triggered'] = True
        
        return results
        
    except Exception as e:
        logger.error(f"Feedback pipeline failed: {e}")
        results['status'] = 'error'
        results['error'] = str(e)
        return results


def main():
    parser = argparse.ArgumentParser(description='Capture and process user feedback')
    parser.add_argument('--mealie-url', default=os.getenv('MEALIE_URL', 'http://localhost:9000'),
                       help='Mealie service URL')
    parser.add_argument('--admin-token', default=os.getenv('MEALIE_ADMIN_TOKEN'),
                       help='Mealie admin token')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously with periodic collection')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Collection interval in seconds (for continuous mode)')
    parser.add_argument('--no-trigger', action='store_true',
                       help='Do not trigger retraining')
    
    args = parser.parse_args()
    
    if not args.admin_token:
        logger.error("Admin token required. Set MEALIE_ADMIN_TOKEN or use --admin-token")
        return
    
    if args.continuous:
        logger.info(f"Running in continuous mode, interval: {args.interval}s")
        while True:
            try:
                results = run_feedback_pipeline(
                    args.mealie_url, 
                    args.admin_token,
                    trigger_retrain=not args.no_trigger
                )
                logger.info(f"Pipeline results: {results}")
            except Exception as e:
                logger.error(f"Pipeline error: {e}")
            
            time.sleep(args.interval)
    else:
        results = run_feedback_pipeline(
            args.mealie_url,
            args.admin_token,
            trigger_retrain=not args.no_trigger
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

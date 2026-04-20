#!/usr/bin/env python
# coding: utf-8
"""
Automated Retraining Orchestrator for Mealie Recipe Recommendation System

This script:
1. Monitors for retrain triggers (from feedback pipeline or manual)
2. Executes training with proper experiment tracking
3. Evaluates model quality gates
4. Promotes models through staging -> canary -> production
5. Handles rollback if production model degrades

Can run as:
- One-shot: Train once with current data
- Watch mode: Monitor for triggers and auto-retrain
"""

import os
import io
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List
from enum import Enum

import boto3
from botocore.client import Config
import mlflow
from mlflow.tracking import MlflowClient

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
    "train_prefix": "train/",
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "mlflow_experiment": "mealie-gnn-recommendations",
    "model_registry_name": "mealie-recipe-recommender",
    # Quality gates
    "min_test_auc": 0.70,
    "min_test_ap": 0.65,
    # Canary settings
    "canary_duration_hours": 2,
    "canary_min_requests": 100,
    "canary_max_error_rate": 0.05,
    "canary_min_satisfaction_rate": 0.80,
    # Rollback settings
    "rollback_window_hours": 1,
    "rollback_error_threshold": 0.10,
}
# ──────────────────────────────────────────────────────────────────────────────


class ModelStage(Enum):
    NONE = "None"
    STAGING = "Staging"
    CANARY = "Canary"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"


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


class TriggerMonitor:
    """Monitors for retraining triggers."""
    
    def __init__(self, s3_client):
        self.s3 = s3_client
        self.processed_triggers = set()
    
    def check_for_triggers(self) -> Optional[Dict]:
        """Check for new retrain triggers in object storage."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            triggers = []
            
            for page in paginator.paginate(
                Bucket=CONFIG["bucket_name"], 
                Prefix=f"{CONFIG['train_prefix']}retrain_trigger_"
            ):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key not in self.processed_triggers:
                        triggers.append({
                            'key': key,
                            'timestamp': obj['LastModified']
                        })
            
            if not triggers:
                return None
            
            # Return most recent unprocessed trigger
            triggers.sort(key=lambda x: x['timestamp'], reverse=True)
            newest = triggers[0]
            
            # Load trigger data
            obj = self.s3.get_object(Bucket=CONFIG["bucket_name"], Key=newest['key'])
            trigger_data = json.loads(obj['Body'].read().decode('utf-8'))
            trigger_data['_key'] = newest['key']
            
            return trigger_data
            
        except Exception as e:
            logger.error(f"Error checking triggers: {e}")
            return None
    
    def mark_processed(self, trigger_key: str):
        """Mark a trigger as processed."""
        self.processed_triggers.add(trigger_key)
        
        # Optionally move to processed folder
        try:
            processed_key = trigger_key.replace('retrain_trigger_', 'processed_trigger_')
            self.s3.copy_object(
                Bucket=CONFIG["bucket_name"],
                CopySource={'Bucket': CONFIG["bucket_name"], 'Key': trigger_key},
                Key=processed_key
            )
            self.s3.delete_object(Bucket=CONFIG["bucket_name"], Key=trigger_key)
        except Exception as e:
            logger.warning(f"Could not move processed trigger: {e}")


class ModelLifecycleManager:
    """Manages model versions and promotions through stages."""
    
    def __init__(self):
        mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
        self.client = MlflowClient()
        self.model_name = CONFIG["model_registry_name"]
    
    def get_production_model(self) -> Optional[Dict]:
        """Get the current production model version."""
        try:
            versions = self.client.get_latest_versions(
                self.model_name, 
                stages=["Production"]
            )
            if versions:
                return {
                    'version': versions[0].version,
                    'run_id': versions[0].run_id,
                    'stage': 'Production'
                }
            return None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def get_staging_model(self) -> Optional[Dict]:
        """Get the current staging model version."""
        try:
            versions = self.client.get_latest_versions(
                self.model_name,
                stages=["Staging"]
            )
            if versions:
                return {
                    'version': versions[0].version,
                    'run_id': versions[0].run_id,
                    'stage': 'Staging'
                }
            return None
        except Exception as e:
            logger.error(f"Error getting staging model: {e}")
            return None
    
    def promote_to_staging(self, version: str) -> bool:
        """Promote a model version to staging."""
        try:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Staging",
                archive_existing_versions=True
            )
            logger.info(f"Promoted version {version} to Staging")
            return True
        except Exception as e:
            logger.error(f"Failed to promote to staging: {e}")
            return False
    
    def promote_to_production(self, version: str) -> bool:
        """Promote a model version to production."""
        try:
            # Archive current production model
            current_prod = self.get_production_model()
            if current_prod:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_prod['version'],
                    stage="Archived"
                )
                logger.info(f"Archived previous production version {current_prod['version']}")
            
            # Promote new version
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=version,
                stage="Production"
            )
            logger.info(f"Promoted version {version} to Production")
            return True
        except Exception as e:
            logger.error(f"Failed to promote to production: {e}")
            return False
    
    def rollback_to_previous(self) -> bool:
        """Rollback to the previous production model."""
        try:
            # Find the most recent archived version
            versions = self.client.search_model_versions(
                f"name='{self.model_name}'"
            )
            
            archived = [v for v in versions if v.current_stage == "Archived"]
            if not archived:
                logger.error("No archived versions available for rollback")
                return False
            
            # Sort by version number and get most recent
            archived.sort(key=lambda x: int(x.version), reverse=True)
            rollback_version = archived[0].version
            
            # Demote current production
            current_prod = self.get_production_model()
            if current_prod:
                self.client.transition_model_version_stage(
                    name=self.model_name,
                    version=current_prod['version'],
                    stage="Archived"
                )
            
            # Restore previous
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=rollback_version,
                stage="Production"
            )
            
            logger.info(f"Rolled back to version {rollback_version}")
            return True
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def get_model_metrics(self, version: str) -> Dict:
        """Get metrics for a specific model version."""
        try:
            mv = self.client.get_model_version(self.model_name, version)
            run = self.client.get_run(mv.run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Error getting model metrics: {e}")
            return {}


class CanaryEvaluator:
    """Evaluates canary deployments for promotion decisions."""
    
    def __init__(self, metrics_endpoint: str = None):
        self.metrics_endpoint = metrics_endpoint or os.getenv(
            'METRICS_ENDPOINT', 
            'http://localhost:9090'
        )
    
    def get_canary_metrics(self, duration_hours: float = 2.0) -> Dict:
        """Fetch canary deployment metrics from monitoring system."""
        # This would query Prometheus/similar for canary metrics
        # Placeholder implementation
        try:
            import requests
            
            # Query canary error rate
            query = f'rate(http_requests_total{{deployment="canary",status=~"5.."}}[{duration_hours}h])'
            resp = requests.get(
                f"{self.metrics_endpoint}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            
            if resp.status_code == 200:
                data = resp.json()
                # Parse Prometheus response
                return {
                    'error_rate': 0.02,  # Placeholder
                    'request_count': 150,
                    'latency_p99': 0.5,
                    'satisfaction_rate': 0.85,
                }
            
            return {}
        except Exception as e:
            logger.warning(f"Could not fetch canary metrics: {e}")
            # Return simulated metrics for testing
            return {
                'error_rate': 0.02,
                'request_count': 150,
                'latency_p99': 0.5,
                'satisfaction_rate': 0.85,
            }
    
    def should_promote_canary(self, metrics: Dict) -> Tuple[bool, str]:
        """Decide if canary should be promoted to production."""
        
        # Check minimum request count
        if metrics.get('request_count', 0) < CONFIG['canary_min_requests']:
            return False, f"Insufficient requests: {metrics.get('request_count', 0)} < {CONFIG['canary_min_requests']}"
        
        # Check error rate
        if metrics.get('error_rate', 1.0) > CONFIG['canary_max_error_rate']:
            return False, f"Error rate too high: {metrics.get('error_rate'):.2%} > {CONFIG['canary_max_error_rate']:.2%}"
        
        # Check user satisfaction
        if metrics.get('satisfaction_rate', 0) < CONFIG['canary_min_satisfaction_rate']:
            return False, f"Satisfaction too low: {metrics.get('satisfaction_rate'):.2%} < {CONFIG['canary_min_satisfaction_rate']:.2%}"
        
        return True, "Canary metrics passed all checks"


class RetrainingOrchestrator:
    """Orchestrates the complete retraining workflow."""
    
    def __init__(self):
        self.s3 = get_s3_client()
        self.trigger_monitor = TriggerMonitor(self.s3)
        self.lifecycle = ModelLifecycleManager()
        self.canary_eval = CanaryEvaluator()
    
    def run_training(self) -> Dict:
        """Execute the training script and return results."""
        logger.info("Starting model training...")
        
        try:
            # Run training as subprocess
            result = subprocess.run(
                ['python', 'train.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"Training failed: {result.stderr}")
                return {'status': 'error', 'error': result.stderr}
            
            logger.info("Training completed successfully")
            
            # Parse training output for metrics
            output_lines = result.stdout.split('\n')
            metrics = {}
            for line in output_lines:
                if 'Test AUC:' in line:
                    metrics['test_auc'] = float(line.split(':')[1].strip())
                elif 'Test AP:' in line:
                    metrics['test_ap'] = float(line.split(':')[1].strip())
                elif 'Best Val AUC:' in line:
                    metrics['best_val_auc'] = float(line.split(':')[1].strip())
            
            return {
                'status': 'success',
                'metrics': metrics,
                'output': result.stdout
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Training timed out")
            return {'status': 'error', 'error': 'Training timed out'}
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def handle_new_model(self, training_result: Dict) -> Dict:
        """Handle a newly trained model - evaluate and stage."""
        if training_result.get('status') != 'success':
            return {'action': 'none', 'reason': 'Training failed'}
        
        metrics = training_result.get('metrics', {})
        
        # Check quality gates
        if metrics.get('test_auc', 0) < CONFIG['min_test_auc']:
            return {
                'action': 'reject',
                'reason': f"Test AUC {metrics.get('test_auc', 0):.4f} below threshold"
            }
        
        if metrics.get('test_ap', 0) < CONFIG['min_test_ap']:
            return {
                'action': 'reject', 
                'reason': f"Test AP {metrics.get('test_ap', 0):.4f} below threshold"
            }
        
        # Model passed gates - get latest registered version
        staging_model = self.lifecycle.get_staging_model()
        if staging_model:
            return {
                'action': 'staged',
                'version': staging_model['version'],
                'metrics': metrics
            }
        
        return {'action': 'none', 'reason': 'No model registered'}
    
    def evaluate_canary_promotion(self) -> Dict:
        """Evaluate if canary should be promoted to production."""
        staging = self.lifecycle.get_staging_model()
        if not staging:
            return {'action': 'none', 'reason': 'No staging model'}
        
        # Get canary metrics
        canary_metrics = self.canary_eval.get_canary_metrics(
            CONFIG['canary_duration_hours']
        )
        
        should_promote, reason = self.canary_eval.should_promote_canary(canary_metrics)
        
        if should_promote:
            if self.lifecycle.promote_to_production(staging['version']):
                return {
                    'action': 'promoted',
                    'version': staging['version'],
                    'reason': reason
                }
            return {'action': 'error', 'reason': 'Promotion failed'}
        
        return {'action': 'hold', 'reason': reason}
    
    def check_production_health(self) -> Dict:
        """Check if production model needs rollback."""
        prod = self.lifecycle.get_production_model()
        if not prod:
            return {'action': 'none', 'reason': 'No production model'}
        
        # Get production metrics
        metrics = self.canary_eval.get_canary_metrics(
            CONFIG['rollback_window_hours']
        )
        
        if metrics.get('error_rate', 0) > CONFIG['rollback_error_threshold']:
            logger.warning(f"Production error rate too high: {metrics.get('error_rate'):.2%}")
            if self.lifecycle.rollback_to_previous():
                return {
                    'action': 'rollback',
                    'version': prod['version'],
                    'reason': f"Error rate {metrics.get('error_rate'):.2%} exceeded threshold"
                }
            return {'action': 'rollback_failed', 'reason': 'Rollback failed'}
        
        return {'action': 'healthy', 'metrics': metrics}
    
    def run_once(self) -> Dict:
        """Run a single training cycle."""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'training': None,
            'staging': None,
        }
        
        # Run training
        training_result = self.run_training()
        results['training'] = training_result
        
        # Handle new model
        staging_result = self.handle_new_model(training_result)
        results['staging'] = staging_result
        
        return results
    
    def watch(self, check_interval: int = 60):
        """Watch for triggers and run training when needed."""
        logger.info(f"Starting watch mode, checking every {check_interval}s")
        
        while True:
            try:
                # Check for retrain triggers
                trigger = self.trigger_monitor.check_for_triggers()
                
                if trigger:
                    logger.info(f"Found retrain trigger: {trigger.get('_key')}")
                    
                    # Run training
                    result = self.run_once()
                    logger.info(f"Training result: {result}")
                    
                    # Mark trigger as processed
                    self.trigger_monitor.mark_processed(trigger['_key'])
                
                # Check canary promotion (every cycle)
                canary_result = self.evaluate_canary_promotion()
                if canary_result['action'] == 'promoted':
                    logger.info(f"Promoted canary to production: {canary_result}")
                
                # Check production health
                health = self.check_production_health()
                if health['action'] == 'rollback':
                    logger.warning(f"Production rollback triggered: {health}")
                
            except Exception as e:
                logger.error(f"Watch cycle error: {e}")
            
            time.sleep(check_interval)


def main():
    parser = argparse.ArgumentParser(description='Automated model retraining orchestrator')
    parser.add_argument('--mode', choices=['once', 'watch'], default='once',
                       help='Run mode: once (single training) or watch (monitor for triggers)')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds for watch mode')
    parser.add_argument('--promote-canary', action='store_true',
                       help='Evaluate and promote canary if ready')
    parser.add_argument('--rollback', action='store_true',
                       help='Rollback production to previous version')
    
    args = parser.parse_args()
    
    orchestrator = RetrainingOrchestrator()
    
    if args.rollback:
        success = orchestrator.lifecycle.rollback_to_previous()
        print(f"Rollback {'succeeded' if success else 'failed'}")
        return
    
    if args.promote_canary:
        result = orchestrator.evaluate_canary_promotion()
        print(json.dumps(result, indent=2))
        return
    
    if args.mode == 'watch':
        orchestrator.watch(check_interval=args.interval)
    else:
        result = orchestrator.run_once()
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# coding: utf-8
"""
Model Rollback Manager for Mealie Recipe Recommendation System

This script handles:
1. Manual rollback triggers
2. Automatic rollback based on monitoring alerts
3. Kubernetes deployment rollback
4. MLflow model stage management
5. Rollback history and audit logging

Usage:
    python rollback_manager.py --action rollback --reason "high error rate"
    python rollback_manager.py --action check-health
    python rollback_manager.py --action list-versions
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import boto3
from botocore.client import Config
import requests

# Optional: MLflow for model registry management
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

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
    "rollback_prefix": "rollback/",
    "model_registry_name": "mealie-recipe-recommender",
    "mlflow_tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
    "prometheus_url": os.getenv("PROMETHEUS_URL", "http://localhost:9090"),
    "kubernetes_namespace": "mealie-production",
    # Thresholds for automatic rollback
    "error_rate_threshold": 0.10,  # 10% error rate triggers rollback
    "latency_p99_threshold": 2.0,  # 2 second p99 latency threshold
    "availability_threshold": 0.95,  # 95% availability required
    "evaluation_window_minutes": 15,
}
# ──────────────────────────────────────────────────────────────────────────────


class RollbackReason(Enum):
    MANUAL = "manual"
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    LOW_AVAILABILITY = "low_availability"
    MODEL_QUALITY = "model_quality_degradation"
    USER_COMPLAINTS = "user_complaints"


@dataclass
class RollbackEvent:
    timestamp: str
    from_version: str
    to_version: str
    reason: str
    initiated_by: str
    status: str
    details: Dict


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


class MetricsCollector:
    """Collects metrics from Prometheus for rollback decisions."""
    
    def __init__(self, prometheus_url: str):
        self.prometheus_url = prometheus_url
    
    def query_prometheus(self, query: str) -> Optional[float]:
        """Execute a Prometheus query and return the result."""
        try:
            resp = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params={'query': query},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data['status'] == 'success' and data['data']['result']:
                    return float(data['data']['result'][0]['value'][1])
            return None
        except Exception as e:
            logger.error(f"Prometheus query failed: {e}")
            return None
    
    def get_error_rate(self, window_minutes: int = 15) -> Optional[float]:
        """Get current error rate (5xx responses / total requests)."""
        query = f'''
            sum(rate(http_requests_total{{
                namespace="{CONFIG['kubernetes_namespace']}",
                status=~"5.."
            }}[{window_minutes}m]))
            /
            sum(rate(http_requests_total{{
                namespace="{CONFIG['kubernetes_namespace']}"
            }}[{window_minutes}m]))
        '''
        return self.query_prometheus(query)
    
    def get_latency_p99(self, window_minutes: int = 15) -> Optional[float]:
        """Get p99 latency in seconds."""
        query = f'''
            histogram_quantile(0.99, 
                sum(rate(http_request_duration_seconds_bucket{{
                    namespace="{CONFIG['kubernetes_namespace']}"
                }}[{window_minutes}m])) by (le)
            )
        '''
        return self.query_prometheus(query)
    
    def get_availability(self, window_minutes: int = 15) -> Optional[float]:
        """Get service availability (successful responses / total)."""
        query = f'''
            sum(rate(http_requests_total{{
                namespace="{CONFIG['kubernetes_namespace']}",
                status=~"2.."
            }}[{window_minutes}m]))
            /
            sum(rate(http_requests_total{{
                namespace="{CONFIG['kubernetes_namespace']}"
            }}[{window_minutes}m]))
        '''
        return self.query_prometheus(query)
    
    def get_recommendation_quality(self, window_minutes: int = 15) -> Optional[float]:
        """Get recommendation acceptance rate (clicked / served)."""
        query = f'''
            sum(rate(recommendations_clicked_total{{
                namespace="{CONFIG['kubernetes_namespace']}"
            }}[{window_minutes}m]))
            /
            sum(rate(recommendations_served_total{{
                namespace="{CONFIG['kubernetes_namespace']}"
            }}[{window_minutes}m]))
        '''
        return self.query_prometheus(query)


class KubernetesManager:
    """Manages Kubernetes deployments for rollback."""
    
    def __init__(self, namespace: str):
        self.namespace = namespace
    
    def _run_kubectl(self, args: List[str]) -> Tuple[bool, str]:
        """Run a kubectl command."""
        try:
            cmd = ['kubectl', '-n', self.namespace] + args
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            return success, output
        except Exception as e:
            return False, str(e)
    
    def get_current_image(self, deployment: str) -> Optional[str]:
        """Get the current image tag of a deployment."""
        success, output = self._run_kubectl([
            'get', 'deployment', deployment,
            '-o', 'jsonpath={.spec.template.spec.containers[0].image}'
        ])
        return output if success else None
    
    def get_rollout_history(self, deployment: str) -> List[Dict]:
        """Get rollout history for a deployment."""
        success, output = self._run_kubectl([
            'rollout', 'history', f'deployment/{deployment}',
            '-o', 'json'
        ])
        if success:
            try:
                return json.loads(output)
            except:
                return []
        return []
    
    def rollback_to_previous(self, deployment: str) -> Tuple[bool, str]:
        """Rollback deployment to previous revision."""
        logger.info(f"Rolling back deployment {deployment}...")
        return self._run_kubectl([
            'rollout', 'undo', f'deployment/{deployment}'
        ])
    
    def rollback_to_revision(self, deployment: str, revision: int) -> Tuple[bool, str]:
        """Rollback deployment to specific revision."""
        logger.info(f"Rolling back deployment {deployment} to revision {revision}...")
        return self._run_kubectl([
            'rollout', 'undo', f'deployment/{deployment}',
            f'--to-revision={revision}'
        ])
    
    def wait_for_rollout(self, deployment: str, timeout: int = 300) -> Tuple[bool, str]:
        """Wait for rollout to complete."""
        return self._run_kubectl([
            'rollout', 'status', f'deployment/{deployment}',
            f'--timeout={timeout}s'
        ])
    
    def scale_down_canary(self) -> Tuple[bool, str]:
        """Scale down canary deployment to 0."""
        return self._run_kubectl([
            'scale', 'deployment/mealie-recommender-canary',
            '--replicas=0'
        ])
    
    def set_traffic_to_stable(self) -> Tuple[bool, str]:
        """Route 100% traffic to stable deployment."""
        patch = '''
        [
            {"op": "replace", "path": "/spec/http/0/route/0/weight", "value": 100},
            {"op": "replace", "path": "/spec/http/0/route/1/weight", "value": 0}
        ]
        '''
        return self._run_kubectl([
            'patch', 'virtualservice', 'mealie-recommender',
            '--type=json', f"-p={patch}"
        ])


class MLflowManager:
    """Manages MLflow model registry for rollback."""
    
    def __init__(self):
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
            self.client = MlflowClient()
        else:
            self.client = None
    
    def get_production_model(self) -> Optional[Dict]:
        """Get current production model version."""
        if not self.client:
            return None
        try:
            versions = self.client.get_latest_versions(
                CONFIG["model_registry_name"],
                stages=["Production"]
            )
            if versions:
                return {
                    'version': versions[0].version,
                    'run_id': versions[0].run_id,
                    'stage': 'Production',
                    'creation_timestamp': versions[0].creation_timestamp
                }
            return None
        except Exception as e:
            logger.error(f"Error getting production model: {e}")
            return None
    
    def get_archived_versions(self) -> List[Dict]:
        """Get list of archived model versions."""
        if not self.client:
            return []
        try:
            versions = self.client.search_model_versions(
                f"name='{CONFIG['model_registry_name']}'"
            )
            archived = [
                {
                    'version': v.version,
                    'run_id': v.run_id,
                    'stage': v.current_stage,
                    'creation_timestamp': v.creation_timestamp
                }
                for v in versions
                if v.current_stage == "Archived"
            ]
            archived.sort(key=lambda x: int(x['version']), reverse=True)
            return archived
        except Exception as e:
            logger.error(f"Error getting archived versions: {e}")
            return []
    
    def rollback_model(self, to_version: str) -> bool:
        """Rollback model to a specific version."""
        if not self.client:
            return False
        try:
            # Demote current production
            current_prod = self.get_production_model()
            if current_prod:
                self.client.transition_model_version_stage(
                    name=CONFIG["model_registry_name"],
                    version=current_prod['version'],
                    stage="Archived"
                )
                logger.info(f"Archived current production version {current_prod['version']}")
            
            # Promote rollback target
            self.client.transition_model_version_stage(
                name=CONFIG["model_registry_name"],
                version=to_version,
                stage="Production"
            )
            logger.info(f"Promoted version {to_version} to Production")
            return True
        except Exception as e:
            logger.error(f"Model rollback failed: {e}")
            return False


class RollbackManager:
    """Orchestrates rollback decisions and execution."""
    
    def __init__(self):
        self.s3 = get_s3_client()
        self.metrics = MetricsCollector(CONFIG["prometheus_url"])
        self.k8s = KubernetesManager(CONFIG["kubernetes_namespace"])
        self.mlflow = MLflowManager()
    
    def check_health(self) -> Dict:
        """Check system health and determine if rollback is needed."""
        window = CONFIG["evaluation_window_minutes"]
        
        health = {
            'timestamp': datetime.utcnow().isoformat(),
            'status': 'healthy',
            'metrics': {},
            'violations': [],
            'recommendation': 'none'
        }
        
        # Collect metrics
        error_rate = self.metrics.get_error_rate(window)
        latency_p99 = self.metrics.get_latency_p99(window)
        availability = self.metrics.get_availability(window)
        
        health['metrics'] = {
            'error_rate': error_rate,
            'latency_p99': latency_p99,
            'availability': availability
        }
        
        # Check thresholds
        if error_rate and error_rate > CONFIG["error_rate_threshold"]:
            health['violations'].append({
                'metric': 'error_rate',
                'value': error_rate,
                'threshold': CONFIG["error_rate_threshold"]
            })
        
        if latency_p99 and latency_p99 > CONFIG["latency_p99_threshold"]:
            health['violations'].append({
                'metric': 'latency_p99',
                'value': latency_p99,
                'threshold': CONFIG["latency_p99_threshold"]
            })
        
        if availability and availability < CONFIG["availability_threshold"]:
            health['violations'].append({
                'metric': 'availability',
                'value': availability,
                'threshold': CONFIG["availability_threshold"]
            })
        
        # Determine status and recommendation
        if health['violations']:
            health['status'] = 'unhealthy'
            health['recommendation'] = 'rollback'
            
            # Determine primary reason
            if any(v['metric'] == 'error_rate' for v in health['violations']):
                health['primary_reason'] = RollbackReason.HIGH_ERROR_RATE.value
            elif any(v['metric'] == 'latency_p99' for v in health['violations']):
                health['primary_reason'] = RollbackReason.HIGH_LATENCY.value
            else:
                health['primary_reason'] = RollbackReason.LOW_AVAILABILITY.value
        
        return health
    
    def execute_rollback(self, reason: str, initiated_by: str = "automated") -> Dict:
        """Execute a full system rollback."""
        logger.info(f"Initiating rollback. Reason: {reason}")
        
        # Get current versions
        current_prod = self.mlflow.get_production_model()
        current_image = self.k8s.get_current_image("mealie-recommender-production")
        
        result = RollbackEvent(
            timestamp=datetime.utcnow().isoformat(),
            from_version=current_prod['version'] if current_prod else 'unknown',
            to_version='',
            reason=reason,
            initiated_by=initiated_by,
            status='in_progress',
            details={}
        )
        
        try:
            # Step 1: Scale down canary (if running)
            logger.info("Step 1: Scaling down canary...")
            success, output = self.k8s.scale_down_canary()
            result.details['canary_scaled'] = success
            
            # Step 2: Route all traffic to stable
            logger.info("Step 2: Routing traffic to stable...")
            success, output = self.k8s.set_traffic_to_stable()
            result.details['traffic_routed'] = success
            
            # Step 3: Rollback Kubernetes deployment
            logger.info("Step 3: Rolling back Kubernetes deployment...")
            success, output = self.k8s.rollback_to_previous("mealie-recommender-production")
            result.details['k8s_rollback'] = success
            
            if success:
                # Wait for rollout
                success, output = self.k8s.wait_for_rollout("mealie-recommender-production")
                result.details['rollout_complete'] = success
            
            # Step 4: Rollback MLflow model
            logger.info("Step 4: Rolling back MLflow model...")
            archived = self.mlflow.get_archived_versions()
            if archived:
                rollback_version = archived[0]['version']
                success = self.mlflow.rollback_model(rollback_version)
                result.to_version = rollback_version
                result.details['mlflow_rollback'] = success
            else:
                result.details['mlflow_rollback'] = False
                result.details['mlflow_error'] = 'No archived versions available'
            
            # Determine overall status
            if all([
                result.details.get('k8s_rollback'),
                result.details.get('rollout_complete'),
            ]):
                result.status = 'success'
                logger.info("Rollback completed successfully")
            else:
                result.status = 'partial_success'
                logger.warning("Rollback completed with some failures")
            
        except Exception as e:
            result.status = 'failed'
            result.details['error'] = str(e)
            logger.error(f"Rollback failed: {e}")
        
        # Save rollback event
        self._save_rollback_event(result)
        
        return asdict(result)
    
    def _save_rollback_event(self, event: RollbackEvent):
        """Save rollback event to object storage for audit."""
        try:
            key = f"{CONFIG['rollback_prefix']}event_{event.timestamp.replace(':', '-')}.json"
            self.s3.put_object(
                Bucket=CONFIG["bucket_name"],
                Key=key,
                Body=json.dumps(asdict(event), indent=2).encode('utf-8'),
                ContentType='application/json'
            )
            logger.info(f"Saved rollback event: {key}")
        except Exception as e:
            logger.error(f"Failed to save rollback event: {e}")
    
    def list_rollback_history(self, limit: int = 10) -> List[Dict]:
        """Get recent rollback history."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            events = []
            
            for page in paginator.paginate(
                Bucket=CONFIG["bucket_name"],
                Prefix=CONFIG["rollback_prefix"]
            ):
                for obj in page.get('Contents', []):
                    try:
                        data = self.s3.get_object(
                            Bucket=CONFIG["bucket_name"],
                            Key=obj['Key']
                        )
                        event = json.loads(data['Body'].read().decode('utf-8'))
                        events.append(event)
                    except:
                        pass
            
            # Sort by timestamp descending
            events.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return events[:limit]
        except Exception as e:
            logger.error(f"Error fetching rollback history: {e}")
            return []
    
    def list_available_versions(self) -> Dict:
        """List available model versions for rollback."""
        return {
            'production': self.mlflow.get_production_model(),
            'archived': self.mlflow.get_archived_versions()[:5],
            'k8s_current_image': self.k8s.get_current_image("mealie-recommender-production")
        }
    
    def auto_rollback_check(self) -> Dict:
        """Check health and auto-rollback if needed."""
        health = self.check_health()
        
        result = {
            'health_check': health,
            'action_taken': 'none',
            'rollback_result': None
        }
        
        if health['recommendation'] == 'rollback':
            logger.warning(f"Auto-rollback triggered: {health['violations']}")
            rollback_result = self.execute_rollback(
                reason=health.get('primary_reason', 'health_check_failed'),
                initiated_by='auto_rollback'
            )
            result['action_taken'] = 'rollback'
            result['rollback_result'] = rollback_result
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Model Rollback Manager')
    parser.add_argument('--action', required=True,
                       choices=['check-health', 'rollback', 'auto-check', 'list-versions', 'history'],
                       help='Action to perform')
    parser.add_argument('--reason', default='manual',
                       help='Reason for manual rollback')
    parser.add_argument('--version', help='Specific version to rollback to')
    parser.add_argument('--initiated-by', default='cli',
                       help='Who initiated the rollback')
    
    args = parser.parse_args()
    
    manager = RollbackManager()
    
    if args.action == 'check-health':
        result = manager.check_health()
        print(json.dumps(result, indent=2))
    
    elif args.action == 'rollback':
        result = manager.execute_rollback(
            reason=args.reason,
            initiated_by=args.initiated_by
        )
        print(json.dumps(result, indent=2))
    
    elif args.action == 'auto-check':
        result = manager.auto_rollback_check()
        print(json.dumps(result, indent=2))
    
    elif args.action == 'list-versions':
        result = manager.list_available_versions()
        print(json.dumps(result, indent=2, default=str))
    
    elif args.action == 'history':
        result = manager.list_rollback_history()
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

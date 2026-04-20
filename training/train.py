#!/usr/bin/env python
# coding: utf-8
# Assisted by Claude Sonnet

import os
import ast
import io
import time
import tqdm
import numpy as np
import pandas as pd
from collections import Counter

import boto3
from botocore.client import Config

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import roc_auc_score, average_precision_score

import mlflow

# Fix compatibility: PyTorch 2.8 removed torch.fx._symbolic_trace.List
import typing
import torch.fx._symbolic_trace as _st
if not hasattr(_st, 'List'):
    _st.List = typing.List
if not hasattr(_st, 'Dict'):
    _st.Dict = typing.Dict
if not hasattr(_st, 'Optional'):
    _st.Optional = typing.Optional

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "hidden_channels": 64,
    "num_epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 128,
    "min_ratings_per_user": 5,
    "positive_rating_threshold": 4,
    "top_k_ingredients": 500,
    "model_output_path": "./best_model.pt",
    "mlflow_experiment": "mealie-gnn-recommendations",
    "run_name": "v1_best",
    "bucket_name": "ObjStore_proj14",
    "endpoint_url": "https://chi.tacc.chameleoncloud.org:7480",
    "local_data_path": "./data",
    "model_s3_key": "training/best_model.pt",
    # Model quality gates - models must pass these to be registered
    "min_test_auc": 0.70,
    "min_test_ap": 0.65,
    "min_val_auc": 0.68,
    # MLflow Model Registry settings
    "model_registry_name": "mealie-recipe-recommender",
    # Kaggle dataset for automatic download
    "kaggle_dataset": "shuyangli94/food-com-recipes-and-user-interactions",
}
# ──────────────────────────────────────────────────────────────────────────────


def ensure_data_available(config):
    """
    Check if required data files exist. If not, download from Kaggle.
    """
    data_path = config["local_data_path"]
    required_files = [
        "PP_recipes.csv",
        "PP_users.csv",
        "interactions_train.csv",
        "interactions_validation.csv",
        "interactions_test.csv",
    ]
    
    # Check if all required files exist
    missing_files = []
    for f in required_files:
        filepath = os.path.join(data_path, f)
        if not os.path.exists(filepath):
            missing_files.append(f)
    
    if not missing_files:
        print(f"All required data files found in {data_path}")
        return True
    
    print(f"Missing data files: {missing_files}")
    print(f"Attempting to download dataset from Kaggle...")
    
    # Create data directory if needed
    os.makedirs(data_path, exist_ok=True)
    
    try:
        # Try importing kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Setup Kaggle credentials if not already configured
        kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
        kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
        
        if not os.path.exists(kaggle_json):
            # Check for credentials in environment variables
            kaggle_user = os.getenv("KAGGLE_USERNAME")
            kaggle_key = os.getenv("KAGGLE_KEY")
            
            if kaggle_user and kaggle_key:
                os.makedirs(kaggle_dir, exist_ok=True)
                import json
                with open(kaggle_json, 'w') as f:
                    json.dump({"username": kaggle_user, "key": kaggle_key}, f)
                # Set permissions (Windows doesn't require chmod 600)
                if os.name != 'nt':
                    os.chmod(kaggle_json, 0o600)
                print("Created kaggle.json from environment variables")
            else:
                print("ERROR: Kaggle credentials not found!")
                print("Please either:")
                print("  1. Create ~/.kaggle/kaggle.json with your credentials, OR")
                print("  2. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
                return False
        
        # Authenticate and download
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading dataset: {config['kaggle_dataset']}")
        api.dataset_download_files(
            config["kaggle_dataset"],
            path=data_path,
            unzip=True
        )
        print(f"Dataset downloaded and extracted to {data_path}")
        
        # Verify download
        still_missing = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
        if still_missing:
            print(f"WARNING: After download, still missing: {still_missing}")
            return False
        
        print("All required data files now available!")
        return True
        
    except ImportError:
        print("ERROR: kaggle package not installed!")
        print("Install with: pip install kaggle")
        return False
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        return False


def check_model_quality_gates(test_auc, test_ap, best_val_auc, config):
    """
    Evaluate if model passes quality gates for registration.
    Returns (passed: bool, reasons: list of strings)
    """
    reasons = []
    passed = True
    
    if test_auc < config["min_test_auc"]:
        reasons.append(f"Test AUC {test_auc:.4f} < threshold {config['min_test_auc']}")
        passed = False
    if test_ap < config["min_test_ap"]:
        reasons.append(f"Test AP {test_ap:.4f} < threshold {config['min_test_ap']}")
        passed = False
    if best_val_auc < config["min_val_auc"]:
        reasons.append(f"Val AUC {best_val_auc:.4f} < threshold {config['min_val_auc']}")
        passed = False
    
    return passed, reasons


def register_model_if_quality_passes(test_auc, test_ap, best_val_auc, config):
    """
    Register model in MLflow Model Registry only if it passes quality gates.
    Returns registration status and details.
    """
    passed, reasons = check_model_quality_gates(test_auc, test_ap, best_val_auc, config)
    
    if not passed:
        print(f"\nMODEL REJECTED - Failed quality gates:")
        for reason in reasons:
            print(f"   - {reason}")
        mlflow.log_param("model_registered", False)
        mlflow.log_param("rejection_reasons", "; ".join(reasons))
        return False, reasons
    
    print(f"\nMODEL PASSED quality gates - Registering in Model Registry...")
    
    # Register the model
    model_uri = f"runs:/{mlflow.active_run().info.run_id}/best_model.pt"
    
    try:
        # Register with MLflow Model Registry
        result = mlflow.register_model(
            model_uri=model_uri,
            name=config["model_registry_name"],
            tags={
                "test_auc": str(test_auc),
                "test_ap": str(test_ap),
                "val_auc": str(best_val_auc),
                "environment": os.getenv("DEPLOY_ENV", "staging"),
            }
        )
        mlflow.log_param("model_registered", True)
        mlflow.log_param("model_version", result.version)
        print(f"   Registered as version {result.version}")
        return True, result.version
    except Exception as e:
        print(f"   Warning: Could not register model: {e}")
        mlflow.log_param("model_registered", False)
        mlflow.log_param("registration_error", str(e))
        return False, [str(e)]


def get_s3_client(config):
    access_key = os.getenv("CHAMELEON_ACCESS_KEY")
    secret_key = os.getenv("CHAMELEON_SECRET_KEY")
    if not access_key or not secret_key:
        raise ValueError("CHAMELEON_ACCESS_KEY and CHAMELEON_SECRET_KEY env vars must be set!")
    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=config["endpoint_url"],
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


def get_latest_snapshot_prefix(s3, config):
    """Find the latest timestamped subfolder under train/ in the object store."""
    paginator = s3.get_paginator('list_objects_v2')
    subfolders = set()
    for page in paginator.paginate(Bucket=config["bucket_name"], Prefix="train/"):
        for obj in page.get('Contents', []):
            key = obj['Key']
            parts = key.split('/')
            # Expected structure: train/<timestamp>/filename.csv
            if len(parts) >= 3 and parts[1]:
                subfolders.add(parts[1])
    if not subfolders:
        raise Exception("No timestamped subfolders found in object storage under train/")
    latest = sorted(subfolders)[-1]
    print(f"Latest object store snapshot: train/{latest}/")
    return f"train/{latest}/"


def read_csv_from_s3(s3, bucket, key):
    """Download a single CSV from object storage and return as DataFrame."""
    print(f"  Downloading: {key}")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


def save_model_to_s3(s3, config):
    print(f"Uploading model to object storage: {config['model_s3_key']}")
    s3.upload_file(config["model_output_path"], config["bucket_name"], config["model_s3_key"])
    print(f"Model uploaded to {config['bucket_name']}/{config['model_s3_key']}")


def load_data(config):
    print("Loading data...")
    try:
        s3 = get_s3_client(config)
        print("Connected to object storage successfully!")
        prefix = get_latest_snapshot_prefix(s3, config)
        bucket = config["bucket_name"]

        # List all files in the latest snapshot
        paginator = s3.get_paginator('list_objects_v2')
        snapshot_files = {}
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get('Contents', []):
                filename = obj['Key'].split('/')[-1]
                if filename.endswith('.csv'):
                    snapshot_files[filename] = obj['Key']

        print(f"Found {len(snapshot_files)} files in snapshot: {list(snapshot_files.keys())}")

        # Load files from object store, fall back to local for missing ones
        data_path = config["local_data_path"]

        if 'PP_recipes.csv' in snapshot_files:
            recipes_df = read_csv_from_s3(s3, bucket, snapshot_files['PP_recipes.csv'])
        else:
            recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")

        if 'PP_users.csv' in snapshot_files:
            users_df = read_csv_from_s3(s3, bucket, snapshot_files['PP_users.csv'])
        else:
            users_df = pd.read_csv(f"{data_path}/PP_users.csv")

        if 'interactions_train.csv' in snapshot_files:
            interactions_train_df = read_csv_from_s3(s3, bucket, snapshot_files['interactions_train.csv'])
        else:
            # Check for older single-CSV format (e.g. train_20260407_0422.csv)
            train_csvs = [k for k in snapshot_files if k.startswith('train_')]
            if train_csvs:
                interactions_train_df = read_csv_from_s3(s3, bucket, snapshot_files[train_csvs[0]])
            else:
                interactions_train_df = pd.read_csv(f"{data_path}/interactions_train.csv")

        if 'interactions_validation.csv' in snapshot_files:
            interactions_val_df = read_csv_from_s3(s3, bucket, snapshot_files['interactions_validation.csv'])
        else:
            interactions_val_df = pd.read_csv(f"{data_path}/interactions_validation.csv")

        if 'interactions_test.csv' in snapshot_files:
            interactions_test_df = read_csv_from_s3(s3, bucket, snapshot_files['interactions_test.csv'])
        else:
            print("  interactions_test.csv not in snapshot, using local file")
            interactions_test_df = pd.read_csv(f"{data_path}/interactions_test.csv")

        # Rename columns if using Junhao's pipeline format
        for df_name, df in [('train', interactions_train_df), ('val', interactions_val_df), ('test', interactions_test_df)]:
            if 'user_id' in df.columns and 'u' not in df.columns:
                df.rename(columns={'user_id': 'u', 'recipe_id': 'i'}, inplace=True)
                print(f"  Renamed columns in {df_name}: user_id/recipe_id -> u/i")

        print(f"Data loaded from object store (snapshot: {prefix})")

    except Exception as e:
        print(f"Object storage failed: {e}")
        print("Falling back to local CSV files...")
        data_path = config["local_data_path"]
        recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
        users_df = pd.read_csv(f"{data_path}/PP_users.csv")
        interactions_train_df = pd.read_csv(f"{data_path}/interactions_train.csv")
        interactions_val_df = pd.read_csv(f"{data_path}/interactions_validation.csv")
        interactions_test_df = pd.read_csv(f"{data_path}/interactions_test.csv")

    threshold = config["positive_rating_threshold"]
    interactions_train_df = interactions_train_df[interactions_train_df['rating'] >= threshold].reset_index(drop=True)
    interactions_val_df = interactions_val_df[interactions_val_df['rating'] >= threshold].reset_index(drop=True)
    interactions_test_df = interactions_test_df[interactions_test_df['rating'] >= threshold].reset_index(drop=True)

    interactions_df = pd.concat([interactions_train_df, interactions_val_df, interactions_test_df], ignore_index=True)

    rating_counts = interactions_df.groupby('u').size()
    active_users = set(rating_counts[rating_counts >= config["min_ratings_per_user"]].index)
    before_count = len(interactions_df)
    interactions_df = interactions_df[interactions_df['u'].isin(active_users)].reset_index(drop=True)
    interactions_train_df = interactions_train_df[interactions_train_df['u'].isin(active_users)].reset_index(drop=True)
    interactions_val_df = interactions_val_df[interactions_val_df['u'].isin(active_users)].reset_index(drop=True)
    interactions_test_df = interactions_test_df[interactions_test_df['u'].isin(active_users)].reset_index(drop=True)

    print(f"Recipes: {len(recipes_df)}")
    print(f"Users (with >= {config['min_ratings_per_user']} ratings): {len(active_users)}")
    print(f"Positive interactions: {len(interactions_df)} (dropped {before_count - len(interactions_df)} from sparse users)")
    print(f"  Train: {len(interactions_train_df)}, Val: {len(interactions_val_df)}, Test: {len(interactions_test_df)}")

    if len(active_users) == 0:
        raise Exception("No active users found! Check data pipeline.")

    return recipes_df, users_df, interactions_df, interactions_train_df, interactions_val_df, interactions_test_df


def build_features(recipes_df, users_df, config):
    print("Building features...")
    recipes_df = recipes_df.sort_values('i').reset_index(drop=True)
    recipe_techniques = np.array(recipes_df['techniques'].apply(ast.literal_eval).tolist())
    calorie_onehot = pd.get_dummies(recipes_df['calorie_level'], prefix='calorie').values

    TOP_K = config["top_k_ingredients"]
    parsed_ingredient_ids = recipes_df['ingredient_ids'].apply(ast.literal_eval).tolist()
    ingredient_counter = Counter(ing for ids in parsed_ingredient_ids for ing in ids)
    top_ingredients = [ing_id for ing_id, _ in ingredient_counter.most_common(TOP_K)]
    ingredient_to_idx = {ing_id: idx for idx, ing_id in enumerate(top_ingredients)}

    ingredient_multihot = np.zeros((len(recipes_df), TOP_K), dtype=np.float32)
    for row_idx, ings in enumerate(parsed_ingredient_ids):
        for ing_id in ings:
            if ing_id in ingredient_to_idx:
                ingredient_multihot[row_idx, ingredient_to_idx[ing_id]] = 1.0

    recipe_features = np.hstack([recipe_techniques, calorie_onehot, ingredient_multihot])
    recipe_feat = torch.from_numpy(recipe_features).to(torch.float)

    users_df = users_df.sort_values('u').reset_index(drop=True)
    user_techniques = np.array(users_df['techniques'].apply(ast.literal_eval).tolist())
    user_feat = torch.from_numpy(user_techniques).to(torch.float)

    print(f"Recipe features shape: {recipe_feat.shape}")
    print(f"User features shape: {user_feat.shape}")

    return recipe_feat, user_feat, recipe_features.shape[1]


def build_graph(recipes_df, users_df, interactions_df, recipe_feat, user_feat):
    print("Building graph...")
    num_users = len(user_feat)
    num_recipes = len(recipe_feat)

    # Create contiguous node ID mapping for recipes and users
    unique_recipe_ids = sorted(recipes_df['i'].unique())
    unique_user_ids = sorted(users_df['u'].unique())
    recipe_id_map = {old: new for new, old in enumerate(unique_recipe_ids)}
    user_id_map = {old: new for new, old in enumerate(unique_user_ids)}

    # Remap interaction IDs to contiguous indices
    mapped_users = interactions_df['u'].map(user_id_map)
    mapped_recipes = interactions_df['i'].map(recipe_id_map)
    valid_mask = mapped_users.notna() & mapped_recipes.notna()
    mapped_users = mapped_users[valid_mask].astype(int)
    mapped_recipes = mapped_recipes[valid_mask].astype(int)

    ratings_user_id = torch.from_numpy(mapped_users.values)
    ratings_recipe_id = torch.from_numpy(mapped_recipes.values)
    edge_index_user_to_recipe = torch.stack([ratings_user_id, ratings_recipe_id], dim=0)

    data = HeteroData()
    data["user"].node_id = torch.arange(num_users)
    data["recipe"].node_id = torch.arange(num_recipes)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = edge_index_user_to_recipe
    data = T.ToUndirected()(data)

    dropped = (~valid_mask).sum()
    if dropped > 0:
        print(f"  Dropped {dropped} interactions with unmapped IDs")
    print(f"Graph built: {num_users} users, {num_recipes} recipes, {edge_index_user_to_recipe.shape[1]} edges")
    return data, num_users, num_recipes


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_recipe: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_recipe = x_recipe[edge_label_index[1]]
        return (edge_feat_user * edge_feat_recipe).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels, recipe_feat_dim, data):
        super().__init__()
        self.user_lin = torch.nn.Linear(58, hidden_channels)
        self.recipe_lin = torch.nn.Linear(recipe_feat_dim, hidden_channels)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, hidden_channels)
        self.recipe_emb = torch.nn.Embedding(data["recipe"].num_nodes, hidden_channels)
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
            "recipe": self.recipe_lin(data["recipe"].x) + self.recipe_emb(data["recipe"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["recipe"],
            data["user", "rates", "recipe"].edge_label_index,
        )
        return pred


def build_loaders(data, config):
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "recipe"),
        rev_edge_types=("recipe", "rev_rates", "user"),
    )
    train_data, val_data, test_data = transform(data)

    edge_label_index = train_data["user", "rates", "recipe"].edge_label_index
    edge_label = train_data["user", "rates", "recipe"].edge_label

    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=2.0,
        edge_label_index=(("user", "rates", "recipe"), edge_label_index),
        edge_label=edge_label,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    val_edge_label_index = val_data["user", "rates", "recipe"].edge_label_index
    val_edge_label = val_data["user", "rates", "recipe"].edge_label
    val_loader = LinkNeighborLoader(
        data=val_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "rates", "recipe"), val_edge_label_index),
        edge_label=val_edge_label,
        batch_size=3 * config["batch_size"],
        shuffle=False,
    )

    test_edge_label_index = test_data["user", "rates", "recipe"].edge_label_index
    test_edge_label = test_data["user", "rates", "recipe"].edge_label
    test_loader = LinkNeighborLoader(
        data=test_data,
        num_neighbors=[20, 10],
        edge_label_index=(("user", "rates", "recipe"), test_edge_label_index),
        edge_label=test_edge_label,
        batch_size=3 * config["batch_size"],
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for sampled_data in loader:
            sampled_data = sampled_data.to(device)
            pred = model(sampled_data)
            target = sampled_data["user", "rates", "recipe"].edge_label
            preds.append(pred.cpu())
            targets.append(target.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    auc = roc_auc_score(targets, preds)
    ap = average_precision_score(targets, preds)
    return auc, ap


def train(config):
    # Ensure data is available (download from Kaggle if missing)
    if not ensure_data_available(config):
        print("ERROR: Cannot proceed without data files!")
        print("Please download the dataset manually or configure Kaggle credentials.")
        return None
    
    mlflow.set_experiment(config["mlflow_experiment"])

    recipes_df, users_df, interactions_df, interactions_train_df, interactions_val_df, interactions_test_df = load_data(config)
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df, config)
    data, num_users, num_recipes = build_graph(recipes_df, users_df, interactions_df, recipe_feat, user_feat)
    train_loader, val_loader, test_loader = build_loaders(data, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with mlflow.start_run(run_name=config["run_name"]):
        mlflow.log_param("model", "GraphSAGE")
        mlflow.log_param("hidden_channels", config["hidden_channels"])
        mlflow.log_param("num_epochs", config["num_epochs"])
        mlflow.log_param("learning_rate", config["learning_rate"])
        mlflow.log_param("batch_size", config["batch_size"])
        mlflow.log_param("min_ratings_per_user", config["min_ratings_per_user"])
        mlflow.log_param("positive_rating_threshold", config["positive_rating_threshold"])
        mlflow.log_param("device", str(device))
        mlflow.log_param("cpu_count", os.cpu_count())
        mlflow.log_param("gpu_available", torch.cuda.is_available())
        mlflow.log_param("gpu_name", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
        mlflow.log_param("data_source", "object_storage" if os.getenv("CHAMELEON_ACCESS_KEY") else "local_csv")

        model = Model(
            hidden_channels=config["hidden_channels"],
            recipe_feat_dim=recipe_feat_dim,
            data=data
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        start = time.time()
        best_val_auc = 0
        best_loss = 0

        for epoch in range(1, config["num_epochs"] + 1):
            epoch_start = time.time()
            model.train()
            total_loss = total_examples = 0

            for sampled_data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch:02d}"):
                optimizer.zero_grad()
                sampled_data = sampled_data.to(device)
                pred = model(sampled_data)
                ground_truth = sampled_data["user", "rates", "recipe"].edge_label
                loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
                loss.backward()
                optimizer.step()
                total_loss += float(loss) * pred.numel()
                total_examples += pred.numel()

            val_auc, val_ap = evaluate(model, val_loader, device)
            epoch_loss = total_loss / total_examples
            epoch_time = round(time.time() - epoch_start, 2)

            mlflow.log_metric("val_auc", val_auc, step=epoch)
            mlflow.log_metric("val_ap", val_ap, step=epoch)
            mlflow.log_metric("loss", epoch_loss, step=epoch)
            mlflow.log_metric("epoch_time_sec", epoch_time, step=epoch)

            print(f"Epoch {epoch:02d} | Loss: {epoch_loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {val_ap:.4f} | Time: {epoch_time}s")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), config["model_output_path"])
                best_loss = epoch_loss

        train_time = time.time() - start
        test_auc, test_ap = evaluate(model, test_loader, device)

        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_ap", test_ap)
        mlflow.log_metric("train_time_sec", round(train_time, 2))
        mlflow.log_artifact(config["model_output_path"])

        try:
            s3 = get_s3_client(config)
            save_model_to_s3(s3, config)
            mlflow.log_param("model_s3_path", f"{config['bucket_name']}/{config['model_s3_key']}")
        except Exception as e:
            print(f"Warning: Could not upload model to object storage: {e}")

        # Save training metrics for monitoring dashboards
        import json
        metrics_path = os.path.join(os.path.dirname(config["model_output_path"]) or ".", "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({
                "test_auc": round(test_auc, 4),
                "test_ap": round(test_ap, 4),
                "best_val_auc": round(best_val_auc, 4),
                "training_loss": round(best_loss, 4),
                "training_duration_seconds": round(train_time, 2),
                "quality_gate_passed": 1 if test_auc >= config["min_test_auc"] and test_ap >= config["min_test_ap"] else 0,
                "model_version": 1,
            }, f)
        print(f"Training metrics saved to: {metrics_path}")

        print(f"\nTraining done!")
        print(f"Best Val AUC: {best_val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        print(f"Train time: {train_time:.2f}s")
        print(f"Model saved to: {config['model_output_path']}")
        
        # Check quality gates and register model if passed
        registered, result = register_model_if_quality_passes(
            test_auc, test_ap, best_val_auc, config
        )
        
        return {
            "test_auc": test_auc,
            "test_ap": test_ap,
            "best_val_auc": best_val_auc,
            "train_time": train_time,
            "model_registered": registered,
            "registration_result": result,
        }


if __name__ == "__main__":
    train(CONFIG)
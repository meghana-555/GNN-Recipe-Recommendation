import os
import io
import time
import ast
import numpy as np
import pandas as pd
import boto3
import requests
from collections import Counter
from botocore.client import Config
from sqlalchemy import create_engine, text

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

# PyTorch Backward Compatibility Fixes for SAGEConv
import typing
import torch.fx._symbolic_trace as _st
if not hasattr(_st, 'List'):
    _st.List = typing.List
if not hasattr(_st, 'Dict'):
    _st.Dict = typing.Dict
if not hasattr(_st, 'Optional'):
    _st.Optional = typing.Optional

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

# --- 🧠 PYTORCH MODEL ARCHITECTURE (Exact replication from Training Node) ---
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
        return x_dict

# --- 📊 FEATURE ENGINEERING FOR INFERENCE ---
def build_features(recipes_df, users_df):
    print("Building inference features from base CSVs...")
    
    # Process Recipes
    recipes_df = recipes_df.sort_values('i').reset_index(drop=True)
    recipe_techniques = np.array(recipes_df['techniques'].apply(ast.literal_eval).tolist())
    calorie_onehot = pd.get_dummies(recipes_df['calorie_level'], prefix='calorie').values
    
    TOP_K = 500
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
    
    # Process Users
    users_df = users_df.sort_values('u').reset_index(drop=True)
    user_techniques = np.array(users_df['techniques'].apply(ast.literal_eval).tolist())
    user_feat = torch.from_numpy(user_techniques).to(torch.float)
    
    return recipe_feat, user_feat, recipe_features.shape[1]

# --- 🚀 PREDICTION ENGINE ---
def execute_gnn_inference(s3, registry_df, num_recipes=7):
    print("--- 2. Fetching Weights & Executing GNN PyTorch Inference ---")
    recipe_registry = registry_df[registry_df['entity_type'] == 'recipe']
    user_registry = registry_df[registry_df['entity_type'] == 'user']
    
    if recipe_registry.empty:
        print("Registry empty. No recipes mapped.")
        return [], None
    
    if user_registry.empty:
        # Cold-start fallback: no Mealie users have interacted yet.
        # Use a default Kaggle baseline user for initial global recommendations.
        target_user_int = 0
        target_user_uuid = None
        print(f"No Mealie users in registry. Falling back to Kaggle User Node {target_user_int} for global recommendations.")
    else:
        target_user_int = int(user_registry['ml_native_id'].max())
        target_user_uuid = user_registry[user_registry['ml_native_id'] == target_user_int]['mealie_uuid'].iloc[0]
        print(f"Locked Target Inference Identity: [User Node {target_user_int}]")
    
    # 1. Load Parquets
    data_path = "/app/local_data" if os.path.exists('/.dockerenv') else "local_data"
    recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
    users_df = pd.read_csv(f"{data_path}/PP_users.csv")
    
    # 2. Build Tensors
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df)
    
    # 3. Pull & Peek S3 Model Weights FIRST to dynamically adjust tensor dimensions
    print("Downloading neural weights (best_model.pt) from S3...")
    device = torch.device('cpu')
    model_path = "/tmp/best_model.pt"
    s3.download_file(BUCKET_NAME, "training/best_model.pt", model_path)
    state_dict = torch.load(model_path, map_location=device)
    
    # Introspect trained dimensions dynamically
    num_users_checkpoint = state_dict['user_emb.weight'].shape[0]
    num_recipes_checkpoint = state_dict['recipe_emb.weight'].shape[0]
    print(f"Aligning Graph Tensors to strict Model Checkpoint boundaries (U: {num_users_checkpoint}, R: {num_recipes_checkpoint})...")
    
    # Zero-Pad local feature arrays to avoid PyTorch Parameter Mismatch exception
    if recipe_feat.shape[0] < num_recipes_checkpoint:
        pad_size = num_recipes_checkpoint - recipe_feat.shape[0]
        recipe_feat = torch.cat([recipe_feat, torch.zeros(pad_size, recipe_feat.shape[1])], dim=0)
    
    if user_feat.shape[0] < num_users_checkpoint:
        pad_size = num_users_checkpoint - user_feat.shape[0]
        user_feat = torch.cat([user_feat, torch.zeros(pad_size, user_feat.shape[1])], dim=0)

    # Construct base evaluation graph (Disconnected for pure isolated Cartesian Scoring)
    data = HeteroData()
    data["user"].node_id = torch.arange(num_users_checkpoint)
    data["recipe"].node_id = torch.arange(num_recipes_checkpoint)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data = T.ToUndirected()(data)

    model = Model(hidden_channels=64, recipe_feat_dim=recipe_feat_dim, data=data).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    # 4. Neural Scoring Matrix (Dot Product logic)
    print("Executing Neural Forward Pass Cartesian product...")
    with torch.no_grad():
        x_dict = model(data.to(device))
        user_emb = x_dict["user"][target_user_int].unsqueeze(0)  # Shape (1, F)
        recipe_embs = x_dict["recipe"]                           # Shape (Num_Recipes, F)
        
        # Calculate scores against ALL available recipes
        scores = (user_emb * recipe_embs).sum(dim=-1)            # Shape (Num_Recipes)
        
        # Constrain to ONLY valid UUID recipes loaded in frontend Mealie DB
        valid_recipe_ints = recipe_registry['ml_native_id'].tolist()
        
        mask = torch.ones_like(scores, dtype=torch.bool)
        # Keep seeded recipes unmasked (valid)
        if valid_recipe_ints:
            mask[valid_recipe_ints] = False
        
        # Impose -inf to completely zero-out unseeded datasets
        scores[mask] = -float('inf')
        
        # 5. Extract Top K recommendations
        top_k_scores, top_k_indices = torch.topk(scores, k=num_recipes)
    
    predicted_recipe_ints = top_k_indices.cpu().numpy().tolist()
    print(f"🧠 ML Top-K Integers: {predicted_recipe_ints}")
    
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


def inject_tags_via_database(per_user_recs):
    """Inject one AI tag per user, each containing their 7 recommended recipes.
    
    per_user_recs: list of dicts with keys: display_name, recipe_slugs (ordered by rank)
    """
    print("--- 4. Direct Database Tag Injection (Per-User Personalized) ---")
    try:
        engine = create_engine(MEALIE_POSTGRES_URL)
        with engine.begin() as conn:
            # Get group ID for tag creation
            group_id = conn.execute(text("SELECT id FROM groups LIMIT 1")).fetchone()[0]
            
            # Clean up ALL old AI recommendation tags (idempotent)
            old_tags = conn.execute(text(
                "SELECT id FROM tags WHERE name LIKE :pattern"
            ), {"pattern": "🤖%"}).fetchall()
            for tag_row in old_tags:
                conn.execute(text("DELETE FROM recipes_to_tags WHERE tag_id = :tid"),
                           {"tid": tag_row[0]})
                conn.execute(text("DELETE FROM tags WHERE id = :tid"),
                           {"tid": tag_row[0]})
            
            total_injected = 0
            for user_rec in per_user_recs:
                display_name = user_rec['display_name']
                slugs = user_rec['recipe_slugs']
                
                # One tag per user
                import uuid
                tag_name = f"🤖 For {display_name}"
                tag_slug = f"ai-for-{display_name.lower().replace(' ','-')}"
                tag_id = str(uuid.uuid4())
                conn.execute(text(
                    "INSERT INTO tags (id, name, slug, group_id) "
                    "VALUES (:id, :name, :slug, :gid)"
                ), {"id": tag_id, "name": tag_name, "slug": tag_slug, "gid": group_id})
                
                # Associate all 7 recommended recipes to this one tag
                for slug in slugs:
                    recipe_row = conn.execute(text(
                        "SELECT id FROM recipes WHERE slug = :slug OR id::text = :slug LIMIT 1"
                    ), {"slug": str(slug)}).fetchone()
                    
                    if recipe_row:
                        conn.execute(text(
                            "INSERT INTO recipes_to_tags (recipe_id, tag_id) "
                            "VALUES (:rid, :tid) ON CONFLICT DO NOTHING"
                        ), {"rid": recipe_row[0], "tid": tag_id})
                        total_injected += 1
            
            print(f"🎉 Injected {total_injected} recipes across {len(per_user_recs)} user tags!")
    except Exception as e:
        print(f"\n❌ Database Tag Injection Failed: {e}")


def run_personalized_recommendations(s3, registry):
    """Run GNN inference for ALL Mealie users and create per-user tags."""
    recipe_registry = registry[registry['entity_type'] == 'recipe']
    user_registry = registry[registry['entity_type'] == 'user']
    
    if recipe_registry.empty:
        print("Registry empty. No recipes mapped.")
        return
    
    # Load model once
    data_path = "/app/local_data" if os.path.exists('/.dockerenv') else "local_data"
    recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
    users_df = pd.read_csv(f"{data_path}/PP_users.csv")
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df)
    
    print("Downloading neural weights (best_model.pt) from S3...")
    device = torch.device('cpu')
    model_path = "/tmp/best_model.pt"
    s3.download_file(BUCKET_NAME, "training/best_model.pt", model_path)
    state_dict = torch.load(model_path, map_location=device)
    
    num_users_checkpoint = state_dict['user_emb.weight'].shape[0]
    num_recipes_checkpoint = state_dict['recipe_emb.weight'].shape[0]
    print(f"Aligning Graph Tensors to strict Model Checkpoint boundaries (U: {num_users_checkpoint}, R: {num_recipes_checkpoint})...")
    
    if recipe_feat.shape[0] < num_recipes_checkpoint:
        pad = num_recipes_checkpoint - recipe_feat.shape[0]
        recipe_feat = torch.cat([recipe_feat, torch.zeros(pad, recipe_feat.shape[1])], dim=0)
    if user_feat.shape[0] < num_users_checkpoint:
        pad = num_users_checkpoint - user_feat.shape[0]
        user_feat = torch.cat([user_feat, torch.zeros(pad, user_feat.shape[1])], dim=0)
    
    data = HeteroData()
    data["user"].node_id = torch.arange(num_users_checkpoint)
    data["recipe"].node_id = torch.arange(num_recipes_checkpoint)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data = T.ToUndirected()(data)
    
    model = Model(hidden_channels=64, recipe_feat_dim=recipe_feat_dim, data=data).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print("Executing Neural Forward Pass Cartesian product...")
    with torch.no_grad():
        x_dict = model(data.to(device))
    
    valid_ints = recipe_registry['ml_native_id'].tolist()
    
    # Get all Mealie users from PostgreSQL
    engine = create_engine(MEALIE_POSTGRES_URL)
    with engine.connect() as conn:
        mealie_users = conn.execute(text(
            "SELECT id, full_name, username FROM users ORDER BY created_at"
        )).fetchall()
    
    per_user_recs = []
    for user_row in mealie_users:
        uid, full_name, username = str(user_row[0]), user_row[1], user_row[2]
        display_name = full_name or username or uid[:8]
        
        # Find GNN node for this user
        match = user_registry[user_registry['mealie_uuid'] == uid]
        if not match.empty:
            user_int = int(match.iloc[0]['ml_native_id'])
        else:
            # Cold-start: hash UUID to pick a diverse Kaggle proxy user
            user_int = hash(uid) % min(num_users_checkpoint, 25076)
        
        if user_int >= num_users_checkpoint:
            # Exceeds model capacity: hash to a valid diverse node
            user_int = hash(uid) % min(num_users_checkpoint, 25076)
        
        # Score all recipes for this user
        user_emb = x_dict["user"][user_int].unsqueeze(0)
        scores = (user_emb * x_dict["recipe"]).sum(dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        for vi in valid_ints:
            if vi < len(mask):
                mask[vi] = False
        scores[mask] = -float('inf')
        top_scores, top_idx = torch.topk(scores, k=min(7, len(valid_ints)))
        
        # Map ML ints back to Mealie slugs
        recipe_slugs = []
        for idx in top_idx.tolist():
            rmatch = recipe_registry[recipe_registry['ml_native_id'] == idx]
            if not rmatch.empty:
                recipe_slugs.append(rmatch.iloc[0]['mealie_uuid'])
        
        per_user_recs.append({
            'display_name': display_name,
            'recipe_slugs': recipe_slugs
        })
        print(f"  🧠 {display_name}: {len(recipe_slugs)} recommendations (node={user_int})")
    
    # Inject tags
    if per_user_recs:
        inject_tags_via_database(per_user_recs)


if __name__ == "__main__":
    s3 = get_s3_client()
    registry = pd.DataFrame()
    try:
        print("--- 1. Pulling Global ID Registry from S3 ---")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key="dataset/registry/id_mapping_registry.parquet")
        registry = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        pass
        
    run_personalized_recommendations(s3, registry)


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
    
    if recipe_registry.empty or user_registry.empty:
        print("Registry empty. Missing fundamental nodes.")
        return [], None
        
    target_user_int = int(user_registry['ml_native_id'].max())
    target_user_uuid = user_registry[user_registry['ml_native_id'] == target_user_int]['mealie_uuid'].iloc[0]
    print(f"Locked Target Inference Identity: [User Node {target_user_int}]")
    
    # 1. Load Parquets
    data_path = "/app/local_data" if os.path.exists('/.dockerenv') else "local_data"
    recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
    users_df = pd.read_csv(f"{data_path}/PP_users.csv")
    
    # 2. Build Tensors
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df)
    
    num_users = users_df['u'].max() + 1
    num_recipes = recipes_df['i'].max() + 1
    
    # Construct base evaluation graph (Disconnected for pure isolated Cartesian Scoring)
    # We pass empty edges for inference without propagating historical neighbor loops since it's zero-shot cold start testing.
    data = HeteroData()
    data["user"].node_id = torch.arange(num_users)
    data["recipe"].node_id = torch.arange(num_recipes)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = torch.empty((2, 0), dtype=torch.long)
    data = T.ToUndirected()(data)

    device = torch.device('cpu')
    model = Model(hidden_channels=64, recipe_feat_dim=recipe_feat_dim, data=data).to(device)
    
    # 3. Pull & Load Best S3 Model Weights
    print("Downloading neural weights (best_model.pt) from S3...")
    model_path = "/tmp/best_model.pt"
    s3.download_file(BUCKET_NAME, "dataset/training/best_model.pt", model_path)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 4. Neural Scoring Matrix (Dot Product logic)
    print("Executing Neural Forward Pass Cartesian product...")
    with torch.no_grad():
        x_dict = model(data.to(device))
        user_emb = x_dict["user"][target_user_int].unsqueeze(0)  # Shape (1, F)
        recipe_embs = x_dict["recipe"]                           # Shape (Num_Recipes, F)
        
        # Calculate scores against ALL available recipes
        scores = (user_emb * recipe_embs).sum(dim=-1)            # Shape (Num_Recipes)
        
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


def inject_tags_via_database(recipe_uuids):
    print("--- 4. Direct Database Tag Injection (Bypassing API Constraints) ---")
    tag_name = "🤖 AI Recommended"
    try:
        engine = create_engine(MEALIE_POSTGRES_URL)
        with engine.begin() as conn:
            # Clear old recommendations for idempotency in DB injections
            tag_query = text("SELECT id FROM tags WHERE name = :name LIMIT 1")
            result = conn.execute(tag_query, {"name": tag_name}).fetchone()
            if not result:
                print("❌ Master Tag not found in Database.")
                return
            tag_id = result[0]
            
            # Optional cleanup: Wipe the AI Recommendations cleanly before updating the new predicted set
            cleanup_query = text("DELETE FROM recipes_to_tags WHERE tag_id = :tag_id")
            conn.execute(cleanup_query, {"tag_id": tag_id})
            
            success_count = 0
            for r_slug in recipe_uuids:
                resolver_query = text("SELECT id FROM recipes WHERE slug = :slug OR id::text = :slug LIMIT 1")
                recipe_row = conn.execute(resolver_query, {"slug": str(r_slug)}).fetchone()
                if not recipe_row: continue
                db_recipe_uuid = recipe_row[0]
                
                query = text("""
                    INSERT INTO recipes_to_tags (recipe_id, tag_id)
                    VALUES (:recipe_id, :tag_id)
                    ON CONFLICT DO NOTHING
                """)
                conn.execute(query, {"recipe_id": db_recipe_uuid, "tag_id": str(tag_id)})
                success_count += 1
                
        print(f"🎉 OPERATION COMPLETE: Successfully injected {success_count} True ML recipes into AI Recommendations!")
    except Exception as e:
        print(f"\n❌ Database Connection Failed: {e}")


if __name__ == "__main__":
    if not MEALIE_TOKEN:
        print("ERROR: MEALIE_API_TOKEN environment variable not set. Please set it in .env")
        exit(1)
        
    s3 = get_s3_client()
    registry = pd.DataFrame()
    try:
        print("--- 1. Pulling Global ID Registry from S3 ---")
        obj = s3.get_object(Bucket=BUCKET_NAME, Key="dataset/registry/id_mapping_registry.parquet")
        registry = pd.read_parquet(io.BytesIO(obj['Body'].read()))
    except Exception as e:
        pass
        
    predicted_ints, target_user_uuid = execute_gnn_inference(s3, registry, num_recipes=7)
    if predicted_ints:
        predicted_uuids = map_predictions_to_uuid(registry, predicted_ints)
        inject_tags_via_database(predicted_uuids)

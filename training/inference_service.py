#!/usr/bin/env python
"""
Mealie Recipe Recommendation Inference Service

FastAPI service that loads the trained GraphSAGE model and serves
personalized recipe recommendations via REST API.
"""

import os
import ast
import time
import logging
from collections import Counter
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Prometheus metrics
try:
    from prometheus_client import Counter as PromCounter, Histogram, Gauge, generate_latest
    from fastapi.responses import Response
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
CONFIG = {
    "model_path": os.getenv("MODEL_PATH", "./best_model.pt"),
    "data_path": os.getenv("DATA_PATH", "./data"),
    "hidden_channels": 64,
    "top_k_ingredients": 500,
    "host": os.getenv("HOST", "0.0.0.0"),
    "port": int(os.getenv("PORT", "8000")),
}

# ─── MODEL CLASSES (must match train.py) ──────────────────────────────────────

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


# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────

class ModelState:
    """Holds loaded model, graph data, and recipe metadata."""
    def __init__(self):
        self.model = None
        self.data = None
        self.device = None
        self.recipes_df = None
        self.raw_recipes_df = None
        self.users_df = None
        self.interactions_df = None
        self.user_embeddings = None
        self.recipe_embeddings = None
        self.recipe_id_to_idx = {}
        self.user_id_to_idx = {}
        self.user_rated_recipes = {}
        self.num_users = 0
        self.num_recipes = 0
        self.recipe_feat_dim = 0
        self.loaded = False


state = ModelState()

# ─── PROMETHEUS METRICS ───────────────────────────────────────────────────────

if PROMETHEUS_AVAILABLE:
    RECOMMENDATIONS_SERVED = PromCounter(
        'recommendations_served_total',
        'Total number of recommendation requests served',
        ['status']
    )
    RECOMMENDATIONS_CLICKED = PromCounter(
        'recommendations_clicked_total',
        'Total number of recommendations clicked by users'
    )
    PREDICTION_LATENCY = Histogram(
        'model_prediction_latency_seconds',
        'Time to generate recommendations',
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
    )
    MODEL_LOADED = Gauge(
        'model_loaded',
        'Whether the model is loaded and ready (1=yes, 0=no)'
    )
    ACTIVE_USERS = Gauge(
        'active_users_total',
        'Total number of users in the model'
    )
    ACTIVE_RECIPES = Gauge(
        'active_recipes_total',
        'Total number of recipes in the model'
    )
    # Training metrics (loaded from training_metrics.json)
    TRAIN_TEST_AUC = Gauge('model_test_auc', 'Test AUC from last training run')
    TRAIN_TEST_AP = Gauge('model_test_ap', 'Test AP from last training run')
    TRAIN_VAL_AUC = Gauge('model_val_auc', 'Best validation AUC from last training run')
    TRAIN_LOSS = Gauge('model_training_loss', 'Training loss from last training run')
    TRAIN_DURATION = Gauge('model_training_duration_seconds', 'Training duration in seconds')
    QUALITY_GATE = Gauge('model_quality_gate_passed', 'Whether model passed quality gates (1=yes)')
    MODEL_VERSION = Gauge('model_registry_version', 'Current model version')


# ─── DATA & MODEL LOADING ────────────────────────────────────────────────────

def build_features(recipes_df, users_df, config):
    """Build feature tensors from dataframes."""
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

    return recipe_feat, user_feat, recipe_features.shape[1]


def load_model_and_data():
    """Load trained model and build graph for inference."""
    logger.info("Loading data...")
    data_path = CONFIG["data_path"]

    recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
    raw_recipes_path = f"{data_path}/RAW_recipes.csv"
    if os.path.exists(raw_recipes_path):
        raw_recipes_df = pd.read_csv(raw_recipes_path, usecols=['name', 'id', 'minutes', 'n_ingredients'])
        raw_recipes_df = raw_recipes_df.set_index('id')
    else:
        raw_recipes_df = None
    users_df = pd.read_csv(f"{data_path}/PP_users.csv")
    interactions_train_df = pd.read_csv(f"{data_path}/interactions_train.csv")
    interactions_val_df = pd.read_csv(f"{data_path}/interactions_validation.csv")
    interactions_test_df = pd.read_csv(f"{data_path}/interactions_test.csv")

    interactions_df = pd.concat([interactions_train_df, interactions_val_df, interactions_test_df], ignore_index=True)

    # Build features
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df, CONFIG)

    # Build graph with contiguous ID mapping (mirrors train.py)
    num_users = len(user_feat)
    num_recipes = len(recipe_feat)

    unique_recipe_ids = sorted(recipes_df['i'].unique())
    unique_user_ids = sorted(users_df['u'].unique())
    recipe_id_map = {old: new for new, old in enumerate(unique_recipe_ids)}
    user_id_map = {old: new for new, old in enumerate(unique_user_ids)}

    mapped_users = interactions_df['u'].map(user_id_map)
    mapped_recipes = interactions_df['i'].map(recipe_id_map)
    valid_mask = mapped_users.notna() & mapped_recipes.notna()
    mapped_users = mapped_users[valid_mask].astype(int)
    mapped_recipes = mapped_recipes[valid_mask].astype(int)

    ratings_user_id = torch.from_numpy(mapped_users.values)
    ratings_recipe_id = torch.from_numpy(mapped_recipes.values)
    edge_index = torch.stack([ratings_user_id, ratings_recipe_id], dim=0)

    data = HeteroData()
    data["user"].node_id = torch.arange(num_users)
    data["recipe"].node_id = torch.arange(num_recipes)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = edge_index
    data = T.ToUndirected()(data)

    # Build user→recipe interaction map using remapped IDs (for filtering already-rated)
    user_rated = {}
    for u_orig, r_orig in zip(interactions_df['u'], interactions_df['i']):
        uid = user_id_map.get(int(u_orig))
        rid = recipe_id_map.get(int(r_orig))
        if uid is None or rid is None:
            continue
        if uid not in user_rated:
            user_rated[uid] = set()
        user_rated[uid].add(rid)

    # Load model - read embedding sizes from checkpoint to handle data version mismatches
    device = torch.device('cpu')
    logger.info(f"Loading model from {CONFIG['model_path']}...")
    checkpoint = torch.load(CONFIG["model_path"], map_location=device, weights_only=False)
    ckpt_num_users = checkpoint["user_emb.weight"].shape[0]
    ckpt_num_recipes = checkpoint["recipe_emb.weight"].shape[0]
    logger.info(f"Checkpoint dimensions: {ckpt_num_users} users, {ckpt_num_recipes} recipes (local data: {num_users} users, {num_recipes} recipes)")

    # Override graph node counts to match checkpoint if they differ
    if ckpt_num_users != num_users or ckpt_num_recipes != num_recipes:
        logger.warning(f"Adjusting node counts from checkpoint: users {num_users}->{ckpt_num_users}, recipes {num_recipes}->{ckpt_num_recipes}")
        data["user"].node_id = torch.arange(ckpt_num_users)
        data["recipe"].node_id = torch.arange(ckpt_num_recipes)
        # Pad or truncate feature tensors to match checkpoint
        if ckpt_num_users > num_users:
            pad = torch.zeros(ckpt_num_users - num_users, data["user"].x.shape[1])
            data["user"].x = torch.cat([data["user"].x, pad], dim=0)
        else:
            data["user"].x = data["user"].x[:ckpt_num_users]
        if ckpt_num_recipes > num_recipes:
            pad = torch.zeros(ckpt_num_recipes - num_recipes, data["recipe"].x.shape[1])
            data["recipe"].x = torch.cat([data["recipe"].x, pad], dim=0)
        else:
            data["recipe"].x = data["recipe"].x[:ckpt_num_recipes]
        num_users = ckpt_num_users
        num_recipes = ckpt_num_recipes

    model = Model(
        hidden_channels=CONFIG["hidden_channels"],
        recipe_feat_dim=recipe_feat_dim,
        data=data
    ).to(device)

    model.load_state_dict(checkpoint)
    model.eval()

    # Precompute embeddings for fast inference
    logger.info("Precomputing embeddings...")
    with torch.no_grad():
        x_dict = {
            "user": model.user_lin(data["user"].x) + model.user_emb(data["user"].node_id),
            "recipe": model.recipe_lin(data["recipe"].x) + model.recipe_emb(data["recipe"].node_id),
        }
        x_dict = model.gnn(x_dict, data.edge_index_dict)

    # Store state
    state.model = model
    state.data = data
    state.device = device
    state.recipes_df = recipes_df
    state.raw_recipes_df = raw_recipes_df
    state.users_df = users_df
    state.interactions_df = interactions_df
    state.user_embeddings = x_dict["user"]
    state.recipe_embeddings = x_dict["recipe"]
    state.user_rated_recipes = user_rated
    state.num_users = num_users
    state.num_recipes = num_recipes
    state.recipe_feat_dim = recipe_feat_dim
    state.recipe_id_to_idx = {int(row['i']): idx for idx, row in recipes_df.iterrows()}
    state.user_id_to_idx = {int(row['u']): idx for idx, row in users_df.iterrows()}
    state.loaded = True

    if PROMETHEUS_AVAILABLE:
        MODEL_LOADED.set(1)
        ACTIVE_USERS.set(int(num_users))
        ACTIVE_RECIPES.set(int(num_recipes))

        # Load training metrics if available
        import json
        metrics_path = os.path.join(CONFIG["data_path"], "..", "training_metrics.json")
        if not os.path.exists(metrics_path):
            metrics_path = os.path.join(".", "training_metrics.json")
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    tm = json.load(f)
                TRAIN_TEST_AUC.set(tm.get("test_auc", 0))
                TRAIN_TEST_AP.set(tm.get("test_ap", 0))
                TRAIN_VAL_AUC.set(tm.get("best_val_auc", 0))
                TRAIN_LOSS.set(tm.get("training_loss", 0))
                TRAIN_DURATION.set(tm.get("training_duration_seconds", 0))
                QUALITY_GATE.set(tm.get("quality_gate_passed", 0))
                MODEL_VERSION.set(tm.get("model_version", 1))
                logger.info(f"Training metrics loaded: AUC={tm.get('test_auc')}, AP={tm.get('test_ap')}")
            except Exception as e:
                logger.warning(f"Could not load training metrics: {e}")
        else:
            logger.warning("training_metrics.json not found, training dashboard will be empty")

    logger.info(f"Model loaded: {num_users} users, {num_recipes} recipes")


# ─── FASTAPI APP ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_and_data()
    yield
    logger.info("Shutting down inference service")

app = FastAPI(
    title="Mealie Recipe Recommendation Service",
    description="Personalized recipe recommendations powered by GraphSAGE GNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── RESPONSE MODELS ─────────────────────────────────────────────────────────

class RecipeRecommendation(BaseModel):
    recipe_id: int
    name: str
    score: float
    calories: Optional[str] = None
    n_ingredients: Optional[int] = None
    minutes: Optional[int] = None


class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[RecipeRecommendation]
    model_version: str = "v1"
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_users: int
    num_recipes: int


class FeedbackRequest(BaseModel):
    user_id: int
    recipe_id: int
    action: str  # "click", "rate", "meal_plan", "dismiss"
    rating: Optional[float] = None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if state.loaded else "loading",
        model_loaded=state.loaded,
        num_users=state.num_users,
        num_recipes=state.num_recipes,
    )


@app.get("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(
    user_id: int = Query(..., description="User ID to get recommendations for"),
    top_k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    exclude_rated: bool = Query(True, description="Exclude already-rated recipes"),
):
    """Get personalized recipe recommendations for a user."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    start_time = time.time()

    try:
        # Validate user exists
        if user_id >= state.num_users or user_id < 0:
            raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        # Compute scores: dot product of user embedding with all recipe embeddings
        with torch.no_grad():
            user_emb = state.user_embeddings[user_id].unsqueeze(0)  # (1, hidden)
            scores = torch.matmul(user_emb, state.recipe_embeddings.T).squeeze(0)  # (num_recipes,)

        scores_np = scores.numpy()

        # Exclude already-rated recipes
        if exclude_rated and user_id in state.user_rated_recipes:
            for rated_id in state.user_rated_recipes[user_id]:
                if rated_id < len(scores_np):
                    scores_np[rated_id] = float('-inf')

        # Get top-k recipe indices
        top_indices = np.argsort(scores_np)[::-1][:top_k]

        # Build response with recipe metadata
        recommendations = []
        for idx in top_indices:
            if scores_np[idx] == float('-inf'):
                continue

            pp_row = state.recipes_df[state.recipes_df['i'] == idx]
            if len(pp_row) == 0:
                continue

            pp_row = pp_row.iloc[0]
            recipe_id_raw = int(pp_row['id'])

            # Look up metadata from RAW_recipes if available
            name = f'Recipe {recipe_id_raw}'
            n_ingredients = None
            minutes = None
            if state.raw_recipes_df is not None and recipe_id_raw in state.raw_recipes_df.index:
                raw_row = state.raw_recipes_df.loc[recipe_id_raw]
                name = str(raw_row['name'])
                n_ingredients = int(raw_row['n_ingredients']) if pd.notna(raw_row['n_ingredients']) else None
                minutes = int(raw_row['minutes']) if pd.notna(raw_row['minutes']) else None

            rec = RecipeRecommendation(
                recipe_id=recipe_id_raw,
                name=name,
                score=round(float(scores_np[idx]), 4),
                calories=str(pp_row.get('calorie_level', '')),
                n_ingredients=n_ingredients,
                minutes=minutes,
            )
            recommendations.append(rec)

        latency_ms = (time.time() - start_time) * 1000

        if PROMETHEUS_AVAILABLE:
            RECOMMENDATIONS_SERVED.labels(status="success").inc()
            PREDICTION_LATENCY.observe(latency_ms / 1000)

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            latency_ms=round(latency_ms, 2),
        )

    except HTTPException:
        raise
    except Exception as e:
        if PROMETHEUS_AVAILABLE:
            RECOMMENDATIONS_SERVED.labels(status="error").inc()
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """Record user feedback on a recommendation (for monitoring/retraining)."""
    if PROMETHEUS_AVAILABLE and feedback.action == "click":
        RECOMMENDATIONS_CLICKED.inc()

    logger.info(f"Feedback: user={feedback.user_id} recipe={feedback.recipe_id} action={feedback.action}")
    return {"status": "recorded", "feedback": feedback.dict()}


@app.get("/api/user/{user_id}/history")
async def get_user_history(user_id: int):
    """Get a user's rating history."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    rated = state.user_rated_recipes.get(user_id, set())
    return {
        "user_id": user_id,
        "num_rated": len(rated),
        "rated_recipe_ids": sorted(list(rated))[:100],  # Limit response size
    }


@app.get("/api/recipes/{recipe_id}")
async def get_recipe_info(recipe_id: int):
    """Get recipe metadata."""
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    recipe_row = state.recipes_df[state.recipes_df['i'] == recipe_id]
    if len(recipe_row) == 0:
        raise HTTPException(status_code=404, detail=f"Recipe {recipe_id} not found")

    recipe = recipe_row.iloc[0]
    return {
        "recipe_id": recipe_id,
        "name": str(recipe.get('name', '')),
        "minutes": int(recipe.get('minutes', 0)) if pd.notna(recipe.get('minutes')) else None,
        "n_ingredients": int(recipe.get('n_ingredients', 0)) if pd.notna(recipe.get('n_ingredients')) else None,
        "calorie_level": str(recipe.get('calorie_level', '')),
    }


@app.get("/api/stats")
async def get_stats():
    """Get model and service statistics."""
    return {
        "model_loaded": state.loaded,
        "num_users": state.num_users,
        "num_recipes": state.num_recipes,
        "total_interactions": len(state.interactions_df) if state.loaded else 0,
        "embedding_dim": CONFIG["hidden_channels"],
        "model_path": CONFIG["model_path"],
    }


if PROMETHEUS_AVAILABLE:
    @app.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        return Response(content=generate_latest(), media_type="text/plain")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=CONFIG["host"],
        port=CONFIG["port"],
        log_level="info",
    )

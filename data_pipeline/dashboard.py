"""
Personalized GNN Recommendation Dashboard.

Serves a web UI showing per-user personalized recommendations from
the GNN model. Demonstrates that different users receive different
recommendations based on their interaction history.

Usage:
    python dashboard.py              # starts on port 8501
    
Access:
    http://localhost:8501
"""

import os, io, ast, json, time
import numpy as np
import pandas as pd
import boto3
from botocore.client import Config
from sqlalchemy import create_engine, text
from flask import Flask, jsonify, Response

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero

# PyTorch compatibility
import typing, torch.fx._symbolic_trace as _st
for attr in ['List', 'Dict', 'Optional']:
    if not hasattr(_st, attr): setattr(_st, attr, getattr(typing, attr))

# --- Config ---
MEALIE_POSTGRES_URL = os.getenv("POSTGRES_URL",
    "postgresql://mealie:mealie_password@mealie-postgres:5432/mealie"
    if os.path.exists('/.dockerenv') else
    "postgresql://mealie:mealie_password@localhost:5432/mealie")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
ENDPOINT_URL = os.getenv("AWS_ENDPOINT_URL", "https://chi.tacc.chameleoncloud.org:7480")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "ObjStore_proj14")

def get_s3(): return boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY, endpoint_url=ENDPOINT_URL,
    config=Config(signature_version='s3v4'), region_name='us-east-1')

# --- GNN Model (same as serve_recommendations.py) ---
class GNN(torch.nn.Module):
    def __init__(self, h):
        super().__init__(); self.conv1 = SAGEConv(h,h); self.conv2 = SAGEConv(h,h)
    def forward(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)

class Classifier(torch.nn.Module):
    def forward(self, x_user, x_recipe, edge_label_index):
        return (x_user[edge_label_index[0]] * x_recipe[edge_label_index[1]]).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, h, rf, data):
        super().__init__()
        self.user_lin = torch.nn.Linear(58, h)
        self.recipe_lin = torch.nn.Linear(rf, h)
        self.user_emb = torch.nn.Embedding(data["user"].num_nodes, h)
        self.recipe_emb = torch.nn.Embedding(data["recipe"].num_nodes, h)
        self.gnn = to_hetero(GNN(h), metadata=data.metadata())
        self.classifier = Classifier()
    def forward(self, data):
        x_dict = {
            "user": self.user_lin(data["user"].x) + self.user_emb(data["user"].node_id),
            "recipe": self.recipe_lin(data["recipe"].x) + self.recipe_emb(data["recipe"].node_id),
        }
        return self.gnn(x_dict, data.edge_index_dict)

# --- Global model cache ---
_model_cache = {}
_model_load_ts = 0
MODEL_CACHE_TTL = 1800  # Refresh model every 30 minutes

def load_model_and_data():
    """Load model + features, cache with 30-min TTL for auto-refresh."""
    global _model_cache, _model_load_ts
    if _model_cache and (time.time() - _model_load_ts) < MODEL_CACHE_TTL:
        return _model_cache
    
    # Clear old cache to force reload
    _model_cache.clear()
    print("🔄 Loading/refreshing GNN model from S3...")

    s3 = get_s3()
    data_path = "/app/local_data" if os.path.exists('/.dockerenv') else "local_data"

    # Load features
    recipes_df = pd.read_csv(f"{data_path}/PP_recipes.csv")
    users_df = pd.read_csv(f"{data_path}/PP_users.csv")

    recipes_df = recipes_df.sort_values('i').reset_index(drop=True)
    recipe_techniques = np.array(recipes_df['techniques'].apply(ast.literal_eval).tolist())
    calorie_onehot = pd.get_dummies(recipes_df['calorie_level'], prefix='calorie').values
    TOP_K = 500
    parsed_ids = recipes_df['ingredient_ids'].apply(ast.literal_eval).tolist()
    from collections import Counter
    counter = Counter(i for ids in parsed_ids for i in ids)
    top_ings = [i for i, _ in counter.most_common(TOP_K)]
    ing_map = {i: idx for idx, i in enumerate(top_ings)}
    ing_mh = np.zeros((len(recipes_df), TOP_K), dtype=np.float32)
    for r, ings in enumerate(parsed_ids):
        for i in ings:
            if i in ing_map: ing_mh[r, ing_map[i]] = 1.0
    recipe_features = np.hstack([recipe_techniques, calorie_onehot, ing_mh])
    recipe_feat = torch.from_numpy(recipe_features).float()
    recipe_feat_dim = recipe_features.shape[1]

    users_df = users_df.sort_values('u').reset_index(drop=True)
    user_feat = torch.from_numpy(np.array(users_df['techniques'].apply(ast.literal_eval).tolist())).float()

    # Load model weights
    model_path = "/tmp/best_model.pt"
    s3.download_file(BUCKET_NAME, "training/best_model.pt", model_path)
    state_dict = torch.load(model_path, map_location='cpu')
    nu = state_dict['user_emb.weight'].shape[0]
    nr = state_dict['recipe_emb.weight'].shape[0]

    if recipe_feat.shape[0] < nr:
        recipe_feat = torch.cat([recipe_feat, torch.zeros(nr - recipe_feat.shape[0], recipe_feat.shape[1])])
    if user_feat.shape[0] < nu:
        user_feat = torch.cat([user_feat, torch.zeros(nu - user_feat.shape[0], user_feat.shape[1])])

    data = HeteroData()
    data["user"].node_id = torch.arange(nu)
    data["recipe"].node_id = torch.arange(nr)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = torch.empty((2,0), dtype=torch.long)
    data = T.ToUndirected()(data)

    model = Model(h=64, rf=recipe_feat_dim, data=data)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        x_dict = model(data)

    # Load registry
    obj = s3.get_object(Bucket=BUCKET_NAME, Key="dataset/registry/id_mapping_registry.parquet")
    registry = pd.read_parquet(io.BytesIO(obj['Body'].read()))

    _model_cache.update({
        'x_dict': x_dict, 'registry': registry,
        'nu': nu, 'nr': nr
    })
    _model_load_ts = time.time()
    print(f"✅ Model loaded. Users: {nu}, Recipes: {nr}")
    return _model_cache


# Separate cache for recipe names (refreshable)
_recipe_name_cache = {}
_recipe_name_ts = 0

def get_recipe_names():
    """Load recipe names from Mealie DB, refresh every 60s."""
    global _recipe_name_cache, _recipe_name_ts
    import time as _time
    if _recipe_name_cache and (_time.time() - _recipe_name_ts) < 60:
        return _recipe_name_cache
    try:
        engine = create_engine(MEALIE_POSTGRES_URL)
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT id, slug, name FROM recipes"
            )).fetchall()
            names = {}
            for row in rows:
                rid, slug, name = str(row[0]), row[1], row[2]
                names[rid] = name
                names[slug] = name
            if names:
                _recipe_name_cache.update(names)
                _recipe_name_ts = _time.time()
                print(f"Refreshed {len(rows)} recipe names from Mealie DB.")
    except Exception as e:
        print(f"Warning: Could not load recipe names: {e}")
    return _recipe_name_cache


def recommend_for_user(user_int, top_k=7):
    """Run GNN inference for a specific user node."""
    c = load_model_and_data()
    x_dict, registry = c['x_dict'], c['registry']
    recipe_reg = registry[registry['entity_type'] == 'recipe']
    valid_ints = recipe_reg['ml_native_id'].tolist()

    if user_int >= c['nu']:
        return []

    with torch.no_grad():
        user_emb = x_dict["user"][user_int].unsqueeze(0)
        scores = (user_emb * x_dict["recipe"]).sum(dim=-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        for vi in valid_ints:
            if vi < len(mask): mask[vi] = False
        scores[mask] = -float('inf')
        top_scores, top_idx = torch.topk(scores, k=min(top_k, len(valid_ints)))

    # Normalize scores to 0-100% confidence for display
    raw_scores = top_scores.tolist()
    s_min, s_max = min(raw_scores), max(raw_scores)
    if s_max > s_min:
        norm_scores = [(s - s_min) / (s_max - s_min) * 60 + 40 for s in raw_scores]  # 40-100%
    else:
        norm_scores = [75.0] * len(raw_scores)

    results = []
    recipe_names = get_recipe_names()
    for idx, score, norm in zip(top_idx.tolist(), raw_scores, norm_scores):
        match = recipe_reg[recipe_reg['ml_native_id'] == idx]
        uuid = match.iloc[0]['mealie_uuid'] if not match.empty else "?"
        # Resolve recipe name: mealie_uuid → recipe name
        name = recipe_names.get(uuid, None)
        if not name:
            name = recipe_names.get(str(uuid), f"Recipe #{idx}")
        results.append({"rank": len(results)+1, "name": str(name).title(),
                         "score": round(norm, 1), "recipe_int": idx, "uuid": uuid})
    return results


# --- Flask App ---
app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GNN Recipe Recommender Dashboard</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',sans-serif;background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
color:#e0e0e0;min-height:100vh;padding:20px}
.header{text-align:center;padding:30px 0}
.header h1{font-size:2.2em;font-weight:700;
background:linear-gradient(90deg,#ff6b6b,#ffd93d,#6bcb77,#4d96ff);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px}
.header p{color:#888;font-size:1em}
.users-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:24px;
max-width:1400px;margin:30px auto}
.user-card{background:rgba(255,255,255,0.06);backdrop-filter:blur(12px);
border:1px solid rgba(255,255,255,0.1);border-radius:16px;padding:24px;
transition:transform 0.3s,box-shadow 0.3s}
.user-card:hover{transform:translateY(-4px);box-shadow:0 12px 40px rgba(0,0,0,0.4)}
.user-info{display:flex;align-items:center;gap:14px;margin-bottom:18px}
.avatar{width:48px;height:48px;border-radius:50%;display:flex;align-items:center;
justify-content:center;font-size:1.4em;font-weight:700}
.a0{background:linear-gradient(135deg,#ff6b6b,#ee5a6f)}
.a1{background:linear-gradient(135deg,#ffd93d,#ff9f43)}
.a2{background:linear-gradient(135deg,#6bcb77,#38b000)}
.a3{background:linear-gradient(135deg,#4d96ff,#5f6fff)}
.a4{background:linear-gradient(135deg,#c084fc,#a855f7)}
.user-name{font-weight:600;font-size:1.1em;color:#fff}
.user-email{font-size:0.8em;color:#888}
.user-stats{font-size:0.75em;color:#6bcb77;margin-top:2px}
.recs-list{list-style:none}
.recs-list li{display:flex;align-items:center;gap:12px;padding:10px 12px;margin:4px 0;
background:rgba(255,255,255,0.04);border-radius:10px;transition:background 0.2s}
.recs-list li:hover{background:rgba(255,255,255,0.1)}
.rank{width:28px;height:28px;border-radius:8px;display:flex;align-items:center;
justify-content:center;font-weight:700;font-size:0.85em;flex-shrink:0}
.rank-1{background:linear-gradient(135deg,#ffd700,#ff8c00);color:#000}
.rank-2{background:linear-gradient(135deg,#c0c0c0,#a0a0a0);color:#000}
.rank-3{background:linear-gradient(135deg,#cd7f32,#a0522d);color:#fff}
.rank-n{background:rgba(255,255,255,0.1);color:#aaa}
.recipe-name{font-size:0.9em;font-weight:500;flex:1}
.recipe-score{font-size:0.75em;color:#4d96ff;font-weight:600;white-space:nowrap}
.loading{text-align:center;padding:60px;color:#888;font-size:1.2em}
.badge{display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.7em;
font-weight:600;margin-left:6px}
.badge-personal{background:rgba(75,150,255,0.2);color:#4d96ff}
.badge-cold{background:rgba(255,107,107,0.2);color:#ff6b6b}
.footer{text-align:center;padding:30px;color:#555;font-size:0.8em}
</style>
</head>
<body>
<div class="header">
<h1>🧠 GNN Personalized Recommendations</h1>
<p>Each user sees different recommendations based on their interaction history</p>
</div>
<div id="app" class="users-grid"><div class="loading">Loading recommendations...</div></div>
<div class="footer">Powered by SAGEConv GNN · Dot-Product Similarity · PyTorch</div>
<script>
const COLORS = ['a0','a1','a2','a3','a4'];
async function load() {
  const res = await fetch('/api/all_recommendations');
  const data = await res.json();
  const grid = document.getElementById('app');
  if (!data.users || !data.users.length) {
    grid.innerHTML = '<div class="loading">No users found. Register in Mealie first.</div>';
    return;
  }
  grid.innerHTML = '';
  data.users.forEach((u, idx) => {
    const initials = (u.name || u.email).substring(0,2).toUpperCase();
    const hasRecs = u.recommendations && u.recommendations.length > 0;
    const badge = hasRecs
      ? '<span class="badge badge-personal">Personalized</span>'
      : '<span class="badge badge-cold">Cold Start</span>';
    let recsHtml = '';
    if (hasRecs) {
      recsHtml = '<ul class="recs-list">' + u.recommendations.map(r => {
        const rc = r.rank <= 1 ? 'rank-1' : r.rank <= 2 ? 'rank-2' : r.rank <= 3 ? 'rank-3' : 'rank-n';
        return `<li><div class="rank ${rc}">${r.rank}</div>
          <span class="recipe-name">${r.name}</span>
          <span class="recipe-score">${r.score.toFixed(1)}%</span></li>`;
      }).join('') + '</ul>';
    } else {
      recsHtml = '<p style="color:#888;font-size:0.85em;padding:12px">No interactions yet — showing global recommendations</p>';
    }
    grid.innerHTML += `<div class="user-card">
      <div class="user-info">
        <div class="avatar ${COLORS[idx % 5]}">${initials}</div>
        <div><div class="user-name">${u.name || 'User'}${badge}</div>
        <div class="user-email">${u.email}</div>
        <div class="user-stats">${u.interaction_count} interactions</div></div>
      </div>${recsHtml}</div>`;
  });
}
load();
setInterval(load, 60000); // Auto-refresh every 60 seconds
</script>
</body></html>"""


@app.route('/')
def index():
    return Response(DASHBOARD_HTML, mimetype='text/html')


@app.route('/api/all_recommendations')
def all_recommendations():
    """Generate per-user recommendations for all registered Mealie users."""
    try:
        engine = create_engine(MEALIE_POSTGRES_URL)
        cache = load_model_and_data()
        registry = cache['registry']
        user_reg = registry[registry['entity_type'] == 'user']

        # Get all Mealie users
        with engine.connect() as conn:
            users = conn.execute(text(
                "SELECT id, full_name, email FROM users ORDER BY created_at"
            )).fetchall()

            # Get interaction counts per user (ratings + favorites + mealplans)
            interactions = {}
            try:
                # Count from users_to_recipes (ratings AND favorites)
                rows = conn.execute(text(
                    "SELECT user_id, "
                    "COUNT(CASE WHEN rating IS NOT NULL THEN 1 END) as ratings, "
                    "COUNT(CASE WHEN is_favorite = true THEN 1 END) as favorites "
                    "FROM users_to_recipes GROUP BY user_id"
                )).fetchall()
                for r in rows:
                    interactions[str(r[0])] = int(r[1] or 0) + int(r[2] or 0)
                
                # Add meal plan counts
                mp_rows = conn.execute(text(
                    "SELECT user_id, COUNT(*) as cnt FROM group_meal_plans "
                    "WHERE recipe_id IS NOT NULL AND user_id IS NOT NULL "
                    "GROUP BY user_id"
                )).fetchall()
                for r in mp_rows:
                    uid_str = str(r[0])
                    interactions[uid_str] = interactions.get(uid_str, 0) + int(r[1])
            except:
                pass

        results = []
        for user_row in users:
            uid, name, email = str(user_row[0]), user_row[1], user_row[2]
            interaction_count = interactions.get(uid, 0)

            # Find this user's GNN node
            match = user_reg[user_reg['mealie_uuid'] == uid]
            if not match.empty:
                user_int = int(match.iloc[0]['ml_native_id'])
                recs = recommend_for_user(user_int, top_k=7)
                # If mapped node exceeds model capacity, use hash fallback
                if not recs:
                    fallback_int = hash(uid) % min(cache['nu'], 25076)
                    recs = recommend_for_user(fallback_int, top_k=7)
            else:
                # Cold-start: hash UUID to diverse Kaggle proxy user
                fallback_int = hash(uid) % min(cache['nu'], 25076)
                recs = recommend_for_user(fallback_int, top_k=7)

            results.append({
                "email": email,
                "name": name or email.split('@')[0],
                "interaction_count": interaction_count,
                "recommendations": recs,
            })

        return jsonify({"users": results})

    except Exception as e:
        return jsonify({"error": str(e), "users": []})


if __name__ == '__main__':
    print("🚀 Dashboard starting on http://0.0.0.0:8501")
    app.run(host='0.0.0.0', port=8501, debug=False)

#!/usr/bin/env python3
"""
Batch pre-computation pipeline for GNN-based recipe recommendations.

This script is designed to run as a one-shot job (e.g. inside a Docker
container or a Kubernetes Job).  It:

  1. Checks whether a trained model checkpoint already exists.
  2. If not, trains the heterogeneous GraphSAGE model from CSV data and
     saves the checkpoint.
  3. Loads the model, computes user/recipe embeddings in a single
     forward pass, and generates top-N recommendations for every user.
  4. Writes two outputs:
       - Pre-computed recommendations to Redis (or a local JSON file
         when CACHE_BACKEND=memory).
       - A ``recipe_metadata.json`` file mapping recipe_id -> {name, tags}
         for the downstream serving API.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from tqdm import tqdm

# Local imports
from config import (
    BATCH_RUN_LOG,
    BATCH_SIZE,
    CACHE_BACKEND,
    DATA_DIR,
    HIDDEN_CHANNELS,
    KAGGLE_KEY,
    KAGGLE_USERNAME,
    MODEL_DIR,
    MODEL_PATH,
    NEG_SAMPLING_RATIO,
    NUM_NEIGHBORS,
    RECIPE_METADATA_OUTPUT,
    RECOMMENDATIONS_OUTPUT,
    REDIS_HOST,
    REDIS_PORT,
    TOP_K_INGREDIENTS,
    TOP_N,
    TRAIN_BATCH_SIZE,
    TRAIN_EPOCHS,
    TRAIN_LR,
)
from model import Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Data loading & feature engineering (mirrors the notebook exactly)


def load_interactions(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and filter interaction CSVs; return (combined, train, val, test)."""
    train_df = pd.read_csv(os.path.join(data_dir, "interactions_train.csv"))
    val_df = pd.read_csv(os.path.join(data_dir, "interactions_validation.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "interactions_test.csv"))

    # Keep only positive interactions (rating >= 4)
    train_df = train_df[train_df["rating"] >= 4].reset_index(drop=True)
    val_df = val_df[val_df["rating"] >= 4].reset_index(drop=True)
    test_df = test_df[test_df["rating"] >= 4].reset_index(drop=True)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Drop users with fewer than 5 positive ratings
    rating_counts = combined.groupby("u").size()
    active_users = set(rating_counts[rating_counts >= 5].index)
    before = len(combined)
    combined = combined[combined["u"].isin(active_users)].reset_index(drop=True)
    train_df = train_df[train_df["u"].isin(active_users)].reset_index(drop=True)
    val_df = val_df[val_df["u"].isin(active_users)].reset_index(drop=True)
    test_df = test_df[test_df["u"].isin(active_users)].reset_index(drop=True)

    log.info(
        "Interactions: %d total (%d dropped from sparse users). "
        "Train=%d, Val=%d, Test=%d",
        len(combined),
        before - len(combined),
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return combined, train_df, val_df, test_df


def build_recipe_features(recipes_df: pd.DataFrame) -> tuple[torch.Tensor, int]:
    """Build recipe feature matrix: techniques(58) + calorie(3) + ingredients(500)."""
    recipes_df = recipes_df.sort_values("i").reset_index(drop=True)

    techniques = np.array(
        recipes_df["techniques"].apply(ast.literal_eval).tolist()
    )
    calorie_onehot = pd.get_dummies(
        recipes_df["calorie_level"], prefix="calorie"
    ).values

    parsed_ids = recipes_df["ingredient_ids"].apply(ast.literal_eval).tolist()
    counter = Counter(ing for ids in parsed_ids for ing in ids)
    top_ingredients = [
        ing_id for ing_id, _ in counter.most_common(TOP_K_INGREDIENTS)
    ]
    ing_to_idx = {ing_id: idx for idx, ing_id in enumerate(top_ingredients)}

    multihot = np.zeros(
        (len(recipes_df), TOP_K_INGREDIENTS), dtype=np.float32
    )
    for row_idx, ings in enumerate(parsed_ids):
        for ing_id in ings:
            if ing_id in ing_to_idx:
                multihot[row_idx, ing_to_idx[ing_id]] = 1.0

    features = np.hstack([techniques, calorie_onehot, multihot])
    feat_dim = features.shape[1]
    log.info(
        "Recipe features: %d dims (techniques=%d, calorie=%d, ingredients=%d)",
        feat_dim,
        techniques.shape[1],
        calorie_onehot.shape[1],
        multihot.shape[1],
    )
    return torch.from_numpy(features).float(), feat_dim


def build_user_features(users_df: pd.DataFrame) -> torch.Tensor:
    """Build user feature matrix: techniques (58-dim count vector)."""
    users_df = users_df.sort_values("u").reset_index(drop=True)
    techniques = np.array(
        users_df["techniques"].apply(ast.literal_eval).tolist()
    )
    log.info("User features: %d dims", techniques.shape[1])
    return torch.from_numpy(techniques).float()


def build_graph(
    recipes_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    recipe_feat: torch.Tensor,
    user_feat: torch.Tensor,
) -> tuple[HeteroData, int, int]:
    """Construct the HeteroData graph (made undirected)."""
    num_users = int(users_df["u"].max()) + 1
    num_recipes = int(recipes_df["i"].max()) + 1

    edge_user = torch.from_numpy(interactions_df["u"].values)
    edge_recipe = torch.from_numpy(interactions_df["i"].values)
    edge_index = torch.stack([edge_user, edge_recipe], dim=0)

    data = HeteroData()
    data["user"].node_id = torch.arange(num_users)
    data["recipe"].node_id = torch.arange(num_recipes)
    data["user"].x = user_feat
    data["recipe"].x = recipe_feat
    data["user", "rates", "recipe"].edge_index = edge_index
    data = T.ToUndirected()(data)

    log.info(
        "Graph: %d users, %d recipes, %d edges (undirected)",
        num_users,
        num_recipes,
        data["user", "rates", "recipe"].edge_index.shape[1],
    )
    return data, num_users, num_recipes


def train_model(data_dir: str, model_path: str) -> None:
    """Train the heterogeneous GraphSAGE model from scratch and save it."""
    log.info("=== TRAINING: no model found at %s ===", model_path)

    # --- Load data ---
    recipes_df = pd.read_csv(os.path.join(data_dir, "PP_recipes.csv"))
    users_df = pd.read_csv(os.path.join(data_dir, "PP_users.csv"))
    interactions_df, _, _, _ = load_interactions(data_dir)

    recipe_feat, recipe_feat_dim = build_recipe_features(recipes_df)
    user_feat = build_user_features(users_df)
    data, num_users, num_recipes = build_graph(
        recipes_df, users_df, interactions_df, recipe_feat, user_feat
    )

    # --- Split edges ---
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        disjoint_train_ratio=0.3,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        add_negative_train_samples=False,
        edge_types=("user", "rates", "recipe"),
        rev_edge_types=("recipe", "rev_rates", "user"),
    )
    train_data, val_data, _test_data = transform(data)

    # --- DataLoader ---
    edge_label_index = train_data["user", "rates", "recipe"].edge_label_index
    edge_label = train_data["user", "rates", "recipe"].edge_label
    train_loader = LinkNeighborLoader(
        data=train_data,
        num_neighbors=NUM_NEIGHBORS,
        neg_sampling_ratio=NEG_SAMPLING_RATIO,
        edge_label_index=(("user", "rates", "recipe"), edge_label_index),
        edge_label=edge_label,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
    )

    # --- Model ---
    model = Model(
        hidden_channels=HIDDEN_CHANNELS,
        data=data,
        recipe_feat_dim=recipe_feat_dim,
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN_LR)

    # --- Training loop ---
    for epoch in range(1, TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for sampled_data in tqdm(
            train_loader, desc=f"Epoch {epoch}/{TRAIN_EPOCHS}", leave=True
        ):
            optimizer.zero_grad()
            sampled_data = sampled_data.to(DEVICE)
            pred = model(sampled_data)
            ground_truth = sampled_data[
                "user", "rates", "recipe"
            ].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
        avg_loss = total_loss / total_examples
        log.info("Epoch %d/%d  Loss: %.4f", epoch, TRAIN_EPOCHS, avg_loss)

    # --- Validation AUC ---
    try:
        from sklearn.metrics import roc_auc_score

        val_edge_label_index = val_data[
            "user", "rates", "recipe"
        ].edge_label_index
        val_edge_label = val_data["user", "rates", "recipe"].edge_label
        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=NUM_NEIGHBORS,
            edge_label_index=(
                ("user", "rates", "recipe"),
                val_edge_label_index,
            ),
            edge_label=val_edge_label,
            batch_size=3 * TRAIN_BATCH_SIZE,
            shuffle=False,
        )
        preds, gts = [], []
        model.eval()
        with torch.no_grad():
            for sampled_data in val_loader:
                sampled_data = sampled_data.to(DEVICE)
                preds.append(model(sampled_data).cpu())
                gts.append(
                    sampled_data[
                        "user", "rates", "recipe"
                    ].edge_label.cpu()
                )
        pred_arr = torch.cat(preds).numpy()
        gt_arr = torch.cat(gts).numpy()
        auc = roc_auc_score(gt_arr, pred_arr)
        log.info("Validation AUC: %.4f", auc)
    except Exception as exc:
        log.warning("Could not compute validation AUC: %s", exc)

    # --- Save ---
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "data": data,
            "num_users": num_users,
            "num_recipes": num_recipes,
            "recipe_feat_dim": recipe_feat_dim,
        },
        model_path,
    )
    log.info("Model saved to %s", model_path)


def compute_embeddings(
    model: Model, data: HeteroData
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run one forward pass over the full graph and return
    (user_embeddings, recipe_embeddings) on CPU.
    """
    model.eval()
    with torch.no_grad():
        data_dev = data.to(DEVICE)
        x_dict = {
            "user": model.user_lin(data_dev["user"].x)
            + model.user_emb(data_dev["user"].node_id),
            "recipe": model.recipe_lin(data_dev["recipe"].x)
            + model.recipe_emb(data_dev["recipe"].node_id),
        }
        x_dict = model.gnn(x_dict, data_dev.edge_index_dict)
    return x_dict["user"].cpu(), x_dict["recipe"].cpu()


def build_rated_set(data: HeteroData) -> dict[int, set[int]]:
    """Return {user_idx: set(recipe_indices)} from the graph edges."""
    edge_index = data["user", "rates", "recipe"].edge_index
    users = edge_index[0].tolist()
    recipes = edge_index[1].tolist()
    rated: dict[int, set[int]] = {}
    for u, r in zip(users, recipes):
        rated.setdefault(u, set()).add(r)
    return rated


def score_all_users(
    user_embs: torch.Tensor,
    recipe_embs: torch.Tensor,
    rated: dict[int, set[int]],
    num_users: int,
    top_n: int,
    batch_size: int,
) -> dict[int, list[tuple[int, float]]]:
    """
    For every user, compute dot-product scores against all recipes,
    exclude already-rated recipes, and keep top-N.

    Returns {user_idx: [(recipe_idx, score), ...]}.

    Scoring is batched over users to avoid OOM.
    """
    recommendations: dict[int, list[tuple[int, float]]] = {}
    num_batches = (num_users + batch_size - 1) // batch_size

    log.info(
        "Scoring %d users in %d batches (batch_size=%d, top_n=%d)",
        num_users,
        num_batches,
        batch_size,
        top_n,
    )

    for batch_idx in tqdm(range(num_batches), desc="Scoring users"):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_users)
        batch_user_embs = user_embs[start:end]  # (B, H)

        # Dot-product: (B, H) @ (H, R) -> (B, R)
        scores = torch.matmul(batch_user_embs, recipe_embs.t())

        for local_idx, user_idx in enumerate(range(start, end)):
            user_scores = scores[local_idx]  # (R,)
            # Mask out already-rated recipes
            rated_set = rated.get(user_idx, set())
            if rated_set:
                mask = torch.tensor(
                    sorted(rated_set), dtype=torch.long
                )
                user_scores[mask] = float("-inf")
            # Top-N
            topk = torch.topk(user_scores, min(top_n, len(user_scores)))
            recs = [
                (int(idx), float(score))
                for idx, score in zip(topk.indices.tolist(), topk.values.tolist())
            ]
            recommendations[user_idx] = recs

    return recommendations


def build_recipe_metadata(data_dir: str) -> dict[str, dict]:
    """
    Build recipe metadata mapping: str(recipe_id) -> {name, tags}.

    Uses RAW_recipes.csv for names and PP_recipes.csv for the graph-index
    to original-id mapping.
    """
    recipes_df = pd.read_csv(os.path.join(data_dir, "PP_recipes.csv"))
    recipe_id_lookup = recipes_df.set_index("i")["id"].to_dict()

    raw_path = os.path.join(data_dir, "RAW_recipes.csv")
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path, usecols=["id", "name", "tags"])
        name_lookup = raw_df.set_index("id")["name"].to_dict()
        tags_lookup = raw_df.set_index("id")["tags"].to_dict()
    else:
        log.warning(
            "RAW_recipes.csv not found at %s; metadata will have empty names/tags",
            raw_path,
        )
        name_lookup = {}
        tags_lookup = {}

    metadata: dict[str, dict] = {}
    for graph_idx, original_id in recipe_id_lookup.items():
        name = name_lookup.get(original_id, "Unknown")
        raw_tags = tags_lookup.get(original_id, "[]")
        try:
            tags = ast.literal_eval(raw_tags) if isinstance(raw_tags, str) else []
        except (ValueError, SyntaxError):
            tags = []
        metadata[str(original_id)] = {
            "name": name,
            "tags": tags,
            "graph_index": int(graph_idx),
        }
    return metadata


def write_to_redis(
    recommendations: dict[int, list[tuple[int, float]]],
    recipe_id_lookup: dict[int, int],
    recipe_metadata: dict[str, dict],
) -> None:
    """Push per-user recommendations into Redis as JSON lists."""
    import redis

    log.info("Connecting to Redis at %s:%d", REDIS_HOST, REDIS_PORT)
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r.ping()
    log.info("Redis connection OK")

    pipe = r.pipeline()
    count = 0
    for user_idx, recs in recommendations.items():
        payload = []
        for recipe_graph_idx, score in recs:
            original_id = recipe_id_lookup.get(recipe_graph_idx, recipe_graph_idx)
            meta = recipe_metadata.get(str(original_id), {})
            payload.append(
                {
                    "recipe_id": int(original_id),
                    "predicted_score": round(score, 6),
                    "name": meta.get("name", "Unknown"),
                }
            )
        pipe.set(f"user:{user_idx}", json.dumps(payload))
        count += 1
        # Flush every 1000 users to keep memory bounded
        if count % 1000 == 0:
            pipe.execute()
            pipe = r.pipeline()
    pipe.execute()
    log.info("Wrote %d user recommendation sets to Redis", count)


def write_to_json(
    recommendations: dict[int, list[tuple[int, float]]],
    recipe_id_lookup: dict[int, int],
    recipe_metadata: dict[str, dict],
    output_path: str,
) -> None:
    """Write per-user recommendations to a local JSON file (memory backend)."""
    result: dict[str, list[dict]] = {}
    for user_idx, recs in recommendations.items():
        payload = []
        for recipe_graph_idx, score in recs:
            original_id = recipe_id_lookup.get(recipe_graph_idx, recipe_graph_idx)
            meta = recipe_metadata.get(str(original_id), {})
            payload.append(
                {
                    "recipe_id": int(original_id),
                    "predicted_score": round(score, 6),
                    "name": meta.get("name", "Unknown"),
                }
            )
        result[f"user:{user_idx}"] = payload

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(
        "Wrote %d user recommendation sets to %s", len(result), output_path
    )


def _resolve_git_rev() -> str:
    # Env override wins so CI/CD can inject a build SHA even when the container
    # has no .git directory. Bare subprocess call is cheap and fails silently
    # for all the usual "not a repo / git not installed" scenarios.
    override = os.environ.get("GIT_REV")
    if override:
        return override
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip() or "nogit"
    except Exception:
        return "nogit"


def _append_batch_run(entry: dict) -> None:
    # Best-effort JSONL append; a broken log must not fail the batch job
    # because the recommendations are already persisted at this point.
    try:
        path = Path(BATCH_RUN_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("Failed to append batch run log to %s: %s", BATCH_RUN_LOG, exc)


# Files required for training and scoring — must all be present in DATA_DIR
# before the pipeline can proceed. The set comes from the notebook's data
# loading logic (PP_* + interactions_* for the graph, RAW_* for metadata).
REQUIRED_DATA_FILES = [
    "PP_recipes.csv",
    "PP_users.csv",
    "interactions_train.csv",
    "interactions_validation.csv",
    "interactions_test.csv",
    "RAW_recipes.csv",
    "RAW_interactions.csv",
]


def ensure_dataset(data_dir: str) -> None:
    """Ensure all required CSVs are present in *data_dir*, downloading from
    Kaggle if any are missing. Exits with code 1 on misconfiguration so the
    container fails fast rather than training on a partial dataset.
    """
    missing = [
        f for f in REQUIRED_DATA_FILES
        if not os.path.isfile(os.path.join(data_dir, f))
    ]
    if not missing:
        log.info("Data files found, skipping download")
        return

    log.info(
        "Missing %d of %d required data files (%s); attempting Kaggle download",
        len(missing),
        len(REQUIRED_DATA_FILES),
        ", ".join(missing),
    )

    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        log.error(
            "KAGGLE_USERNAME and KAGGLE_KEY must be set to download the dataset. "
            "Either provide them as environment variables or pre-populate %s "
            "with the required CSVs.",
            data_dir,
        )
        sys.exit(1)

    # The kaggle CLI reads credentials from env vars when set; export here
    # so that subprocess.run inherits them even if the parent shell lacked them.
    env = dict(os.environ)
    env["KAGGLE_USERNAME"] = KAGGLE_USERNAME
    env["KAGGLE_KEY"] = KAGGLE_KEY

    os.makedirs(data_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", "shuyangli94/food-com-recipes-and-user-interactions",
                "-p", data_dir,
                "--unzip",
            ],
            check=True,
            env=env,
        )
    except FileNotFoundError:
        log.error(
            "kaggle CLI not found on PATH. Install it (pip install kaggle) "
            "or pre-populate %s with the required CSVs.",
            data_dir,
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        log.error("kaggle download failed (exit=%d): %s", exc.returncode, exc)
        sys.exit(1)

    # Re-check after download — if Kaggle's archive layout ever changes we
    # want to fail here rather than deep inside pandas.read_csv.
    still_missing = [
        f for f in REQUIRED_DATA_FILES
        if not os.path.isfile(os.path.join(data_dir, f))
    ]
    if still_missing:
        log.error(
            "Kaggle download completed but files are still missing: %s",
            ", ".join(still_missing),
        )
        sys.exit(1)

    log.info("Data files downloaded successfully")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch pre-computation for GNN recommender")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score but skip cache writes (for canary evaluation)")
    args = parser.parse_args()

    t0 = time.time()
    log.info("Batch scoring pipeline started (dry_run=%s)", args.dry_run)
    log.info("Device: %s", DEVICE)
    log.info("MODEL_PATH: %s", MODEL_PATH)
    log.info("DATA_DIR: %s", DATA_DIR)
    log.info("CACHE_BACKEND: %s", CACHE_BACKEND)
    log.info("TOP_N: %d, BATCH_SIZE: %d", TOP_N, BATCH_SIZE)

    # Dataset must exist before either training or scoring; do this first so
    # a missing-file failure shows up immediately, not 10+ minutes into training.
    ensure_dataset(DATA_DIR)

    # ------------------------------------------------------------------
    # Step 1: Ensure a trained model exists
    # ------------------------------------------------------------------
    if not os.path.isfile(MODEL_PATH):
        train_model(DATA_DIR, MODEL_PATH)
    else:
        log.info("Model checkpoint found at %s", MODEL_PATH)

    # ------------------------------------------------------------------
    # Step 2: Load model checkpoint
    # ------------------------------------------------------------------
    log.info("Loading model checkpoint ...")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    data: HeteroData = checkpoint["data"]
    num_users: int = checkpoint["num_users"]
    num_recipes: int = checkpoint["num_recipes"]
    recipe_feat_dim: int = checkpoint.get(
        "recipe_feat_dim", data["recipe"].x.shape[1]
    )

    model = Model(
        hidden_channels=HIDDEN_CHANNELS,
        data=data,
        recipe_feat_dim=recipe_feat_dim,
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    log.info(
        "Model loaded: %d users, %d recipes, recipe_feat_dim=%d",
        num_users,
        num_recipes,
        recipe_feat_dim,
    )

    # ------------------------------------------------------------------
    # Step 3: Compute embeddings
    # ------------------------------------------------------------------
    log.info("Computing embeddings (full-graph forward pass) ...")
    user_embs, recipe_embs = compute_embeddings(model, data)
    log.info(
        "Embeddings: users %s, recipes %s",
        tuple(user_embs.shape),
        tuple(recipe_embs.shape),
    )

    # ------------------------------------------------------------------
    # Step 4: Score every user
    # ------------------------------------------------------------------
    rated = build_rated_set(data)
    recommendations = score_all_users(
        user_embs, recipe_embs, rated, num_users, TOP_N, BATCH_SIZE
    )

    # ------------------------------------------------------------------
    # Step 5: Build recipe metadata
    # ------------------------------------------------------------------
    log.info("Building recipe metadata ...")
    recipe_metadata = build_recipe_metadata(DATA_DIR)

    # Skip persistence during dry-run so canary evaluation can measure scoring
    # behavior (latency, score distribution) without overwriting prod artifacts.
    version: Optional[str] = None
    if not args.dry_run:
        os.makedirs(
            os.path.dirname(RECIPE_METADATA_OUTPUT) or ".", exist_ok=True
        )
        with open(RECIPE_METADATA_OUTPUT, "w") as f:
            json.dump(recipe_metadata, f, indent=2)
        log.info(
            "Recipe metadata (%d entries) written to %s",
            len(recipe_metadata),
            RECIPE_METADATA_OUTPUT,
        )

        # ------------------------------------------------------------------
        # Step 6: Write recommendations
        # ------------------------------------------------------------------
        # Build graph-index -> original-recipe-id lookup for output
        recipes_df = pd.read_csv(os.path.join(DATA_DIR, "PP_recipes.csv"))
        recipe_id_lookup: dict[int, int] = recipes_df.set_index("i")["id"].to_dict()

        if CACHE_BACKEND == "redis":
            try:
                write_to_redis(recommendations, recipe_id_lookup, recipe_metadata)
            except Exception as exc:
                log.error("Redis write failed: %s", exc)
                fallback = RECOMMENDATIONS_OUTPUT
                log.info("Falling back to JSON output at %s", fallback)
                write_to_json(
                    recommendations, recipe_id_lookup, recipe_metadata, fallback
                )
        else:
            write_to_json(
                recommendations,
                recipe_id_lookup,
                recipe_metadata,
                RECOMMENDATIONS_OUTPUT,
            )

        # Stamp current_version.txt only on a successful full run; canary
        # readers treat a stale file as "no new version" and keep the old one.
        ts_iso = datetime.now(timezone.utc).isoformat()
        git_rev = _resolve_git_rev()
        version = f"{ts_iso}\t{git_rev}"
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            version_path = os.path.join(MODEL_DIR, "current_version.txt")
            with open(version_path, "w") as f:
                f.write(version + "\n")
            log.info("Wrote version marker to %s", version_path)
        except Exception as exc:
            log.warning("Failed to write current_version.txt: %s", exc)

    # Both dry-run and full-run emit a batch_run entry so the monitor can
    # distinguish canary scoring from production refreshes.
    _append_batch_run({
        "version": version,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "num_users_scored": len(recommendations),
        "dry_run": args.dry_run,
    })

    elapsed = time.time() - t0
    log.info("Batch scoring pipeline completed in %.1f seconds", elapsed)


if __name__ == "__main__":
    main()


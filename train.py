#!/usr/bin/env python
# coding: utf-8

import os
import ast
import time
import tqdm
import numpy as np
import pandas as pd
from collections import Counter

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
    "data_path": "./data",
    "model_output_path": "./best_model.pt",
    "mlflow_experiment": "mealie-gnn-recommendations",
    "run_name": "v1_best",
}
# ──────────────────────────────────────────────────────────────────────────────

def load_data(config):
    data_path = config["data_path"]
    print("Loading data...")

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

    num_users = users_df['u'].max() + 1
    num_recipes = recipes_df['i'].max() + 1

    ratings_user_id = torch.from_numpy(interactions_df['u'].values)
    ratings_recipe_id = torch.from_numpy(interactions_df['i'].values)
    edge_index_user_to_recipe = torch.stack([ratings_user_id, ratings_recipe_id], dim=0)

    data = HeteroData()
    data["user"].node_id = torch.arange(num_users)
    data["recipe"].node_id = torch.arange(num_recipes)
    data["recipe"].x = recipe_feat
    data["user"].x = user_feat
    data["user", "rates", "recipe"].edge_index = edge_index_user_to_recipe
    data = T.ToUndirected()(data)

    print(f"Graph built: {num_users} users, {num_recipes} recipes")
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
    mlflow.set_experiment(config["mlflow_experiment"])

    recipes_df, users_df, interactions_df, interactions_train_df, interactions_val_df, interactions_test_df = load_data(config)
    recipe_feat, user_feat, recipe_feat_dim = build_features(recipes_df, users_df, config)
    data, num_users, num_recipes = build_graph(recipes_df, users_df, interactions_df, recipe_feat, user_feat)
    train_loader, val_loader, test_loader = build_loaders(data, config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with mlflow.start_run(run_name=config["run_name"]):
        # log config
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

        model = Model(
            hidden_channels=config["hidden_channels"],
            recipe_feat_dim=recipe_feat_dim,
            data=data
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        start = time.time()
        best_val_auc = 0

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

        train_time = time.time() - start
        test_auc, test_ap = evaluate(model, test_loader, device)

        mlflow.log_metric("best_val_auc", best_val_auc)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_ap", test_ap)
        mlflow.log_metric("train_time_sec", round(train_time, 2))
        mlflow.log_artifact(config["model_output_path"])

        print(f"\nTraining done!")
        print(f"Best Val AUC: {best_val_auc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test AP: {test_ap:.4f}")
        print(f"Train time: {train_time:.2f}s")
        print(f"Model saved to: {config['model_output_path']}")


if __name__ == "__main__":
    train(CONFIG)
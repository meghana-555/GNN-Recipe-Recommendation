"""
GNN model definitions for the recipe recommendation system.

Contains the heterogeneous GraphSAGE model, dot-product classifier,
and the combined Model class -- extracted verbatim from the training
notebook with RECIPE_FEAT_DIM made a constructor parameter.
"""
# Assisted by Claude

import typing

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero

# Compatibility shim: PyTorch >= 2.8 removed symbols from
# torch.fx._symbolic_trace that PyG 2.6.x still references inside
# to_hetero().  Patch them back so the conversion does not crash.
import torch.fx._symbolic_trace as _st

if not hasattr(_st, "List"):
    _st.List = typing.List
if not hasattr(_st, "Dict"):
    _st.Dict = typing.Dict
if not hasattr(_st, "Optional"):
    _st.Optional = typing.Optional


class GNN(torch.nn.Module):
    """Two-layer homogeneous GraphSAGE backbone."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    """Dot-product classifier that scores (user, recipe) edges."""

    def forward(
        self, x_user: Tensor, x_recipe: Tensor, edge_label_index: Tensor
    ) -> Tensor:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_recipe = x_recipe[edge_label_index[1]]
        return (edge_feat_user * edge_feat_recipe).sum(dim=-1)


class Model(torch.nn.Module):
    """
    Full heterogeneous GNN model.

    Parameters
    ----------
    hidden_channels : int
        Dimensionality of the shared latent space.
    data : HeteroData
        The heterogeneous graph -- used to read ``num_nodes`` for
        embedding tables and ``metadata()`` for the hetero conversion.
    recipe_feat_dim : int
        Number of input features per recipe node
        (techniques + calorie_level + ingredient multi-hot).
    user_feat_dim : int
        Number of input features per user node (default 58 technique
        counts, matching the notebook).
    """

    def __init__(
        self,
        hidden_channels: int,
        data: HeteroData,
        recipe_feat_dim: int,
        user_feat_dim: int = 58,
    ):
        super().__init__()
        self.user_lin = torch.nn.Linear(user_feat_dim, hidden_channels)
        self.recipe_lin = torch.nn.Linear(recipe_feat_dim, hidden_channels)
        self.user_emb = torch.nn.Embedding(
            data["user"].num_nodes, hidden_channels
        )
        self.recipe_emb = torch.nn.Embedding(
            data["recipe"].num_nodes, hidden_channels
        )
        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "user": self.user_lin(data["user"].x)
            + self.user_emb(data["user"].node_id),
            "recipe": self.recipe_lin(data["recipe"].x)
            + self.recipe_emb(data["recipe"].node_id),
        }
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["user"],
            x_dict["recipe"],
            data["user", "rates", "recipe"].edge_label_index,
        )
        return pred

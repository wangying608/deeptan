r"""
Modules for DeepTAN.
"""

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing


class WGATLayer(MessagePassing):
    r"""
    (NMIC) Weighted Graph Attention Layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        negative_slope: float,
        num_heads: int,
        dropout: float = 0.1,
    ):
        r"""
        Initialize the Weighted Graph Attention Layer.

        Args:
            input_dim: The dimension of the input embeddings.
            output_dim: The dimension of the output embeddings.
            negative_slope: The negative slope of the LeakyReLU activation.
            num_heads: The number of attention heads.
            dropout: The dropout probability for the attention weights.
        """
        super().__init__(aggr="add")
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.trans = nn.Sequential(
            nn.Linear(input_dim, input_dim * num_heads),
            nn.LeakyReLU(negative_slope),
            nn.Linear(input_dim * num_heads, output_dim * num_heads),
        )
        # Adjusted for per-head attention computation
        self.W = nn.Parameter(torch.empty(2 * input_dim, num_heads * output_dim))
        self.attn = nn.Parameter(torch.empty(num_heads, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data)
        nn.init.xavier_uniform_(self.attn.data)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Compute attention scores per head
        # h = torch.cat([x_i, x_j], -1) @ self.W  # [E, num_heads * output_dim]
        # h = h.view(-1, self.num_heads, self.output_dim)  # [E, num_heads, output_dim]
        h = (torch.cat([x_i, x_j], -1) @ self.W).view(
            -1, self.num_heads, self.output_dim
        )

        # Calculate attention coefficients [E, num_heads]
        e = (h * self.attn.unsqueeze(0)).sum(dim=-1)  # Dot product per head

        # Integrate edge attributes
        if edge_attr is not None:
            # Expand edge_attr to match num_heads [E, num_heads]
            e = e * edge_attr.view(-1, 1).expand(-1, self.num_heads)

        # Normalize attention scores
        a = F.leaky_relu(e, 0.2)
        a = F.softmax(a, dim=0)  # Normalize over neighbors
        a = F.dropout(a, self.dropout, training=self.training)

        # Transform features and prepare multi-head output
        x_trans = self.trans(x_j).view(
            -1, self.num_heads, self.output_dim
        )  # [E, num_heads, output_dim]

        # Weight features by attention scores
        # Average features across heads
        h = (x_trans * a.unsqueeze(-1)).mean(dim=1)  # [E, output_dim]

        return h


class NodeEmbedding(nn.Module):
    r"""
    For node embedding.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        fusion_dims: List[int],
        dict_node_names: Dict[str, int],
        n_heads: int,
        negative_slope: float = 0.2,
        dropout: float = 0.2,
    ):
        r"""
        Embedding nodes in a graph like embedding words in a sentence.

        Args:
            input_dim: Dimension of input features.
            embedding_dim: Dimension of the embedding.
            fusion_dims: Dimensions for the fusion (fusing observation value and inherent feature) layers.
            dict_node_names: Dictionary mapping node names to indices.
            n_heads: Number of attention heads.
            negative_slope: Negative slope for the LeakyReLU activation.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.fusion_dims = fusion_dims
        self.dict_node_names = dict_node_names
        self.n_heads = n_heads
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.embed = nn.Embedding(len(dict_node_names), embedding_dim)

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, embedding_dim), nn.LayerNorm(embedding_dim), nn.GELU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

        # WGAT layers with skip connections
        self.layers = nn.ModuleList(
            [
                WGATLayer(
                    dim_in if i else embedding_dim,
                    dim_out,
                    negative_slope,
                    n_heads,
                    dropout,
                )
                for i, (dim_in, dim_out) in enumerate(
                    zip([embedding_dim] + fusion_dims[:-1], fusion_dims)
                )
            ]
        )

        # Skip connections
        self.skips = (
            nn.ModuleList([nn.Linear(dim, fusion_dims[-1]) for dim in fusion_dims])
            if len(fusion_dims) > 1
            else None
        )

    def forward(self, node_names, x, edge_attr, edge_index):
        if isinstance(node_names[0], list):
            node_names = [n for sublist in node_names for n in sublist]

        # Verify node indices in edge_index
        # num_nodes = x.size(0)
        # assert torch.all(edge_index >= 0) and torch.all(edge_index < num_nodes), (
        #     "Invalid edge indices detected"
        # )

        # Initial embeddings
        ids = torch.tensor(
            [self.dict_node_names[n] for n in node_names],
            dtype=torch.long,
            device=x.device,
        )

        E_i = self.embed(ids)
        x_mlp1 = self.mlp1(x)
        combined = torch.cat([x_mlp1, E_i], dim=-1)
        x_mlp2 = self.mlp2(combined)
        emb = x_mlp2 + E_i

        # Multi-scale processing
        skips = []
        if self.skips:
            for i, layer in enumerate(self.layers):
                # emb = layer(emb, edge_index, edge_attr)
                emb = checkpoint(layer, emb, edge_index, edge_attr, use_reentrant=False)
                if i < len(self.skips):
                    skips.append(self.skips[i](emb))

            # Skip fusion
            emb = emb + torch.stack(skips).mean(dim=0)

        # emb = F.layer_norm(emb, emb.shape)

        return emb


class GE_Decoder(nn.Module):
    r"""
    Graph Embedding Decoder for reconstructing node features from latent representations (biological state-specific embeddings).
    """

    def __init__(self, z_dim: int, h_dim: int, output_dim: int, hidden_dim: int = 128):
        r"""
        Initialize graph embedding decoder.

        Args:
            z_dim: Dimension of the latent representation.
            h_dim: Dimension of node features.
        """
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.ffn_i = nn.Sequential(
            nn.Linear(z_dim + h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, h_dim),
        )
        self.ffn_q = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, Embedding: nn.Embedding):
        z_expanded = z.unsqueeze(1).expand(-1, Embedding.num_embeddings, -1)
        E = Embedding.weight.unsqueeze(0).expand(z.size(0), -1, -1)
        combined = torch.cat([z_expanded, E], dim=-1)
        h_s = self.ffn_i(combined) + E
        h_ = self.ffn_q(h_s)
        return h_s, h_


class GLabelPredictor(nn.Module):
    r"""
    A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        r"""
        Args:
            input_dim: The input dimension.
            output_dim: The output dimension.
            hidden_dims: The hidden dimensions of the feedforward network.
        """
        super().__init__()
        layers = []
        for dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, dim),
                nn.GELU(),
            ]
            input_dim = dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class SelfAttPool(nn.Module):
    r"""
    Self-attention pooling layer.
    This layer performs self-attention on the input tensor and then pools the results by taking the mean.
    """

    def __init__(self, dim: int):
        r"""Initialize the Self-attention pooling layer.
        Args:
            dim: The dimension of the input tensor.
        """
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).mean(dim=0)
        return self.proj(x)

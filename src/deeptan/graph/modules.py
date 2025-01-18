r"""
Modules for biological state-specific graph embedding.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class WGATLayer(MessagePassing):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        negative_slope: float,
        num_heads: int,
    ):
        r"""Edge weight guided GAT layer.

        Args:
            input_dim: Input node embedding dimension.

            output_dim: Output node embedding dimension.

            negative_slope: LeakyReLU negative slope.

            num_heads: Number of attention heads.
        """
        super().__init__(aggr="add")

        self.output_dim = output_dim
        self.num_heads = num_heads

        self.trans = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.W = nn.Parameter(torch.zeros(size=(input_dim * 2, output_dim * num_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.attn = nn.Parameter(torch.zeros(size=(output_dim * num_heads, num_heads)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)

        self.activation = nn.LeakyReLU(negative_slope)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        if edge_attr is not None:
            edge_attr = edge_attr.view(-1, 1)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Reshape the output to (num_nodes, output_dim * num_heads)
        return h.view(-1, self.output_dim * self.num_heads)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        # edge_attr has shape: torch.Size([E, 1])
        h_cat = torch.cat([x_i, x_j], dim=-1)
        # Shape of h_cat: (E, 2 * input_dim)
        Wh = h_cat @ self.W
        # Shape of Wh: (E, output_dim * num_heads)

        # Reshape Wh to (E, output_dim, num_heads)
        Wh = Wh.view(-1, self.output_dim, self.num_heads)

        e_ij: torch.Tensor = self.activation(Wh.matmul(self.attn))
        # Shape of e_ij: (E, num_heads)

        if edge_attr is None:
            e_ij = e_ij.softmax(dim=0)
        else:
            # Mask attention scores with adjacency matrix
            e_ij = e_ij.mul(edge_attr).softmax(dim=0)
        # Shape of e_ij: (E, num_heads)

        # Compute transformed node embeddings for each head
        # Shape of x_j_transformed: (E, output_dim * num_heads)
        x_j_transformed = self.trans(x_j).view(-1, self.output_dim, self.num_heads)

        # Multiply transformed embeddings by attention scores
        output: torch.Tensor = x_j_transformed * e_ij.unsqueeze(1)
        # Shape of output: (E, output_dim, num_heads)

        # Reshape output to (E, output_dim * num_heads)
        return output.view(-1, self.output_dim * self.num_heads)


class NodeEmbedding(nn.Module):
    r"""
    Biological state-specific feature embedding.
    """

    def __init__(
        self,
        input_dim: int,
        num_embeddings: int,
        embedding_dim: int,
        fusion_dims: List[int],
        dict_node_names: Dict[str, int],
        n_heads: List[int],
        negative_slope: float,
    ):
        r"""WGAT layers with skip connections.

        Args:
            input_dim: Input node embedding dimension.

            embedding_dim: Target node embedding dimension.

            fusion_dims: Node embedding dimensions after embedding_dims achiving.

            n_heads: Number of attention heads for each layer.

            negative_slope: LeakyReLU negative slope.

        """
        super().__init__()
        self.node_embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=None,
            sparse=True,
        )
        self.dict_node_names = dict_node_names

        # Feature encoding layers (Part 1)
        self.raiseQuaDim = nn.Sequential(
            nn.Linear(
                input_dim, embedding_dim // 2
            ),  # Input: quantitative features (e.g., gene expression)
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, embedding_dim),
            nn.Tanh(),
        )
        self.quaEncoder = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )

        # Feature encoding layers (Part 2) for fusing node embeddings and their quantitative features
        self.n_layers_fusion = len(fusion_dims)
        self.layers = nn.ModuleList()
        for i in range(self.n_layers_fusion):
            if i == 0:
                self.layers.append(
                    WGATLayer(embedding_dim, fusion_dims[i], negative_slope, n_heads[0])
                )
            else:
                self.layers.append(
                    WGATLayer(
                        fusion_dims[i - 1], fusion_dims[i], negative_slope, n_heads[i]
                    )
                )

        if self.n_layers_fusion > 2:
            # Enable skip connections
            self.skip_connections = True
            self.Ws = nn.ModuleList()
            for i in range(self.n_layers_fusion - 2):
                self.Ws.append(nn.Linear(fusion_dims[i], fusion_dims[-1]))
                nn.init.xavier_uniform_(self.Ws[-1].weight.data, gain=1.414)
        else:
            self.skip_connections = False
            self.Ws = None

    def forward(
        self,
        node_names: List[str],
        x: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        indices_embedding = torch.tensor(
            [self.dict_node_names[node] for node in node_names],
            dtype=torch.long,
            device=x.device,
        )
        node_embedding = self.node_embedding(indices_embedding)

        x = self.raiseQuaDim(x)
        x = self.quaEncoder(torch.cat([node_embedding, x], dim=-1)) + node_embedding

        h_list = []
        for i in range(self.n_layers_fusion):
            x = self.layers[i](x, edge_index, edge_attr)
            h_list.append(x)

        if self.skip_connections and self.Ws is not None:
            for i in range(self.n_layers_fusion - 2):
                h_list[i] = self.Ws[i](h_list[i])

            h_list[-1] = h_list[-1] + torch.mean(torch.stack(h_list[:-2]), dim=0)

        return h_list[-1]


class SelfAttPool(nn.Module):
    r"""
    Self-attention pooling layer for graph pooling.
    """

    def __init__(self, input_dim: int):
        r"""Apply self-attention to multi-scale subgraphs' embeddings.

        Args:
            input_dim (int): The dimension of the input embeddings.
        """
        super().__init__()
        self.input_dim = input_dim

        # Self-attention layers
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)

        # Output projection
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, g_emb: torch.Tensor):
        r"""
        Args:
            g_emb (torch.Tensor): Graph embeddings of shape ``(num_graphs, input_dim)``.

        Returns:
            Pooled representation of shape ``(output_dim_g_emb,)``.
        """
        # Compute query, key, and value
        Q = self.query(g_emb)
        K = self.key(g_emb)
        V = self.value(g_emb)

        # Scaled dot-product attention
        scale = torch.sqrt(torch.tensor(self.input_dim, dtype=torch.float32))
        attention_scores = torch.matmul(Q, K.transpose(0, 1)) / scale
        attention_weights = nn.functional.softmax(
            attention_scores, dim=1
        )  # (num_graphs, num_graphs)

        # Apply attention to the value
        weighted = torch.matmul(attention_weights, V)  # (num_graphs, input_dim)

        # Project the output
        weighted = self.proj(weighted)  # (num_graphs, input_dim)

        # Concatenate the pooled representation with the mean and std of the graphs
        weighted_mean = weighted.sum(dim=0)
        weighted_std = weighted.mul(weighted.shape[0]).std(dim=0)
        g_emb_mean = g_emb.mean(dim=0)
        g_emb_std = g_emb.std(dim=0)
        out_emb = torch.cat([weighted_mean, weighted_std, g_emb_mean, g_emb_std])

        # Shape of the output: (graph_embedding_dim * 4)
        return out_emb


class VGAE_Decoder(nn.Module):
    def __init__(self, graph_embedding_dim: int, node_embedding_dim: int):
        super().__init__()
        self.graph_embedding_dim = graph_embedding_dim
        self.node_embedding_dim = node_embedding_dim

        self.guess_num_nodes = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.Sigmoid(),
            nn.Linear(graph_embedding_dim // 2, 1),
            nn.Softplus(),
        )
        self.fc_mu = nn.Linear(graph_embedding_dim, graph_embedding_dim)
        self.fc_var = nn.Linear(graph_embedding_dim, graph_embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(graph_embedding_dim // 2, node_embedding_dim * 4),
            nn.Tanh(),
            nn.Linear(node_embedding_dim * 4, node_embedding_dim),
            nn.LeakyReLU(),
        )

    def forward(self, graph_embedding: torch.Tensor):
        num_nodes = torch.exp(self.guess_num_nodes(graph_embedding)).round().long()
        graph_embedding_expanded = graph_embedding.unsqueeze(0).expand(
            int(num_nodes.item()), -1
        )
        # Shape of graph_embedding_expanded: (num_nodes, graph_embedding_dim)

        # mu, var
        mu = self.fc_mu(graph_embedding_expanded)
        log_var = self.fc_var(graph_embedding_expanded)
        z = self.reparameterize(mu, log_var)

        node_embeddings = self.decoder(z)

        return node_embeddings

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class GLabelPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)


# class AttPool(nn.Module):
#     r"""
#     Attention pooling layer for graph pooling.
#     """
#     def __init__(self, input_dim: int):
#         r"""Apply attention to multi-scale subgraphs' embeddings.

#         """
#         super().__init__()
#         # self.attention = nn.Linear(input_dim, 1)
#         self.attention = nn.Sequential(
#             nn.Linear(input_dim, input_dim // 4),
#             nn.Tanh(),
#             nn.Linear(input_dim // 4, 1),
#         )

#     def forward(self, g_emb: torch.Tensor):
#         r"""
#         Args:
#             g_emb: Graph embeddings of shape ``(num_graphs, input_dim)``.

#         Returns:
#             Pooled representation of shape ``(output_dim_g_emb,)``.
#         """
#         weight_per_g = self.attention(g_emb).softmax(dim=0)
#         weighted = g_emb.mul(weight_per_g)
#         # pooled = weighted.sum(dim=0)

#         # Concatenate the pooled representation with the mean and std of the graphs
#         weighted_mean = weighted.sum(dim=0)
#         weighted_std = weighted.mul(weighted.shape[0]).std(dim=0)
#         g_emb_mean = g_emb.mean(dim=0)
#         g_emb_std = g_emb.std(dim=0)
#         out_emb = torch.cat([weighted_mean, weighted_std, g_emb_mean, g_emb_std])

#         # Shape of the output: (graph_embedding_dim * 4)
#         return out_emb

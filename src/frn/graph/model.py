r"""
xxx definition.
"""
from typing import List
import numpy as np
import torch
import torch.nn as nn
# from torch.optim.adam import Adam
import lightning as ltn
import networkx as nx
# from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
# from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
import frn.constants as const


class XGAT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_heads: int,
            dropout: float,
            negative_slope: float,
        ):
        r"""XGAT layer.

        Args:
            input_dim: Input node embedding dimension.
            output_dim: Output node embedding dimension.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
            negative_slope: LeakyReLU negative slope.
        
        """
        super().__init__()

        self.n_heads = n_heads
        self.output_dim = output_dim

        self.W = nn.Parameter(torch.zeros(size=(input_dim * 2, output_dim * n_heads)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.attn = nn.Parameter(torch.zeros(size=(output_dim * n_heads, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.activation = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        n_nodes = h.shape[0]

        # Shape of W: [input_dim * 2, output_dim * n_heads]
        # Shape of h: [n_nodes, input_dim]
        # Shape of Wh: [n_nodes, output_dim * n_heads]

        # Repeat h to create all pairs (h_i, h_j)
        h_repeated = h.repeat(n_nodes, 1)
        h_i = h_repeated.view(n_nodes, n_nodes, -1)
        h_j = h_i.transpose(0, 1)

        # Concatenate h_i and h_j
        h_cat = torch.cat([h_i, h_j], dim=-1)

        # Apply linear transformation W
        Wh = torch.matmul(h_cat, self.W)
        Wh = Wh.view(n_nodes, n_nodes, self.output_dim, self.n_heads)

        # Compute attention scores e_ij
        e_ = self.activation(torch.matmul(Wh, self.attn).squeeze(-1))

        # Mask attention scores with adjacency matrix
        mask_ = adj.bool().unsqueeze(-1)
        attention = torch.where(mask_, e_, -9e15 * torch.ones_like(e_))
        
        # Apply softmax to attention scores
        # a_{ij} = softmax(M_{ij} * e_{ij})
        attention: torch.Tensor = self.softmax(attention * adj)

        # Apply dropout to attention scores
        attention = self.dropout(attention)
        
        # Compute transformed node embeddings
        h_prime = torch.matmul(attention.unsqueeze(-2), Wh.unsqueeze(-1)).squeeze(-2)
        
        # Average over heads
        h_prime = h_prime.mean(dim=-1)

        return h_prime


class XGATLayers(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dims: List[int],
            n_heads: List[int],
            dropout: float,
            negative_slope: float,
        ):
        r"""XGAT layers with skip connections.

        Args:
            input_dim: Input node embedding dimension.

            output_dims: Output node embedding dimensions for each layer.
                The length denotes the number of layers.

            n_heads: Number of attention heads for each layer.
                The length must be the same as output_dims.
            
            dropout: Dropout rate.
            
            negative_slope: LeakyReLU negative slope.

        """
        super().__init__()

        assert len(output_dims) == len(n_heads), "output_dims and n_heads must have the same length."
        self.n_layers = len(output_dims)

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(XGAT(input_dim, output_dims[i], n_heads[i], dropout, negative_slope))
            else:
                self.layers.append(XGAT(output_dims[i-1], output_dims[i], n_heads[i], dropout, negative_slope))
        
        if self.n_layers > 2:
            # Enable skip connections
            self.skip_connections = True
            self.Ws = nn.ModuleList()
            for i in range(self.n_layers - 2):
                self.Ws.append(nn.Linear(output_dims[i], output_dims[-1]))
                nn.init.xavier_uniform_(self.Ws[-1].weight.data, gain=1.414)
        else:
            self.skip_connections = False
            self.Ws = None
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h_list = []
        for i_layer in self.layers:
            h = i_layer(h, adj)
            h_list.append(h)
        
        if self.skip_connections and self.Ws is not None:
            for i in range(self.n_layers - 2):
                h_list[i] = torch.matmul(h_list[i], self.Ws[i].weight)
            
            h_list[-1] = h_list[-1] + torch.mean(torch.stack(h_list[:-2]), dim=0)
        
        return h_list[-1]


def weighted_adj_to_nx(adj: torch.Tensor):
    r"""
    Converts weighted adjacency matrix `adj` to a NetworkX graph.
    
    Args:
        adj: Weighted adjacency matrix of shape (num_nodes, num_nodes).
    
    Returns:
        nx.Graph: NetworkX graph with weighted edges.

    """
    if adj.device != torch.device('cpu'):
        adj = adj.cpu()
    G = nx.Graph()
    G.add_nodes_from(range(adj.size(0)))
    # Only need to check upper triangle for undirected graph
    for i in range(adj.size(0)):
        for j in range(i + 1, adj.size(1)):
            weight = adj[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)
    return G

def compute_C_BCE(G: nx.Graph) -> np.ndarray:
    r"""Computes the betweenness centrality, closeness centrality and eigenvector centrality of each node in the graph ``G``.
    """
    c_B = nx.betweenness_centrality(G, weight='weight')
    c_C = nx.closeness_centrality(G, distance='weight')
    c_E = nx.eigenvector_centrality(G, weight='weight')
    n_nodes = G.number_of_nodes()
    out_array = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        out_array[i, 0] = c_B[i]
        out_array[i, 1] = c_C[i]
        out_array[i, 2] = c_E[i]
    return out_array


class DynamicCentrality(nn.Module):
    def __init__(
            self,
            node_emb_dim: int,
        ):
        r"""Dynamic centrality: weighted sum of
        the betweenness centrality, closeness centrality and eigenvector centrality
        of each node based on the node embedding.

        Args:
            node_emb_dim: Dimension of the node embedding.
        
        """
        super().__init__()

        self.Q_BCE = nn.Sequential(nn.Linear(node_emb_dim, 16), nn.Linear(16, 3), nn.Softmax(dim=1))
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        w_BCE = self.Q_BCE(h)
        c_BCE = self.get_c_BCE(adj)
        cs = torch.mul(w_BCE, c_BCE).sum(dim=1).unsqueeze(1)
        return cs
    
    def get_c_BCE(self, adj: torch.Tensor) -> torch.Tensor:
        G = weighted_adj_to_nx(adj)
        c_BCE = compute_C_BCE(G)
        return torch.tensor(c_BCE, dtype=torch.float32)


class MSGP(ltn.LightningModule):
    def __init__(
            self,
            input_dim: int,
            output_dims: List[int],
            n_heads: List[int],
            dropout: float,
            negative_slope: float,
        ):
        r"""Multi-Scale Graph Pooling for graph-level representation learning.

        Args:
            input_dim: Input node embedding dimension.

            output_dims: Output node embedding dimensions for each layer.
                The length denotes the number of layers.

            n_heads: Number of attention heads for each layer.
                The length must be the same as output_dims.
            
            dropout: Dropout rate.
            
            negative_slope: LeakyReLU negative slope.
        
        """
        super().__init__()
        self.save_hyperparameters()

        self.xgat_layers = XGATLayers(input_dim, output_dims, n_heads, dropout, negative_slope)
        self.dynamic_centrality = DynamicCentrality(output_dims[-1])
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # Node embedding
        h = self.xgat_layers(h, adj)
        
        # Nodes' dynamic centrality
        adj = self.cosine_sim(h)
        s_ = self.dynamic_centrality(h, adj)
        rank_s = torch.argsort(s_, descending=True)
        
        # Histogram
        hist_counts = self.hist_fd(s_)

        # Multi-scale subgraphs
        # Find central nodes in the k top bins and get their subgraphs.
        # The local subgraphs are constructed by including their neighbors up to a specific depth.
        # The global subgraph is constructed by including all nodes.
        # The core subgraph is constructed by including all key nodes in the top bins.

    def get_subgraphs(self, adj: torch.Tensor, bins: List[int], depth: int) -> List[int]:
        r"""Get the nodes in the subgraph of a central node.

        Args:
            adj: Adjacency matrix of the graph.

            central_node_i: Index of the central node.
        """
    
    def get_central_nodes_index(self, rank_s: torch.Tensor, hist_counts: torch.Tensor, bins: List[int]):
        r"""Get the indices of the central nodes in the specific bins.
        """
        for i in range(len(bins)):
            if hist_counts[i] > 0:
                

    def cosine_sim(self, h: torch.Tensor) -> torch.Tensor:
        norm_h = nn.functional.normalize(h, p=2, dim=1)
        cosine_simi = torch.matmul(norm_h, norm_h.T)
        # upper_triangular = torch.triu(torch.mm(norm_h, norm_h.T), diagonal=1)
        return cosine_simi
    
    def hist_fd(self, tensor1d: torch.Tensor) -> torch.Tensor:
        r"""Freedman-Diaconis rule for histogram bin width.
        """
        iqr = tensor1d.quantile(0.75) - tensor1d.quantile(0.25)
        bin_width = 2 * iqr * tensor1d.numel() ** (-1 / 3)
        num_bins = int((tensor1d.max() - tensor1d.min()) / bin_width)

        hist_counts = torch.histc(tensor1d, bins=num_bins)
        # bin_edges = torch.linspace(tensor1d.min(), tensor1d.max(), steps=num_bins + 1)
        return hist_counts

r"""
xxx definition.
"""
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
# from torch.optim.adam import Adam
import lightning as ltn
import networkx as nx
# from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassAUROC, MulticlassPrecision, MulticlassRecall, MatthewsCorrCoef
# from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, R2Score, PearsonCorrCoef
from torch_geometric.nn import global_mean_pool, global_max_pool
import frn.constants as const

torch.set_float32_matmul_precision(const.default.matmul_precision)


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
        h_list = [i_layer(h, adj) for i_layer in self.layers]
        
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
        r"""Dynamic centrality for multi-omics nodes (specific attributes and categories are considered):
        The weighted sum of the betweenness centrality, closeness centrality and eigenvector centrality
        of each node based on the node embedding.

        Args:
            node_emb_dim: Dimension of the node embedding.
        """
        super().__init__()

        self.Q_BCE = nn.Sequential(nn.Linear(node_emb_dim, 32), nn.Sigmoid(), nn.Linear(32, 3), nn.Softmax(dim=1))
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        w_BCE = self.Q_BCE(h)
        c_BCE = self.get_c_BCE(adj)
        cs = torch.mul(w_BCE, c_BCE).sum(dim=1).unsqueeze(1)
        # Shape: (n_nodes, 1)

        # Square the values to avoid negative values and change the distribution to be more scale-free
        cs = torch.pow(cs, 2)
        
        return cs
    
    def get_c_BCE(self, adj: torch.Tensor) -> torch.Tensor:
        G = weighted_adj_to_nx(adj)
        c_BCE = compute_C_BCE(G)
        return torch.tensor(c_BCE, dtype=torch.float32)


class AttPool(nn.Module):
    def __init__(
            self,
            input_dim: int,
            # dropout: float,
        ):
        r"""Apply attention to multi-scale subgraphs.
        
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, multiscale_subg_emb: torch.Tensor):
        weight_per_subg = torch.softmax(torch.pow(torch.sigmoid(self.attention(multiscale_subg_emb)), 2), dim=0)
        pooled = torch.mul(multiscale_subg_emb, weight_per_subg).sum(dim=0)
        return pooled


class MSGP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dims_nd: List[int],
            output_dim_g_emb: int,
            n_heads: List[int],
            n_orders: int,
            threshold_subgraph_overlap: float,
            dropout: float,
            negative_slope: float,
        ):
        r"""Multi-Scale Graph Pooling for graph-level representation learning.

        Args:
            input_dim: Input node embedding dimension.
            output_dims_nd: Output node embedding dimensions for each layer.
                The length denotes the number of layers.
            output_dim_g_emb: Output graph embedding dimension.
            n_heads: Number of attention heads for each layer.
                The length must be the same as output_dims.
            n_orders: Number of orders of neighbors to find for central nodes.
            threshold_subgraph_overlap: Threshold for the overlap between subgraphs.
            dropout: Dropout rate.
            negative_slope: LeakyReLU negative slope.
        """
        super().__init__()

        self.n_orders = n_orders
        self.threshold_subgraph_overlap = threshold_subgraph_overlap

        self.xgat_layers = XGATLayers(input_dim, output_dims_nd, n_heads, dropout, negative_slope)
        self.dynamic_centrality = DynamicCentrality(output_dims_nd[-1])

        self.global_att_pool = AttPool(output_dims_nd[-1])
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # Node embedding
        h = self.xgat_layers(h, adj)
        
        # Nodes' dynamic centrality
        adj = self.cosine_sim(h)
        ds = self.dynamic_centrality(h, adj)
        s_ranks = torch.argsort(ds, descending=False)
        
        # Histogram
        hist_counts = self.hist_fd(ds)

        # Generate multi-scale subgraphs
        subgraphs_nodes = self.collect_subgraphs(adj, s_ranks, hist_counts)
        # n_scales = len(subgraphs_nodes)
        # dk_scales = subgraphs_nodes.keys()

        # Extract subgraphs for each scale
        subgraphs_nodes = [subgraphs_nodes[i][j] for i in subgraphs_nodes.keys() for j in range(len(subgraphs_nodes[i]))]
        print(f"\nNumber of subgraphs: {len(subgraphs_nodes)} .\n")

        # Pool subgraphs to graph representations
        multiscale_subg_emb = torch.cat([self.pool_subgraph(h, subgraphs_nodes[i], ds) for i in range(len(subgraphs_nodes))], dim=0)

        # Subgraph pooling and graph-level representation by attention pooling.
        x = self.global_att_pool(multiscale_subg_emb)
        return x
    
    def pool_subgraph(self, h: torch.Tensor, subgraph_nodes: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
        r"""Pool a subgraph to its graph representation.

        Args:
            h: Node embedding.
            subgraph_nodes: Nodes in the subgraph.
            ds: Dynamic centrality scores of nodes.
        
        Returns:
            Graph representation.
        """
        _softmax = nn.Softmax()
        _s = _softmax(ds[subgraph_nodes])
        assert _s.shape[0] == len(subgraph_nodes)
        # Assigns attention weights to nodes and computes a weighted sum of node features
        pooled = torch.mean(torch.matmul(_s, h[subgraph_nodes]), dim=0)
        return pooled

    def collect_subgraphs(self, adj: torch.Tensor, s_ranks: torch.Tensor, hist_counts: torch.Tensor) -> Dict[int, List[torch.Tensor]]:
        r"""Generate multi-scale subgraphs.
        
        Args:
            adj: Adjacency matrix.
            s_ranks: Rank of nodes based on their dynamic centrality scores.
            hist_counts: Histogram of dynamic centrality scores.
        
        Returns:
            A dictionary of multi-scale subgraphs.
        """
        n_bins = hist_counts.shape[0]
        half_n_bins = n_bins // 2
        subgraphs_nodes: Dict[int, List[torch.Tensor]] = {}
        nodes_in_subg = torch.tensor([], dtype=torch.long)
        for i in range(half_n_bins):
            # Get the subgraphs with low-centrality central nodes.
            subgraphs_nodes[i] = self.get_subgraphs(adj, slice(i, i+1), s_ranks, hist_counts, self.n_orders)
            # Get the subgraphs with high-centrality central nodes.
            subgraphs_nodes[-i-1] = self.get_subgraphs(adj, slice(-i-1, -i), s_ranks, hist_counts, self.n_orders)
            
            # Check if the subgraphs cover all nodes.
            nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
            nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[-i-1])])
            if self.get_subgraphs_coverage(adj.shape[0], nodes_in_subg) < 1 - 1e-5:
                continue
            else:
                break
        
        return subgraphs_nodes

    def get_subgraphs(self, adj: torch.Tensor, bins: slice, s_ranks: torch.Tensor, hist_counts: torch.Tensor, n: int) -> List[torch.Tensor]:
        r"""Get the subgraphs based on the central nodes in the top bins.

        Args:
            adj: Adjacency matrix of the graph.
            bins: bins to consider.
                e.g. ``slice(-top_n_bins, None)`` to consider the top bins.
            s_ranks: Ranks of the nodes based on their centrality.
            hist_counts: Counts of the nodes in each bin.
            n: Number of orders of neighbors to find.
        
        Returns:
            List of subgraphs' nodes.
        """
        # Find central nodes in the top bins.
        central_nodes_index = self.get_central_nodes_index(s_ranks, hist_counts, bins)
        
        # Get the subgraphs.
        subgraphs_nodes: List[torch.Tensor] = []
        for central_node in central_nodes_index:
            assert central_node.dim() == 0
            subgraphs_nodes.append(self.find_n_order_neighbors(adj, central_node.long(), n))
        
        # Check the coverage.
        # self.get_subgraphs_coverage(adj.shape[0], subgraphs_nodes)

        # Check the overlap between subgraphs.
        # If any two subgraphs have more than threshold_subgraph_overlap nodes in common, merge them.
        subgraphs_nodes_o: List[torch.Tensor] = []
        n_subgraphs = len(subgraphs_nodes)
        for i in range(n_subgraphs):
            for j in range(i + 1, n_subgraphs):
                overlap_nodes = set(subgraphs_nodes[i].tolist()).intersection(set(subgraphs_nodes[j].tolist()))
                check_1 = len(overlap_nodes) / len(subgraphs_nodes[i]) > self.threshold_subgraph_overlap
                check_2 = len(overlap_nodes) / len(subgraphs_nodes[j]) > self.threshold_subgraph_overlap
                if check_1 or check_2:
                    _tmp_graph = torch.unique(torch.cat([subgraphs_nodes[i], subgraphs_nodes[j]]))
                    subgraphs_nodes_o.append(_tmp_graph)
                else:
                    subgraphs_nodes_o.append(subgraphs_nodes[i])
        
        return subgraphs_nodes_o
    
    def get_subgraphs_coverage(self, n_nodes: int, subgraphs_nodes: List[torch.Tensor] | torch.Tensor) -> float:
        r""" Get the coverage of the subgraphs.

        Args:
            n_nodes: Number of nodes in the graph.
            subgraphs_nodes: List of subgraphs' nodes.
        
        Returns:
            The node coverage ratio.
        """
        if type(subgraphs_nodes) == torch.Tensor:
            n_uniq_nodes = len(torch.unique(subgraphs_nodes))
        elif type(subgraphs_nodes) == list:
            n_uniq_nodes = len(torch.unique(torch.cat(subgraphs_nodes)))
        else:
            raise ValueError("subgraphs_nodes must be a list or a tensor.")
        coverage_ratio = n_uniq_nodes / n_nodes
        # Print the coverage ratio.
        print(f"\n-------- Node coverage: {coverage_ratio:.4f} --------\n")
        return coverage_ratio
    
    def find_n_order_neighbors(self, adj: torch.Tensor, start_node: torch.Tensor, n: int) -> torch.Tensor:
        """Find the n-order neighbors of a given node in a graph.
        
        Args:
            adj_matrix: Adjacency matrix with weights.
            start_node: Index of the starting node.
            n: Number of orders of neighbors to find.

        Returns:
            Indices of the n-order neighbors.
        """
        # current_neighbors = torch.tensor([start_node], dtype=torch.long)
        current_neighbors = start_node.unsqueeze(0)
        all_neighbors_set = set(current_neighbors.tolist())

        for _ in range(n):
            # The adjacency matrix row of the current neighbors.
            neighbor_rows = adj[current_neighbors]
            # Find the next-order neighbors by matrix multiplication.
            next_neighbors = torch.matmul(neighbor_rows, adj)
            # The indices of the non-zero elements are the next-order neighbors.
            next_neighbors = torch.nonzero(next_neighbors, as_tuple=False).squeeze()
            # Remove duplicates and convert to long type.
            current_neighbors = torch.unique(next_neighbors).long()
            # Add neighbors of the current order to the set
            all_neighbors_set.update(current_neighbors.tolist())
        
        all_neighbors = torch.tensor(list(all_neighbors_set), dtype=torch.long)
        return all_neighbors
    
    def get_central_nodes_index(self, s_ranks: torch.Tensor, hist_counts: torch.Tensor, bins: slice) -> torch.Tensor:
        r"""Get the indices of the central nodes in the specific bins.

        Args:
            s_ranks: Rank of the nodes based on their centrality. (``0`` is the smallest, ``-1`` is the largest)
            hist_counts: Histogram of the centrality values.
            bins: Bins' indices for the histogram. (From left(``0``) to right(``-1``). The rightmost bin is the most central.)
        
        Returns:
            Indices of the central nodes in the specific bins.
        """
        rank_start = torch.sum(hist_counts[:bins.start])
        rank_end = torch.sum(hist_counts[:bins.stop])
        # Find indices of s_ranks that elements are in the range of rank_start and rank_end.
        central_nodes_index = torch.where((s_ranks >= rank_start) & (s_ranks < rank_end))[0]
        return central_nodes_index

    def cosine_sim(self, h: torch.Tensor) -> torch.Tensor:
        r"""Compute the cosine similarity between all pairs of nodes in a graph.

        Args:
            h: Node features.

        Returns:
            Cosine similarity matrix.
        """
        norm_h = nn.functional.normalize(h, p=2, dim=1)
        cosine_simi = torch.matmul(norm_h, norm_h.T)
        # upper_triangular = torch.triu(torch.mm(norm_h, norm_h.T), diagonal=1)
        return cosine_simi
    
    def hist_fd(self, tensor1d: torch.Tensor) -> torch.Tensor:
        r"""Compute the histogram of a tensor using the Freedman-Diaconis rule for bin width.
        
        Args:
            tensor1d: A 1D tensor.
        
        Returns:
            A tensor of histogram counts.
        """
        iqr = tensor1d.quantile(0.75) - tensor1d.quantile(0.25)
        bin_width = 2 * iqr * tensor1d.numel() ** (-1 / 3)
        num_bins = int((tensor1d.max() - tensor1d.min()) / bin_width)

        hist_counts = torch.histc(tensor1d, bins=num_bins)
        # bin_edges = torch.linspace(tensor1d.min(), tensor1d.max(), steps=num_bins + 1)
        return hist_counts

r"""
xxx definition.
"""
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import graph_tool.all as gt


class XGAT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            dropout: float,
            negative_slope: float,
        ):
        r"""XGAT layer.

        Args:
            input_dim: Input node embedding dimension.
            output_dim: Output node embedding dimension.
            dropout: Dropout rate.
            negative_slope: LeakyReLU negative slope.
        """
        super().__init__()

        self.output_dim = output_dim

        self.W = nn.Parameter(torch.zeros(size=(input_dim * 2, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.attn = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.activation = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor]):
        n_nodes = h.shape[0]

        # Shape of W: [input_dim * 2, output_dim]
        # Shape of h: [n_nodes, input_dim]
        # Shape of Wh: [n_nodes, output_dim]

        # Repeat h to create all pairs (h_i, h_j)
        h_repeated = h.repeat(n_nodes, 1)
        h_i = h_repeated.view(n_nodes, n_nodes, -1)
        h_j = h_i.transpose(0, 1)

        # Concatenate h_i and h_j
        h_cat = torch.cat([h_i, h_j], dim=-1)

        # Apply linear transformation W
        Wh = torch.matmul(h_cat, self.W)
        Wh = Wh.view(n_nodes, n_nodes, self.output_dim)

        # Compute attention scores e_ij
        e_ = self.activation(torch.matmul(Wh, self.attn).squeeze(-1))

        if adj is not None:
            # Mask attention scores with adjacency matrix
            mask_ = adj.bool()
            attention = torch.where(mask_, e_, -9e15 * torch.ones_like(e_))
        
            # Apply softmax to attention scores
            # a_{ij} = softmax(M_{ij} * e_{ij})
            self.attention: torch.Tensor = self.softmax(attention * adj)
        
        else:
            self.attention: torch.Tensor = self.softmax(e_)

        # Apply dropout to attention scores
        attention = self.dropout(self.attention)
        
        # Compute transformed node embeddings
        h_prime = torch.matmul(attention, Wh)
        
        # Aggregate node embeddings
        h_prime = torch.mean(h_prime, dim=1)
        
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

        # assert len(output_dims) == len(n_heads), "output_dims and n_heads must have the same length."
        self.n_layers = len(output_dims)

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(XGAT(input_dim, output_dims[i], dropout, negative_slope))
            else:
                self.layers.append(XGAT(output_dims[i-1], output_dims[i], dropout, negative_slope))
        
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
        for i in range(self.n_layers):
            h = self.layers[i](h, adj)
            h_list.append(h)
        
        if self.skip_connections and self.Ws is not None:
            for i in range(self.n_layers - 2):
                h_list[i] = torch.matmul(h_list[i], self.Ws[i].weight)
            
            h_list[-1] = h_list[-1] + torch.mean(torch.stack(h_list[:-2]), dim=0)
        
        return h_list[-1]


def weighted_adj_to_gt(adj: torch.Tensor):
    r"""Converts weighted adjacency matrix ``adj`` to a graph using the ``graph-tool`` package.

    Args:
        adj: Weighted adjacency matrix of shape (num_nodes, num_nodes).
    
    Returns:
        gt.Graph: Graph using the `graph-tool` package.
    """
    if adj.is_cpu:
        adj_mat = adj.detach().numpy()
    else:
        adj_mat = adj.cpu().detach().numpy()
    
    g = gt.Graph(directed=False)
    # Add vertices
    num_vertices = adj_mat.shape[0]
    g.add_vertex(num_vertices)
    edge_weights = g.new_edge_property("double")

    # Add edges with weights
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if adj_mat[i][j] != 0:  # Check if there is an edge
                e = g.add_edge(g.vertex(i), g.vertex(j))
                edge_weights[e] = adj_mat[i][j]
    
    return g, edge_weights

def sparse_weighted_adj_to_gt(adj: torch.Tensor):
    r"""Converts weighted adjacency matrix ``adj`` to a graph using the ``graph-tool`` package.
    ``adj`` is a sparse tensor.
    
    Args:
        adj: Weighted adjacency matrix of shape (num_nodes, num_nodes).
    
    Returns:
        gt.Graph: Graph using the `graph-tool` package.

    """
    if adj.is_cpu:
        adj_mat = adj.detach()
    else:
        adj_mat = adj.cpu().detach()
    
    num_vertices = adj.shape[0]
    g = gt.Graph(directed=False)
    g.add_vertex(num_vertices)
    edge_weights = g.new_edge_property("double")
    for i in range(len(adj_mat.values())):
        e = g.add_edge(g.vertex(adj_mat.indices()[0][i].item()), g.vertex(adj_mat.indices()[1][i].item()))
        edge_weights[e] = adj_mat.values()[i].item()
    
    return g, edge_weights


def compute_C_BCE(g: gt.Graph, edge_w: gt.EdgePropertyMap) -> np.ndarray:
    r"""Computes the betweenness centrality, closeness centrality and eigenvector centrality of each node in the graph ``g``.
    """
    # print("Computing betweenness centrality...")
    c_B = np.array(gt.betweenness(g, weight=edge_w)[0])
    # print("Computing closeness centrality...")
    c_C = np.array(gt.closeness(g, weight=edge_w))
    # print("Computing eigenvector centrality...\n")
    c_E = np.array(gt.eigenvector(g, weight=edge_w)[1])
    n_nodes = g.num_vertices()
    out_array = np.zeros((n_nodes, 3))
    for i in range(n_nodes):
        out_array[i, 0] = c_B[i]
        out_array[i, 1] = c_C[i]
        out_array[i, 2] = c_E[i]
    return out_array


class DynamicCentrality(nn.Module):
    def __init__(self, node_emb_dim: int):
        r"""Dynamic centrality for multi-omics nodes (specific attributes and categories are considered):
        The weighted sum of the betweenness centrality, closeness centrality and eigenvector centrality
        of each node based on the node embedding.

        Args:
            node_emb_dim: Dimension of the node embedding.
        """
        super().__init__()

        self.Q_BCE = nn.Sequential(nn.Linear(node_emb_dim, 32), nn.Sigmoid(), nn.Linear(32, 3), nn.Softmax(dim=1))
        nn.init.xavier_uniform_(self.Q_BCE[0].weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.Q_BCE[2].weight.data, gain=1.414)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        r"""
        Returns:
            Dynamic centrality scores of shape ``(num_nodes, 1)``.
        """
        c_BCE = self.get_c_BCE(adj)
        w_BCE = self.Q_BCE(h)
        cs = torch.mul(w_BCE, c_BCE).sum(dim=1).unsqueeze(1)
        assert cs.shape == (h.shape[0], 1)

        # Square the values to avoid negative values and change the distribution to be more scale-free
        cs = torch.pow(cs, 2)
        return cs
    
    def get_c_BCE(self, adj: torch.Tensor) -> torch.Tensor:
        if adj.is_sparse:
            G, edge_w = sparse_weighted_adj_to_gt(adj)
        else:
            G, edge_w = weighted_adj_to_gt(adj)
        c_BCE = compute_C_BCE(G, edge_w)
        c_BCE_o = torch.tensor(c_BCE, dtype=torch.float32)
        # Normalize the values per column
        max3 = torch.max(c_BCE_o, dim=0)
        min3 = torch.min(c_BCE_o, dim=0)
        c_BCE_o = (c_BCE_o - min3[0]) / (max3[0] - min3[0])
        return c_BCE_o


class AttPool(nn.Module):
    def __init__(self, input_dim: int):
        r"""Apply attention to multi-scale subgraphs.
        
        """
        super().__init__()
        self.attention = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.attention.weight)
    
    def forward(self, multiscale_subg_emb: torch.Tensor):
        r"""
        Returns:
            Pooled representation of shape ``(output_dim_g_emb,)``.
        """
        weight_per_subg = self.attention(multiscale_subg_emb).softmax(dim=0)
        pooled = torch.mul(multiscale_subg_emb, weight_per_subg).sum(dim=0)
        return pooled


class MSGP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dims_nd: List[int],
            output_dim_g_emb: int,
            n_heads: List[int],
            n_hop: int,
            threshold_subgraph_overlap: float,
            dropout: float,
            negative_slope: float,
            use_all_subgraphs: bool = False,
        ):
        r"""Multi-Scale Graph Pooling for graph-level representation learning.

        Args:
            input_dim: Input node embedding dimension.

            output_dims_nd: Output node embedding dimensions for each layer.
                The length denotes the number of layers.
            
            output_dim_g_emb: Output graph embedding dimension.
            
            n_heads: Number of attention heads for each layer.
                The length must be the same as output_dims.
            
            n_hop: Maximum number of hops for searching central nodes' neighbors.
                ``n_hop >= 2`` is necessary to graph attention.
            
            threshold_subgraph_overlap: Threshold for the overlap between subgraphs.
            
            dropout: Dropout rate.
            
            negative_slope: LeakyReLU negative slope.
            
            use_all_subgraphs: Whether to use all subgraphs.
                If ``False``, only the largest and smallest subgraphs cover all nodes will be used.
        
        Returns:
            Graph representation.
        
        """
        super().__init__()
        if n_hop < 2:
            raise Warning("n_hop < 2")
        self.n_hop = n_hop
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
        self.use_all_subgraphs = use_all_subgraphs

        # Increasing dimensions
        self.xgat_layers = XGATLayers(input_dim, output_dims_nd, n_heads, dropout, negative_slope)
        self.dynamic_centrality = DynamicCentrality(output_dims_nd[-1])

        # Decreasing dimensions
        self.xgat_pool = XGAT(output_dims_nd[-1], output_dim_g_emb, dropout, negative_slope)
        self.att_pool = AttPool(output_dim_g_emb)

        self.global_xgat_pool = XGAT(output_dim_g_emb, output_dim_g_emb, dropout, negative_slope)
        self.global_att_pool = AttPool(output_dim_g_emb)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        # Node embedding
        h = self.xgat_layers(h, adj)
        
        # Nodes' dynamic centrality: normalized cosine similarity
        adj = self.cosine_sim(h).add(1).div(2).mul(adj).to_sparse()

        ds = self.dynamic_centrality(h, adj)
        s_ranks = torch.argsort(ds, dim=0, descending=False)
        
        # Histogram
        hist_counts = self.hist_fd(ds)
        # print(f"\nHistogram counts:\n{hist_counts}")

        # Remove blank bins
        hist_counts = hist_counts[hist_counts > 0]
        # print(f"\nHistogram counts after removing blank bins:\n{hist_counts}\n")
        assert hist_counts.shape[0] > 3

        # Generate multi-scale subgraphs
        subgraphs_nodes = self.collect_subgraphs(adj, s_ranks, hist_counts)
        keys_scales = list(subgraphs_nodes.keys())
        n_scales = len(keys_scales)
        # dk_scales = subgraphs_nodes.keys()
        assert n_scales > 1

        # Extract subgraphs for each scale
        subgraphs_nodes_flat = [subgraphs_nodes[i][j] for i in keys_scales for j in range(len(subgraphs_nodes[i]))]
        # print(f"\nNumber of subgraphs: {len(subgraphs_nodes_flat)}")

        # Embedding subgraphs
        # multiscale_subg_emb = torch.cat([self.pool_subgraph(h, subgraphs_nodes_flat[i], ds) for i in range(len(subgraphs_nodes_flat))], dim=0)
        multiscale_subg_emb = torch.stack([self.att_pool(self.xgat_pool(h[subgraphs_nodes_flat[i]], None)) for i in range(len(subgraphs_nodes_flat))])
        # print(f"\nShape of subgraphs pooling: {multiscale_subg_emb.shape}")

        # Subgraph pooling and graph-level representation via attention pooling.
        x = self.global_xgat_pool(multiscale_subg_emb, None)
        x = self.global_att_pool(x)
        # print(f"\nShape of graph-level embedding: {x.shape}")

        return x
    
    # def pool_subgraph(self, h: torch.Tensor, subgraph_nodes: torch.Tensor, ds: torch.Tensor) -> torch.Tensor:
    #     r"""Pool a subgraph to its graph representation.

    #     Args:
    #         h: Node embedding.
    #         subgraph_nodes: Nodes in the subgraph.
    #         ds: Dynamic centrality scores of nodes.
        
    #     Returns:
    #         Graph representation.
    #     """
    #     _softmax = nn.Softmax(dim=1)
    #     _s = _softmax(ds[subgraph_nodes])
    #     assert _s.shape[0] == len(subgraph_nodes)
    #     # Assigns attention weights to nodes and computes a weighted sum of node features
    #     pooled = torch.matmul(_s.T, h[subgraph_nodes])
    #     return pooled

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
        assert n_bins > 2
        subgraphs_nodes: Dict[int, List[torch.Tensor]] = {}
        nodes_in_subg = torch.tensor([], dtype=torch.long)

        if self.use_all_subgraphs:
            for i in range(n_bins):
                subgraphs_nodes[i] = self.get_subgraphs(adj, slice(i, i+1), s_ranks, hist_counts)
                assert len(subgraphs_nodes[i]) > 0
        else:
            # Add small subgraphs that are not connected to the main subgraph
            for i in range(n_bins // 3):
                subgraphs_nodes[i] = self.get_subgraphs(adj, slice(i, i+1), s_ranks, hist_counts)
                assert len(subgraphs_nodes[i]) > 0
                nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
            
            # Add "skeletons"
            for i in range(n_bins):

                # Get the subgraphs with high-centrality central nodes.
                i_desc = n_bins - i - 1
                if i_desc not in subgraphs_nodes.keys():
                    subgraphs_nodes[i_desc] = self.get_subgraphs(adj, slice(i_desc, n_bins - i), s_ranks, hist_counts)
                    assert len(subgraphs_nodes[i_desc]) > 0
                    nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i_desc])])
                    
                # Check if the subgraphs cover all nodes.
                if self.get_subgraphs_coverage(s_ranks.shape[0], nodes_in_subg) > 1 - 1e-5:
                    break

                # Get the subgraphs with low-centrality central nodes.
                if i not in subgraphs_nodes.keys():
                    subgraphs_nodes[i] = self.get_subgraphs(adj, slice(i, i+1), s_ranks, hist_counts)
                    assert len(subgraphs_nodes[i]) > 0
                    nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
                
        return subgraphs_nodes

    def get_subgraphs(self, adj: torch.Tensor, bins: slice, s_ranks: torch.Tensor, hist_counts: torch.Tensor, central_nodes_as_subg: bool=False) -> List[torch.Tensor]:
        r"""Get the subgraphs based on the central nodes in the top bins.

        Args:
            adj: Adjacency matrix of the graph.

            bins: bins to consider.
                e.g. ``slice(-top_n_bins, None)`` to consider the top bins.
            
            s_ranks: Ranks of the nodes based on their centrality.
            
            hist_counts: Counts of the nodes in each bin.

            central_nodes_as_subg: If True, return the central nodes as subgraphs.
        
        Returns:
            List of subgraphs' nodes.
        """
        # Find central nodes in the top bins.
        central_nodes_index = self.get_central_nodes_index(s_ranks, hist_counts, bins)
        # print(f"\nNumber of central nodes in the bins: {central_nodes_index.shape[0]}")
        
        # Get the subgraphs.
        subgraphs_nodes: Dict[int, torch.Tensor] = {}
        if central_nodes_as_subg:
            return [central_nodes_index]
        else:
            for i in range(central_nodes_index.shape[0]):
                subgraphs_nodes[i] = self.find_n_order_neighbors(adj, central_nodes_index[i])
        
        # Check the coverage.
        # self.get_subgraphs_coverage(adj.shape[0], subgraphs_nodes)

        # Check the overlap between subgraphs. If any two subgraphs have more than threshold_subgraph_overlap nodes in common, merge them.
        # print(f"\nChecking the overlap between subgraphs...")
        n_subgraphs = len(subgraphs_nodes)
        # print(f"  {n_subgraphs} subgraphs before merging.")
        
        subg2remove: List[int] = []
        tmp_merged_subgraphs: List[torch.Tensor] = []
        for i in range(n_subgraphs):
            for j in range(i + 1, n_subgraphs):
                overlap_nodes = set(subgraphs_nodes[i].tolist()).intersection(set(subgraphs_nodes[j].tolist()))
                check_1 = len(overlap_nodes) / len(subgraphs_nodes[i]) > self.threshold_subgraph_overlap
                check_2 = len(overlap_nodes) / len(subgraphs_nodes[j]) > self.threshold_subgraph_overlap
                if check_1 or check_2:
                    tmp_merged_subgraphs.append(torch.unique(torch.cat([subgraphs_nodes[i], subgraphs_nodes[j]])))
                    subg2remove.extend([i,j])
                    break
        
        subg2remove = list(set(subg2remove))
        subgraphs_nodes_o: List[torch.Tensor] = [subgraphs_nodes[i] for i in range(n_subgraphs) if i not in subg2remove]
        subgraphs_nodes_o.extend(tmp_merged_subgraphs)
        # print(f"  {len(subgraphs_nodes_o)} subgraphs after merging.\n")
        assert len(subgraphs_nodes_o) > 0
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
        # print(f"\n-------- Node coverage: {coverage_ratio:.4f} --------\n")
        return coverage_ratio
    
    def find_n_order_neighbors(self, adj: torch.Tensor, start_node: torch.Tensor) -> torch.Tensor:
        """Find the n-order neighbors of a given node in a graph.
        
        Args:
            adj_matrix: Adjacency matrix with weights.
            start_node: Index of the starting node.

        Returns:
            Indices of the n-order neighbors.
        """
        if adj.is_sparse:
            _adj = adj.to_dense()
        else:
            _adj = adj
        # _adj = (_adj > 0).float()
        # print(f"\nNumber of zeros in adjancency matrix: {torch.sum(adj == 0)}")
        
        all_neighbors_set = set()
        _n_hop_neighbors = start_node
        for n_hop in range(1, self.n_hop + 1):
            _n_hop_neighbors = torch.nonzero(_adj[_n_hop_neighbors]).squeeze().unique()
            all_neighbors_set.update(_n_hop_neighbors.tolist())
            # print(f"{len(all_neighbors_set)} neighbors found.")
        
        all_neighbors = torch.tensor(list(all_neighbors_set), dtype=torch.long)
        assert all_neighbors.shape[0] > 0
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
        central_nodes_index = torch.where((s_ranks >= rank_start) & (s_ranks < rank_end), s_ranks, -1)
        central_nodes_index = central_nodes_index[central_nodes_index != -1]
        # print(central_nodes_index)
        return central_nodes_index.long()

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

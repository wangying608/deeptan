r"""
xxx definition.
"""
from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data as GData
from torch_geometric.utils import k_hop_subgraph, softmax
import graph_tool.all as gt
from tqdm import tqdm
import frn.constants as const


class XGAT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            negative_slope: float,
        ):
        r"""XGAT layer.

        Args:
            input_dim: Input node embedding dimension.
            output_dim: Output node embedding dimension.
            negative_slope: LeakyReLU negative slope.
        """
        super().__init__()

        self.output_dim = output_dim

        self.W = nn.Parameter(torch.zeros(size=(input_dim * 2, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.attn = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, g: GData):
        assert g.x is not None

        n_nodes = g.num_nodes
        assert n_nodes is not None

        # Repeat h to create all pairs (h_i, h_j)
        h_repeated = g.x.repeat(n_nodes, 1)
        h_i = h_repeated.view(n_nodes, n_nodes, -1)
        h_j = h_i.transpose(0, 1)

        # Concatenate h_i and h_j
        h_cat = torch.cat([h_i, h_j], dim=-1)

        # Apply linear transformation W
        Wh = h_cat @ self.W
        Wh = Wh.view(n_nodes, n_nodes, self.output_dim)

        # Compute attention scores e_ij
        e_: torch.Tensor = self.activation(Wh.matmul(self.attn).squeeze(-1))

        if g.edge_index is None:
            self.attention: torch.Tensor = e_.softmax(dim=1)
        else:
            assert g.edge_attr is not None
            # Generate sparse adjacency matrix from edge_index
            # Mask attention scores with adjacency matrix (MIC)
            e_ = e_.mul(torch.sparse_coo_tensor(g.edge_index, g.edge_attr, size=(n_nodes, n_nodes))).coalesce()
            self.attention = torch.sparse_coo_tensor(
                indices=e_.indices(),
                values=softmax(src=e_.values(), index=e_.indices()[0], num_nodes=n_nodes),
                size=(n_nodes, n_nodes),
            )

        # Compute transformed node embeddings
        h_prime = self.attention @ Wh
        
        # Aggregate node embeddings
        h_prime = torch.mean(h_prime, dim=1)
        # h_prime_var = torch.var(h_prime, dim=1)
        
        return h_prime


class XGATLayer(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            negative_slope: float,
        ):
        r"""XGAT layer.

        Args:
            input_dim: Input node embedding dimension.
            output_dim: Output node embedding dimension.
            negative_slope: LeakyReLU negative slope.
        """
        super().__init__(aggr="add")

        self.output_dim = output_dim

        self.trans = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid(),
        )
        self.W = nn.Parameter(torch.zeros(size=(input_dim * 2, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.attn = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.activation = nn.LeakyReLU(negative_slope)
    
    def forward(self, g: GData):
        assert g.x is not None
        assert g.edge_index is not None
        assert g.edge_attr is not None
        n_nodes = g.num_nodes
        assert n_nodes is not None

        out = self.propagate(g.edge_index, x=g.x, edge_attr=g.edge_attr.view(-1,1))
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor):
        # edge_attr has shape: torch.Size([35003, 1])
        h_cat = torch.cat([x_i, x_j], dim=-1)
        # Shape of h_cat: (E, 2 * input_dim)
        Wh = (h_cat @ self.W)
        # Shape of Wh: (E, output_dim)
        e_ij: torch.Tensor = self.activation(Wh.matmul(self.attn))
        if edge_attr is None:
            e_ij = e_ij.softmax(dim=0)
        else:
            e_ij = e_ij.mul(edge_attr).softmax(dim=0)
        # Shape of e_ij: (E, 1)
        # Output shape: (E, output_dim)
        output: torch.Tensor = self.trans(x_j) * e_ij
        return output


class XGATLayers(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dims: List[int],
            n_heads: List[int],
            negative_slope: float,
        ):
        r"""XGAT layers with skip connections.

        Args:
            input_dim: Input node embedding dimension.

            output_dims: Output node embedding dimensions for each layer.
                The length denotes the number of layers.
            
            n_heads: Number of attention heads for each layer.
                The length must be the same as output_dims.
                        
            negative_slope: LeakyReLU negative slope.
        
        """
        super().__init__()

        self.n_layers = len(output_dims)

        self.layers = nn.ModuleList()
        for i in range(self.n_layers):
            if i == 0:
                self.layers.append(XGATLayer(input_dim, output_dims[i], negative_slope))
            else:
                self.layers.append(XGATLayer(output_dims[i-1], output_dims[i], negative_slope))
        
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
        
    def forward(self, g: GData) -> torch.Tensor:
        assert g.x is not None
        assert g.edge_attr is not None
        assert g.edge_index is not None

        _g = g.clone()
        h_list = []
        for i in range(self.n_layers):
            h = self.layers[i](_g)
            h_list.append(h)
            _g.x = h
        
        if self.skip_connections and self.Ws is not None:
            for i in range(self.n_layers - 2):
                h_list[i] = self.Ws[i](h_list[i])
            
            h_list[-1] = h_list[-1] + torch.mean(torch.stack(h_list[:-2]), dim=0)
        
        return h_list[-1]


def compute_C_BCE(g: gt.Graph, edge_w: gt.EdgePropertyMap) -> np.ndarray:
    r"""Computes the betweenness centrality, closeness centrality and eigenvector centrality of each node in the graph ``g``.
    """
    c_B = np.array(gt.betweenness(g, weight=edge_w)[0])
    c_C = np.array(gt.closeness(g, weight=edge_w))
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

        self.Q_BCE = nn.Sequential(nn.Linear(node_emb_dim, const.default.hidden_dim_dyn_cen), nn.Sigmoid(), nn.Linear(const.default.hidden_dim_dyn_cen, 3), nn.Softmax(dim=1))
        nn.init.xavier_uniform_(self.Q_BCE[0].weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.Q_BCE[2].weight.data, gain=1.414)
    
    def forward(self, g_pyg: GData) -> torch.Tensor:
        r"""
        Returns:
            Dynamic centrality scores of shape ``(num_nodes, 1)``.
        """
        assert g_pyg.x is not None

        c_BCE = self.get_c_BCE(g_pyg)
        w_BCE = self.Q_BCE(g_pyg.x)
        cs = torch.mul(w_BCE, c_BCE).sum(dim=1).unsqueeze(1)
        assert cs.shape == (g_pyg.x.shape[0], 1)

        # Square the values to avoid negative values and change the distribution to be more scale-free
        cs = torch.pow(cs, 2)
        return cs
    
    def get_c_BCE(self, g_pyg: GData) -> torch.Tensor:
        g, edge_weights = self.pyg2gtg(g_pyg)

        c_BCE = compute_C_BCE(g, edge_weights)
        c_BCE_o = torch.tensor(c_BCE, dtype=torch.float32)
        # Normalize the values per column
        max3 = torch.max(c_BCE_o, dim=0)
        min3 = torch.min(c_BCE_o, dim=0)
        c_BCE_o = (c_BCE_o - min3[0]) / (max3[0] - min3[0])
        return c_BCE_o

    def pyg2gtg(self, g_pyg: GData):
        assert g_pyg.x is not None
        assert g_pyg.edge_attr is not None
        assert g_pyg.edge_index is not None
        num_nodes = g_pyg.num_nodes
        assert num_nodes is not None

        g = gt.Graph(directed=False)
        g.add_vertex(num_nodes)
        edge_weights = g.new_edge_property("double")
        
        for i in range(g_pyg.edge_attr.shape[0]):
            e = g.add_edge(g.vertex(g_pyg.edge_index[0][i].item()), g.vertex(g_pyg.edge_index[1][i].item()))
            edge_weights[e] = g_pyg.edge_attr[i].item()
        
        return g, edge_weights


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
        pooled = multiscale_subg_emb.mul(weight_per_subg).sum(dim=0)
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
                        
            negative_slope: LeakyReLU negative slope.
            
            use_all_subgraphs: Whether to use all subgraphs.
                If ``False``, only the largest and smallest subgraphs cover all nodes will be used.
        
        Returns:
            Graph representation.
        
        """
        super().__init__()
        # if n_hop < 2:
        #     raise Warning("n_hop < 2")
        self.n_hop = n_hop
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
        self.use_all_subgraphs = use_all_subgraphs

        # Increasing dimensions
        self.xgat_layers = XGATLayers(input_dim, output_dims_nd, n_heads, negative_slope)
        self.dynamic_centrality = DynamicCentrality(output_dims_nd[-1])

        # Decreasing dimensions
        self.xgat_pool = XGAT(output_dims_nd[-1], output_dim_g_emb, negative_slope)
        self.att_pool = AttPool(output_dim_g_emb)

        self.global_xgat_pool = XGAT(output_dim_g_emb, output_dim_g_emb, negative_slope)
        self.global_att_pool = AttPool(output_dim_g_emb)
    
    def forward(self, g: GData):
        assert g.x is not None
        assert g.edge_attr is not None
        assert g.edge_index is not None

        # Node embedding
        h = self.xgat_layers(g)
        
        # Nodes' dynamic centrality: normalized cosine similarity
        adj = self.cosine_sim(h).add(1).div(2).mul(torch.sparse_coo_tensor(indices=g.edge_index, values=g.edge_attr, size=(h.shape[0], h.shape[0]))).to_sparse(layout=torch.sparse_coo).coalesce()

        g = GData(x=h, edge_index=adj.indices(), edge_attr=adj.values())

        ds = self.dynamic_centrality(g)
        s_ranks = torch.argsort(ds, dim=0, descending=False)
        
        # Histogram
        hist_counts = self.hist_fd(ds)
        # print(f"\nHistogram counts:\n{hist_counts}")

        # Remove blank bins
        hist_counts = hist_counts[hist_counts > 0]
        # print(f"\nHistogram counts after removing blank bins:\n{hist_counts}\n")
        assert hist_counts.shape[0] > 3

        # Generate multi-scale subgraphs
        subgraphs_nodes = self.collect_subgraphs(g, s_ranks, hist_counts)
        keys_scales = list(subgraphs_nodes.keys())
        n_scales = len(keys_scales)
        # dk_scales = subgraphs_nodes.keys()
        print(f"\nNumber of scales: {n_scales}\n")
        assert n_scales > 1

        # Extract subgraphs for each scale
        subgraphs_nodes_flat = [subgraphs_nodes[i][j] for i in keys_scales for j in range(len(subgraphs_nodes[i]))]
        # print(f"\nNumber of subgraphs: {len(subgraphs_nodes_flat)}")

        # Set edge indices and edge weights to None for flexible graph embedding
        g.edge_attr = None
        g.edge_index = None

        print("\nEmbedding subgraphs...")
        # Embedding subgraphs
        # multiscale_subg_emb = torch.stack([self.att_pool(self.xgat_pool(g.subgraph(subgraphs_nodes_flat[i]))) for i in range(len(subgraphs_nodes_flat))])
        
        # Optimized version for less memory usage
        _tmp_pooled: List[torch.Tensor] = []
        for i in tqdm(range(len(subgraphs_nodes_flat))):
            _tmp_pooled.append(self.att_pool(self.xgat_pool(g.subgraph(subgraphs_nodes_flat[i]))))
            # gc.collect()
            # torch.cuda.empty_cache()
        multiscale_subg_emb = torch.stack(_tmp_pooled)

        print("\nPooling subgraphs...")
        # Subgraph pooling and graph-level representation via attention pooling.
        x = self.global_xgat_pool(GData(x=multiscale_subg_emb))
        x = self.global_att_pool(x)
        # print(f"\nShape of graph-level embedding: {x.shape}")

        return x
    
    def collect_subgraphs(self, g: GData, s_ranks: torch.Tensor, hist_counts: torch.Tensor) -> Dict[int, List[torch.Tensor]]:
        r"""Generate multi-scale subgraphs.
        
        Args:
            g: Graph data.
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
                subgraphs_nodes[i] = self.get_subgraphs(g, slice(i, i+1), s_ranks, hist_counts)
                assert len(subgraphs_nodes[i]) > 0
        else:
            # Add small subgraphs that are not connected to the main subgraph
            for i in range(n_bins // 3):
                subgraphs_nodes[i] = self.get_subgraphs(g, slice(i, i+1), s_ranks, hist_counts)
                assert len(subgraphs_nodes[i]) > 0
                nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
            
            # Add "skeletons"
            for i in range(n_bins):

                # Get the subgraphs with high-centrality central nodes.
                i_desc = n_bins - i - 1
                if i_desc not in subgraphs_nodes.keys():
                    subgraphs_nodes[i_desc] = self.get_subgraphs(g, slice(i_desc, n_bins - i), s_ranks, hist_counts)
                    assert len(subgraphs_nodes[i_desc]) > 0
                    nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i_desc])])
                    
                # Check if the subgraphs cover all nodes.
                if self.get_subgraphs_coverage(s_ranks.shape[0], nodes_in_subg) > 1 - 1e-5:
                    break

                # Get the subgraphs with low-centrality central nodes.
                if i not in subgraphs_nodes.keys():
                    subgraphs_nodes[i] = self.get_subgraphs(g, slice(i, i+1), s_ranks, hist_counts)
                    assert len(subgraphs_nodes[i]) > 0
                    nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
        
        return subgraphs_nodes

    def get_subgraphs(self, g: GData, bins: slice, s_ranks: torch.Tensor, hist_counts: torch.Tensor, central_nodes_as_subg: bool=False) -> List[torch.Tensor]:
        r"""Get the subgraphs based on the central nodes in the top bins.

        Args:
            g: Graph data.

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
        assert central_nodes_index.shape[0] > 0
        # print(f"\nNumber of central nodes in the bins: {central_nodes_index.shape[0]}")
        # print(f"\nCentral nodes: {central_nodes_index}")
        
        assert g.x is not None
        assert g.edge_index is not None
        num_nodes = g.num_nodes
        assert num_nodes is not None

        # Get the subgraphs.
        subgraphs_nodes: Dict[int, torch.Tensor] = {}
        if central_nodes_as_subg:
            return [central_nodes_index]
        else:
            for i in range(central_nodes_index.shape[0]):
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=int(central_nodes_index[i].item()), num_hops=self.n_hop, edge_index=g.edge_index, num_nodes=num_nodes)
                # print(f"  ---- Subgraph {i}: {subset.shape[0]} nodes")
                subgraphs_nodes[i] = subset
        
        # Check the coverage.
        # self.get_subgraphs_coverage(adj.shape[0], subgraphs_nodes)

        # Check the overlap between subgraphs. If any two subgraphs have more than threshold_subgraph_overlap nodes in common, merge them.
        # print(f"\nChecking the overlap between subgraphs...")
        n_subgraphs = len(subgraphs_nodes)
        # print(f"\n  {n_subgraphs} subgraphs before merging.")
        
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
        # print(f"  Removing {len(subg2remove)} subgraphs.")
        print(f"  {len(subgraphs_nodes_o)} subgraphs after merging.\n")
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
        return central_nodes_index.long()

    def cosine_sim(self, h: torch.Tensor) -> torch.Tensor:
        r"""Compute the cosine similarity between all pairs of nodes in a graph.

        Args:
            h: Node features.

        Returns:
            Cosine similarity matrix.
        """
        norm_h = nn.functional.normalize(h, p=2, dim=1)
        cosine_simi = norm_h @ norm_h.T
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
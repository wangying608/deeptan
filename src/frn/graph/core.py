r"""
MSGP: Multi-Scale Graph Pooling for Graph-Level Representation Learning.
"""
from typing import List, Dict
# from tqdm import tqdm
import torch
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data as GData
from torch_geometric.utils import k_hop_subgraph, to_undirected
from frn.graph.modules import XGATLayer, XGATLayers, DynamicCentrality, AttPool


class MSGP(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dims_nd: List[int],
            output_dim_g_emb: int,
            n_heads: List[int],
            n_hop: int,
            threshold_edge_exist: float,
            threshold_subgraph_overlap: float,
            negative_slope: float,
            use_all_subgraphs: bool = True,
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
            
            threshold_edge_exist: Threshold for the existence of edges.
            
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
        if output_dim_g_emb < 8 or output_dim_g_emb % 4 != 0:
            raise ValueError("output_dim_g_emb must be greater than 8 and divisible by 4.")

        self.output_dim_g_emb = output_dim_g_emb
        self.n_hop = n_hop
        self.threshold_edge_exist = threshold_edge_exist
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
        self.use_all_subgraphs = use_all_subgraphs

        # Increasing dimensions
        self.xgat_layers = XGATLayers(input_dim, output_dims_nd, n_heads, negative_slope)
        
        self.dynamic_centrality = DynamicCentrality(output_dims_nd[-1])

        # Decreasing dimensions
        self.xgat_pool = XGATLayer(output_dims_nd[-1], output_dim_g_emb, negative_slope)
        self.att_pool = AttPool(output_dim_g_emb)

        self.global_xgat_pool = XGATLayer(output_dim_g_emb * 4, output_dim_g_emb // 4, negative_slope)
        self.global_att_pool = AttPool(output_dim_g_emb // 4)
    
    def forward(self, g: GData):
        if g.x is None:
            raise ValueError("g.x cannot be None")
        if g.edge_attr is None:
            raise ValueError("g.edge_attr cannot be None")
        if g.edge_index is None:
            raise ValueError("g.edge_index cannot be None")

        # Node embedding
        h: torch.Tensor = self.xgat_layers(g)
        
        # Nodes' dynamic centrality: normalized cosine similarity
        adj = self.cosine_sim(h).relu().mul(torch.sparse_coo_tensor(indices=g.edge_index, values=g.edge_attr, size=(h.shape[0], h.shape[0]))).to_sparse(layout=torch.sparse_coo).coalesce()
        ew_min, ew_max = adj.values().aminmax()
        norm_ew = (adj.values() - ew_min) / (ew_max - ew_min)

        # Apply edge mask
        edge_mask = norm_ew > self.threshold_edge_exist

        g = GData(x=h, edge_index=adj.indices()[:, edge_mask], edge_attr=norm_ew[edge_mask])
        if g.edge_index is None:
            raise ValueError("g.edge_index cannot be None.")
        if g.edge_attr is None:
            raise ValueError("g.edge_attr cannot be None.")
        num_nodes = g.num_nodes
        if num_nodes is None:
            raise ValueError("Number of nodes is not provided.")

        ds = self.dynamic_centrality(g.x, g.edge_attr, g.edge_index, num_nodes)
        s_ranks = torch.argsort(ds, dim=0, descending=False)
        
        # Histogram
        hist_counts = self.hist_fd(ds)
        # Remove blank bins
        hist_counts = hist_counts[hist_counts > 0]
        if hist_counts.shape[0] < 4:
            raise ValueError("hist_counts must have more than 3 elements.")

        # Generate multi-scale subgraphs
        subgraphs_nodes = self.collect_subgraphs(g.edge_index, num_nodes, s_ranks, hist_counts)
        keys_scales = list(subgraphs_nodes.keys())
        n_scales = len(keys_scales)
        if n_scales < 2:
            raise ValueError("n_scales must be greater than 1.")
        # dk_scales = subgraphs_nodes.keys()
        # print(f"\nNumber of scales: {n_scales}\n")

        # Extract subgraphs for each scale
        subgraphs_nodes_flat = [node for sublist in subgraphs_nodes.values() for node in sublist]
        # print(f"\nNumber of subgraphs: {len(subgraphs_nodes_flat)}")

        # Set edge indices and edge weights to None for flexible graph embedding
        g_ = g
        g.edge_attr = None
        if g_.x is None:
            raise ValueError("g_.x cannot be None")

        # Embedding subgraphs
        num_subg = len(subgraphs_nodes_flat)
        _pooled: List[torch.Tensor] = []
        
        # for i in tqdm(range(num_subg)):
        for i in range(num_subg):
            subgraph = g.subgraph(subgraphs_nodes_flat[i])
            
            def forward_fn(subgraph):
                subgraph = self.xgat_pool(subgraph)
                subgraph = self.att_pool(subgraph)
                return subgraph
            
            subgraph = checkpoint(forward_fn, subgraph, use_reentrant=False)
            if not isinstance(subgraph, torch.Tensor):
                raise ValueError("subgraph must be a tensor.")
            _pooled.append(subgraph)
            del subgraph
        
        torch.cuda.empty_cache()
        multiscale_subg_emb = torch.stack(_pooled)

        # Subgraph pooling and graph-level representation via attention pooling.
        g_super_nodes = GData(x=multiscale_subg_emb, edge_index=to_undirected(torch.combinations(torch.arange(num_subg, device=g_.x.device)).t()))
        x = self.global_xgat_pool(g_super_nodes)
        x = self.global_att_pool(x)
        
        return x, g_, g_super_nodes

    def collect_subgraphs(self, edge_index: torch.Tensor, num_nodes: int, s_ranks: torch.Tensor, hist_counts: torch.Tensor) -> Dict[int, List[torch.Tensor]]:
        r"""Generate multi-scale subgraphs.
        
        Args:
            edge_index: Edge index of the graph.
            num_nodes: Number of nodes in the graph.
            s_ranks: Rank of nodes based on their dynamic centrality scores.
            hist_counts: Histogram of dynamic centrality scores.
        
        Returns:
            A dictionary of multi-scale subgraphs.
        """
        n_bins = hist_counts.shape[0]
        if n_bins < 4:
            raise ValueError("n_bins must be greater than 3.")
        subgraphs_nodes: Dict[int, List[torch.Tensor]] = {}
        nodes_in_subg = torch.tensor([], dtype=torch.long, device=edge_index.device)

        if self.use_all_subgraphs:
            for i in range(n_bins):
                central_nodes_indices = self.get_central_nodes_indices(s_ranks, hist_counts, slice(i, i+1))
                subgraphs_nodes[i] = self.get_subgraphs(edge_index, num_nodes, central_nodes_indices)
                if len(subgraphs_nodes[i]) == 0:
                    raise ValueError("subgraphs_nodes[i] must not be empty.")
        else:
            # Add small subgraphs that are not connected to the main subgraph
            # for i in range(n_bins // 3):
            #     subgraphs_nodes[i] = self.get_subgraphs(g, slice(i, i+1), s_ranks, hist_counts)
            #     assert len(subgraphs_nodes[i]) > 0
            #     nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])

            #     # Check if the subgraphs cover all nodes.
            #     if self.get_subgraphs_coverage(s_ranks.shape[0], nodes_in_subg) > 1 - 1e-5:
            #         break

            central_nodes_indices = self.get_central_nodes_indices(s_ranks, hist_counts, slice(0, 1))
            subgraphs_nodes[0] = self.get_subgraphs(edge_index, num_nodes, central_nodes_indices)
            if len(subgraphs_nodes[0]) == 0:
                raise ValueError("subgraphs_nodes[0] must not be empty.")
            nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[0])])
            
            # Add "skeletons"
            for i in range(n_bins):

                # Check if the subgraphs cover all nodes.
                if self.get_subgraphs_coverage(s_ranks.shape[0], nodes_in_subg) > 1 - 1e-5:
                    break

                # Get the subgraphs with high-centrality central nodes.
                i_desc = n_bins - i - 1
                if i_desc not in subgraphs_nodes.keys():
                    central_nodes_indices = self.get_central_nodes_indices(s_ranks, hist_counts, slice(i_desc, n_bins - i))
                    subgraphs_nodes[i_desc] = self.get_subgraphs(edge_index, num_nodes, central_nodes_indices)
                    if len(subgraphs_nodes[i_desc]) == 0:
                        raise ValueError("subgraphs_nodes[i_desc] must not be empty.")
                    nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i_desc])])
                    
                # Check if the subgraphs cover all nodes.
                # if self.get_subgraphs_coverage(s_ranks.shape[0], nodes_in_subg) > 1 - 1e-5:
                #     break

                # Get the subgraphs with low-centrality central nodes.
                # if i not in subgraphs_nodes.keys():
                #     subgraphs_nodes[i] = self.get_subgraphs(g, slice(i, i+1), s_ranks, hist_counts)
                #     assert len(subgraphs_nodes[i]) > 0
                #     nodes_in_subg = torch.cat([nodes_in_subg, torch.cat(subgraphs_nodes[i])])
        
        return subgraphs_nodes

    def get_subgraphs(self, edge_index: torch.Tensor, num_nodes: int, central_nodes_indices: torch.Tensor, central_nodes_as_subg: bool=False) -> List[torch.Tensor]:
        r"""Get the subgraphs based on the central nodes in the top bins.

        Args:
            edge_index: Edge index of the graph.
            num_nodes: Number of nodes in the graph.
            central_nodes_indices: Indices of the central nodes.
            central_nodes_as_subg: If True, return the central nodes as subgraphs.
        
        Returns:
            List of subgraphs' nodes.
        """
        # Find central nodes in the top bins.
        num_central_nodes = central_nodes_indices.shape[0]
        if num_central_nodes < 1:
            raise ValueError("No central nodes found.")
        
        # Get the subgraphs.
        subgraphs_nodes: Dict[int, torch.Tensor] = {}
        if central_nodes_as_subg:
            return [central_nodes_indices]
        else:
            for i in range(num_central_nodes):
                subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx=int(central_nodes_indices[i].item()), num_hops=self.n_hop, edge_index=edge_index, num_nodes=num_nodes)
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
        # print(f"  {len(subgraphs_nodes_o)} subgraphs after merging.\n")
        if len(subgraphs_nodes_o) == 0:
            raise ValueError("subgraphs_nodes_o must not be empty.")
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
        
    def get_central_nodes_indices(self, s_ranks: torch.Tensor, hist_counts: torch.Tensor, bins: slice) -> torch.Tensor:
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
        norm_h = torch.nn.functional.normalize(h, p=2, dim=1)
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

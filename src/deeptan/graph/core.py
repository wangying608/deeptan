r"""
AMSGP: Adaptive Multi-Scale Graph Pooling for Graph-Level Representation Learning.
"""

from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from torch_geometric.data import Data as GData
from torch_geometric.utils import k_hop_subgraph, to_undirected, subgraph
from deeptan.graph.modules import WGATLayer, NodeEmbedding, SelfAttPool


class AMSGP(torch.nn.Module):
    r"""Enhanced Adaptive Multi-Scale Graph Pooling with dynamic subgraph sampling."""

    def __init__(
        self,
        dict_node_names: Dict[str, int],
        input_dim: int,
        node_emb_dim: int,
        fusion_dims_node_emb: List[int],
        n_heads_node_emb: int,
        output_dim_g_emb: int,
        n_heads_pooling: int,
        n_hop: int,
        threshold_edge_exist: float,
        threshold_subgraph_overlap: float,
        negative_slope: float,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dict_node_names = dict_node_names
        self.output_dim_g_emb = output_dim_g_emb
        self.n_hop = n_hop
        self.threshold_edge_exist = threshold_edge_exist
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
        self.dropout = dropout

        # Enhanced node embedding with dropout
        self.node_embedding_layers = NodeEmbedding(
            input_dim,
            node_emb_dim,
            fusion_dims_node_emb,
            dict_node_names,
            n_heads_node_emb,
            negative_slope,
            dropout,
        )

        # Multi-scale pooling architecture
        self._init_pooling_layers(
            fusion_dims_node_emb[-1], output_dim_g_emb, negative_slope, n_heads_pooling
        )

    def _init_pooling_layers(self, input_dim, output_dim, slope, heads):
        # Local subgraph pooling
        self.xgat_pool = WGATLayer(input_dim, output_dim, slope, heads, self.dropout)
        self.att_pool = SelfAttPool(output_dim)

        # Global graph pooling
        self.global_xgat_pool = WGATLayer(
            output_dim, output_dim, slope, heads, self.dropout
        )
        self.global_att_pool = SelfAttPool(output_dim)

    def forward(self, node_names, x, edge_attr, edge_index):
        # Feature regularization
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Node embedding with layer norm
        h = self.node_embedding_layers(node_names, x, edge_attr, edge_index)
        h = F.layer_norm(h, h.size()[1:])

        # Dynamic subgraph generation
        filtered_edge_index, centrality = self._calculate_dynamic_centrality(
            h, edge_attr, edge_index
        )
        subgraphs = self._generate_multiscale_subgraphs(
            filtered_edge_index, x.size(0), centrality, h, x.device
        )

        # Fallback for empty subgraphs
        if not subgraphs:
            return torch.zeros((x.size(0), self.output_dim_g_emb), device=x.device)

        g_emb = self._create_graph_embeddings(subgraphs)

        return g_emb, self.node_embedding_layers.embed

    def _calculate_dynamic_centrality(self, h, edge_attr, edge_index):
        # Calculate edge importance
        row, col = edge_index
        h_i, h_j = h[row], h[col]
        cos_sim = F.cosine_similarity(h_i, h_j).abs()

        # Combine with edge attributes
        edge_weight = cos_sim * edge_attr.view(-1) if edge_attr is not None else cos_sim

        # Filter edges
        mask = edge_weight > self.threshold_edge_exist
        filtered_edge = edge_index[:, mask]
        filtered_weight = edge_weight[mask]

        # Compute node centrality
        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.scatter_add_(0, filtered_edge[0], filtered_weight)
        centrality.scatter_add_(0, filtered_edge[1], filtered_weight)
        return filtered_edge, centrality

    def _generate_multiscale_subgraphs(
        self, edge_index, num_nodes, centrality, h, device
    ):
        # Adaptive binning using quantiles
        q_low, q_high = torch.quantile(
            centrality, torch.tensor([0.2, 0.8], device=device)
        )
        central_nodes = torch.where(centrality > q_high)[0]

        # Parallel subgraph generation with index remapping
        subgraphs = []
        for node_idx in central_nodes:
            subset, subg_edge_idx, _, _ = k_hop_subgraph(
                int(node_idx.item()),
                self.n_hop,
                edge_index,
                num_nodes=num_nodes,
                flow="source_to_target",
            )

            # Remap edge indices to local subgraph indices
            node_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device)
            node_mapping[subset] = torch.arange(len(subset), device=device)
            subg_edge_local = node_mapping[subg_edge_idx]

            # Skip isolated nodes or invalid subgraphs
            if len(subset) < 2 or subg_edge_local.size(1) == 0:
                continue

            subgraphs.append(
                GData(
                    x=h[subset],
                    edge_index=subg_edge_local,
                    center_node=node_idx,
                    node_idx=subset,
                )
            )

        # Node coverage-based merging
        merged = []
        coverage = torch.zeros(num_nodes, device=device)
        for subg in sorted(subgraphs, key=lambda x: -x.num_nodes):
            overlap = coverage[subg.node_idx].mean()
            if overlap < self.threshold_subgraph_overlap:
                merged.append(subg)
                coverage[subg.node_idx] += 1
        return merged

    def _create_graph_embeddings(self, subgraphs):
        # Process each subgraph
        pooled = [self._process_subgraph(g) for g in subgraphs]
        emb_stack = torch.stack(pooled)

        # Build super graph
        super_nodes = GData(
            x=emb_stack,
            edge_index=to_undirected(
                torch.combinations(
                    torch.arange(len(subgraphs), device=emb_stack.device)
                ).t()
            ),
        )

        # Global pooling
        global_emb = self.global_xgat_pool(super_nodes.x, super_nodes.edge_index)
        return self.global_att_pool(global_emb)

    def _process_subgraph(self, subgraph):
        assert subgraph.edge_index.max() < subgraph.x.size(0), (
            "Edge index exceeds node count"
        )
        h = self.xgat_pool(subgraph.x, subgraph.edge_index)
        return self.att_pool(h.unsqueeze(0)).squeeze(0)

    def hist_fd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freedman-Diaconis histogram calculation."""
        q75, q25 = torch.quantile(
            tensor, torch.tensor([0.75, 0.25], device=tensor.device)
        )
        iqr = q75 - q25
        bin_width = 2 * iqr * (tensor.numel() ** (-1 / 3))
        num_bins = max(1, int((tensor.max() - tensor.min()) / (bin_width + 1e-8)))
        counts = torch.histc(tensor, bins=num_bins)
        edges = torch.linspace(
            tensor.min(), tensor.max(), num_bins + 1, device=tensor.device
        )
        return counts, edges

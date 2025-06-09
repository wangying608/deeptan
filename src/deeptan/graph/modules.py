r"""
Modules for DeepTAN.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected

import deeptan.constants as const
from deeptan.utils.uni import GetAdaptiveChunkSize


class AMSGP(torch.nn.Module):
    r"""
    AMSGP: Adaptive Multi-Scale Graph Pooling for Graph-Level Representation Learning.
    """

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
        dropout: float = const.default.dropout,
        chunk_size: int = const.default.chunk_size,
    ):
        r"""
        Initialize the AMSGP model.
        Args:
            dict_node_names: A dictionary mapping node names to their indices.
            input_dim: The dimension of the input node features.
            node_emb_dim: The dimension of the node embeddings.
            fusion_dims_node_emb: A list of dimensions for the fusion layers in the node embedding.
            n_heads_node_emb: The number of attention heads for the node embedding.
            output_dim_g_emb: The dimension of the output graph embedding.
            n_heads_pooling: The number of attention heads for the pooling layers.
            n_hop: The number of hops for subgraph extraction.
            threshold_edge_exist: The threshold for edge existence in subgraphs.
            threshold_subgraph_overlap: The threshold for subgraph overlap.
            dropout: The dropout rate for the model.
            chunk_size: The chunk size for parallel processing.
        """
        super().__init__()
        self.dict_node_names = dict_node_names
        self.output_dim_g_emb = output_dim_g_emb
        self.n_hop = n_hop
        self.thre_edge_exist = threshold_edge_exist
        self.thre_sg_overlap = threshold_subgraph_overlap
        self.dropout = dropout
        self.chunk_size = chunk_size
        self.node_emb_dim = node_emb_dim
        self.n_heads_node_emb = n_heads_node_emb
        self.n_heads_pooling = n_heads_pooling

        # Node embedding
        self.node_embedding_layers = NodeEmbedding(
            input_dim,
            node_emb_dim,
            fusion_dims_node_emb,
            dict_node_names,
            n_heads_node_emb,
            dropout,
            chunk_size,
        )

        # Multi-scale pooling architecture
        self._init_pooling_layers(fusion_dims_node_emb[-1], output_dim_g_emb, n_heads_pooling)

        # Adaptive chunk size calculation
        self.adap_chunk_size_dyn_cent = GetAdaptiveChunkSize()
        self.adap_chunk_size_mul_subg = GetAdaptiveChunkSize()

    def _init_pooling_layers(self, input_dim, output_dim, heads):
        # Local subgraph pooling
        self.xgat_pool = WGATLayer_chunked(input_dim, output_dim, heads, self.dropout, self.chunk_size)
        self.att_pool = SelfAtt_(output_dim, self.dropout, True)

        # Global graph pooling
        self.global_xgat_pool = WGATLayer_chunked(output_dim, output_dim, heads, self.dropout, self.chunk_size)
        self.global_att_pool = SelfAtt_(output_dim, self.dropout, True)

    def forward(self, node_names, x, edge_attr, edge_index, batch):
        # Node embedding with layer norm
        h, E_all, ids = self.node_embedding_layers(node_names, x, edge_attr, edge_index)

        # Graph embedding
        unique_batches = torch.unique(batch)
        graph_embs = []
        self._device = x.device

        for graph_id in unique_batches:
            # Extract node mask for the current graph
            mask = batch == graph_id
            node_indices = torch.where(mask)[0]

            # Skip empty graphs
            if node_indices.numel() == 0:
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)

            # Validate subgraph edge indices
            if sub_edge_index.numel() == 0:
                print("Warning: Empty subgraph detected")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            # Compute dynamic centrality
            h_masked = h[mask]
            filtered_edge_index, centrality = self._calculate_dynamic_centrality(h_masked, sub_edge_index)

            # Generate multiscale subgraphs
            subgraphs = self._generate_multiscale_subgraphs(filtered_edge_index, centrality, h_masked)

            # Create graph embeddings
            if subgraphs:
                g_emb = self._create_graph_embeddings(subgraphs)
                graph_embs.append(g_emb)
            else:
                # Process empty subgraph case
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))

        # Stack all graph embeddings
        return torch.stack(graph_embs), E_all, ids

    def _calculate_dynamic_centrality(self, h, edge_index):
        chunk_size = self.adap_chunk_size_dyn_cent.calc(h.size(), 0)

        row, col = edge_index
        num_edges = edge_index.size(1)
        sim_ = []

        # Calculate similarity for each edge in batches to reduce peak memory usage.
        for i in range(0, num_edges, chunk_size):
            idx = slice(i, min(i + chunk_size, num_edges))
            h_i = h[row[idx]]
            h_j = h[col[idx]]

            sim_.append((h_i * h_j).sum(dim=1).abs())

        sim_ = torch.cat(sim_, dim=0)

        min_sim_ = sim_.min()
        max_sim_ = sim_.max()
        sim_ = (sim_ - min_sim_) / (max_sim_ - min_sim_) if max_sim_ - min_sim_ != 0 else sim_

        # Filter edges
        mask = sim_ > self.thre_edge_exist

        filtered_edge = edge_index[:, mask]
        filtered_weight = sim_[mask]

        # Compute node centrality
        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.scatter_add_(0, filtered_edge[0], filtered_weight)
        centrality.scatter_add_(0, filtered_edge[1], filtered_weight)

        return filtered_edge, centrality

    def _generate_multiscale_subgraphs(self, edge_index, centrality, h) -> List[GData]:
        """Generate hierarchical subgraphs using centrality histogram bins.

        基于节点中心性的多尺度子图生成算法，通过分层处理和子图合并策略构建层次化子图结构。

        This method generates multiscale subgraphs by dividing nodes into bins based on their centrality.
        It processes nodes in descending order of centrality and merges subgraphs with significant overlap.
        Subgraphs are created and stored in a pool, and the method ensures that all nodes are covered.

        根据Freedman-Diaconis规则将不同中心性的节点分组，按节点中心性从高到低的顺序逐个作为中心节点生成k跳子图，
        收集到覆盖全部节点的多尺度子图，合并重叠程度超过指定阈值的子图，最终返回按节点数降序排列的子图列表。
        生成的子图在后续的图嵌入过程中扮演重要角色，帮助模型更好地理解图的全局结构。

        Args:
            edge_index: The edge index tensor representing the graph edges.
            centrality: The centrality scores of the nodes.
            h: The node embeddings.

        Returns:
            A list of subgraphs sorted by the number of nodes in descending order.
        """
        device = h.device
        num_nodes = h.size(0)

        # Step 1: Compute FD bins
        _, edges = self.hist_fd(centrality)
        num_bins = len(edges) - 1
        covered_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Step 2: Initialize subgraph pool
        subgraph_masks = []
        subgraph_centers = []

        # Step 3: Process each bin in descending order of centrality
        for bin_idx in reversed(range(num_bins)):
            bin_mask = self._generate_bin_mask(centrality, edges, bin_idx)
            current_nodes = torch.where(bin_mask)[0]
            n_current = current_nodes.numel()
            if n_current == 0:
                continue

            chunk_size = self.adap_chunk_size_mul_subg.calc((n_current, edge_index.shape[1], h.shape[1]), 0)

            # Step 4: Batch process nodes
            for chunk in current_nodes.split(chunk_size):
                # Get k-hop subgraphs for this batch of nodes
                subsets, subg_edge_indices = self._batch_k_hop_subgraph(chunk, self.n_hop, edge_index, num_nodes)

                for subset, center_node in zip(subsets, chunk.tolist()):
                    if subset.numel() < 2:
                        continue

                    new_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                    new_mask[subset] = True

                    # Step 5: Vectorized overlap detection
                    overlapping_indices = []
                    if subgraph_masks:
                        existing_masks = torch.stack(subgraph_masks)
                        intersections = (existing_masks & new_mask).sum(dim=1)
                        min_sizes = torch.min(existing_masks.sum(dim=1), torch.full_like(intersections, new_mask.sum()))
                        overlaps = intersections / (min_sizes + 1e-8)
                        overlapping_indices = (overlaps > self.thre_sg_overlap).nonzero().squeeze(1).tolist()

                    # Step 6: Merge or add
                    if overlapping_indices.__len__() > 0:
                        merged_mask = new_mask.clone()
                        for idx in sorted(overlapping_indices, reverse=True):
                            merged_mask |= subgraph_masks[idx]
                            del subgraph_masks[idx], subgraph_centers[idx]
                        subgraph_masks.append(merged_mask)
                        subgraph_centers.append(center_node)
                    else:
                        subgraph_masks.append(new_mask)
                        subgraph_centers.append(center_node)

                    # Step 7: Update coverage
                    covered_nodes |= new_mask

                    if covered_nodes.all():
                        break
                if covered_nodes.all():
                    break
            if covered_nodes.all():
                break

        # Step 8: Convert to GData objects
        subgraph_data = []
        for mask, center in zip(subgraph_masks, subgraph_centers):
            node_count = mask.sum().item()
            subgraph_data.append((node_count, mask, center))

        # Sort by node count descending
        subgraph_data.sort(key=lambda x: x[0], reverse=True)

        # Step 9: Create GData objects in sorted order
        subgraphs = [
            GData(
                x=h[mask],
                edge_index=self._fast_edge_index(edge_index, mask),
                center_node=center,
                node_idx=torch.where(mask)[0],
                mask=mask,
            )
            for _, mask, center in subgraph_data
        ]

        return subgraphs

    @staticmethod
    @torch.jit.script
    def _generate_bin_mask(centrality: torch.Tensor, edges: torch.Tensor, bin_idx: int) -> torch.Tensor:
        return (centrality >= edges[bin_idx]) & (centrality <= edges[bin_idx + 1])

    @staticmethod
    def _fast_edge_index(edge_index: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sub_edge_index, _ = subgraph(mask, edge_index, relabel_nodes=True)
        return sub_edge_index

    @staticmethod
    def _batch_k_hop_subgraph(nodes: torch.Tensor, num_hops: int, edge_index: torch.Tensor, num_nodes: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Custom implementation of batch k-hop subgraph extraction."""
        nodes_cpu = nodes.cpu().numpy().tolist()
        subsets = []
        edge_indices = []

        for _node in nodes_cpu:
            subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=_node, num_hops=num_hops, edge_index=edge_index, num_nodes=num_nodes, relabel_nodes=True)
            subsets.append(subset)
            edge_indices.append(sub_edge_index)
        return subsets, edge_indices

    @staticmethod
    def hist_fd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freedman-Diaconis rule with memory optimization."""
        n = tensor.numel()

        # Calculate statistics on GPU first
        min_val = tensor.min()
        max_val = tensor.max()
        q75, q25 = torch.quantile(tensor, torch.tensor([0.75, 0.25], device=tensor.device))
        iqr = q75 - q25

        # Move scalar values to CPU for bin calculation
        min_cpu = min_val.cpu().item()
        max_cpu = max_val.cpu().item()
        iqr_cpu = iqr.cpu().item()
        n_cpu = n  # numel() returns Python int

        # Calculate bin parameters on CPU
        bin_width = 2 * iqr_cpu / (n_cpu ** (1 / 3)) if iqr_cpu > 0 else (max_cpu - min_cpu)
        num_bins = int((max_cpu - min_cpu) / bin_width) if bin_width > 0 else 1
        num_bins = max(1, min(num_bins, 1000))  # Cap bins to 1000

        # Create edges on CPU first
        edges_cpu = torch.linspace(min_cpu, max_cpu, num_bins + 1)
        edges = edges_cpu.to(device=tensor.device)

        # Calculate histogram on GPU
        hist = torch.histc(tensor.cpu(), bins=num_bins, min=min_cpu, max=max_cpu)
        return hist.to(tensor.device), edges

    def _create_graph_embeddings(self, subgraphs):
        num_subgraphs = len(subgraphs)
        if num_subgraphs == 0:
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        all_embs = [self._process_subgraph(g) for g in subgraphs]
        embs = torch.cat(all_embs, dim=0) if all_embs else torch.zeros(0, self.output_dim_g_emb, device=self._device)

        # Build super graph
        edge_index = to_undirected(torch.combinations(torch.arange(num_subgraphs, device=embs.device)).t())
        super_nodes = GData(x=embs, edge_index=edge_index)

        # Global pooling with chunked edge processing
        global_emb = self.global_xgat_pool(super_nodes.x, super_nodes.edge_index)
        return self.global_att_pool(global_emb)

    def _process_subgraph(self, subgraph):
        h = self.xgat_pool(subgraph.x, subgraph.edge_index)
        return self.att_pool(h.unsqueeze(0))


class NodeEmbedding(nn.Module):
    r"""
    Embedding nodes in a graph like embedding words in a sentence.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        fusion_dims: List[int],
        dict_node_names: Dict[str, int],
        n_heads: int,
        dropout: float = const.default.dropout,
        chunk_size: int = const.default.chunk_size,
    ):
        r"""
        Embedding nodes in a graph like embedding words in a sentence.

        Args:
            input_dim: Dimension of input features.
            embedding_dim: Dimension of the embedding.
            fusion_dims: Dimensions for the fusion (fusing observation value and inherent feature) layers.
            dict_node_names: Dictionary mapping node names to indices.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
            chunk_size: Size of chunks for large tensors.
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.x_increased_dim = embedding_dim // 2
        self.fusion_dims = fusion_dims
        self.dict_node_names = dict_node_names
        self.n_heads = n_heads

        self.embed = nn.Embedding(len(dict_node_names), embedding_dim, scale_grad_by_freq=True, sparse=True)

        # Embedding feature values like position embeddings
        self.quant_emb = nn.Sequential(
            SelfAtt_(self.embedding_dim + self.x_increased_dim, dropout),
            nn.Linear(self.embedding_dim + self.x_increased_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )

        # WGAT layers with skip connections
        self.layers = nn.ModuleList(
            [
                WGATLayer_chunked(
                    dim_in if i else embedding_dim,
                    dim_out,
                    n_heads,
                    dropout,
                    chunk_size,
                )
                for i, (dim_in, dim_out) in enumerate(zip([embedding_dim] + fusion_dims[:-1], fusion_dims))
            ]
        )

        self.norm = nn.LayerNorm(fusion_dims[-1])

        # Skip connections
        self.skips = nn.ModuleList([nn.Linear(dim, fusion_dims[-1]) for dim in fusion_dims]) if len(fusion_dims) > 1 else None

    def forward(self, node_names, x, edge_attr, edge_index):
        if isinstance(node_names[0], list):
            node_names = [n for sublist in node_names for n in sublist]

        # Initial embeddings
        ids = torch.tensor(
            [self.dict_node_names[n] for n in node_names],
            dtype=torch.long,
            device=x.device,
        )

        # Get embeddings for all nodes
        E_all = self.embed.weight
        E_i = E_all[ids]

        # Get embeddings for current nodes
        # E_i = self.embed(ids)

        # But use repeating method
        x_increased = x.repeat(1, self.x_increased_dim)

        emb = self.quant_emb(torch.cat([x_increased, E_i], dim=-1))
        emb = emb + E_i

        # Multi-scale processing
        skips = []
        if self.skips:
            for i, layer in enumerate(self.layers):
                emb = layer(emb, edge_index, edge_attr)
                if i < len(self.skips):
                    skips.append(self.skips[i](emb))

            # Skip fusion
            emb = emb + torch.stack(skips).mean(dim=0)

            emb = self.norm(emb)

        return emb, E_all, ids


class WGATLayer_chunked(MessagePassing):
    r"""
    (NMIC) Weighted Graph Attention Layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_heads: int,
        dropout: float = const.default.dropout,
        chunk_size: int = const.default.chunk_size,
    ):
        r"""
        Initialize the Weighted Graph Attention Layer.

        Args:
            input_dim: The dimension of the input embeddings.
            output_dim: The dimension of the output embeddings.
            num_heads: The number of attention heads.
            dropout: The dropout probability for the attention weights.
            chunk_size: The chunk size for processing large tensors.
        """
        super().__init__(aggr="add")
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.chunk_size = chunk_size

        # Split weight matrix into two parts to avoid concatenation
        self.W_i = nn.Linear(input_dim, output_dim * num_heads)
        self.W_j = nn.Linear(input_dim, output_dim * num_heads)

        self.trans = nn.Linear(input_dim, output_dim * num_heads)
        self.attn = nn.Parameter(torch.empty(num_heads, output_dim))
        self.reset_parameters()

        # Adaptive chunk size calculation
        self.adap_chunk_size = GetAdaptiveChunkSize()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_i.weight, gain=nn.init.calculate_gain("leaky_relu", 0.2))
        nn.init.xavier_uniform_(self.W_j.weight, gain=nn.init.calculate_gain("leaky_relu", 0.2))
        nn.init.normal_(self.attn, mean=0, std=0.1)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        num_edges = x_i.size(0)
        if num_edges == 0:
            print("Empty edge index, returning zeros.")
            return torch.zeros_like(x_i)

        chunk_size = min(
            self.adap_chunk_size.calc(tensor_shape=(num_edges, self.num_heads * self.output_dim * 4 + self.num_heads), dim=0),
            const.default.chunk_size,
        )

        h_chunks = []
        for i in range(0, num_edges, chunk_size):
            idx = slice(i, min(i + chunk_size, num_edges))
            _chunk_size = idx.stop - idx.start
            # Split computation to avoid concatenation
            h_i = self.W_i(x_i[idx]).view(_chunk_size, self.num_heads, self.output_dim)
            h_j = self.W_j(x_j[idx]).view(_chunk_size, self.num_heads, self.output_dim)
            h = h_i + h_j

            # Calculate attention coefficients
            a = torch.einsum("bho,ho->bh", h, self.attn)

            if edge_attr is not None:
                a = a * edge_attr[idx].view(-1, 1)  # .expand(-1, self.num_heads)

            # Process attention scores
            a = F.softmax(a, dim=0)

            # Transform and weight features
            x_trans = self.trans(x_j[idx])
            x_trans = x_trans.view(_chunk_size, self.num_heads, self.output_dim)
            x_trans = torch.einsum("blh,bl->bh", x_trans, a)

            h_chunks.append(x_trans)

        h = torch.cat(h_chunks, dim=0)
        return h


class GE_Decoder(nn.Module):
    r"""
    Graph Embedding Decoder.
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = const.default.dropout,
        chunk_size: int = const.default.chunk_size,
        n_heads: int = const.default.n_heads_ge_decoder,
        n_res_blocks: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.n_heads = n_heads

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.SiLU(),
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_dim, 2 * h_dim),
                    nn.SiLU(),
                    nn.Linear(2 * h_dim, h_dim),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(n_res_blocks)
            ]
        )

        # Final projection
        self.ffn_q = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        # Adaptive chunk size calculation
        self.adap_chunk_size = GetAdaptiveChunkSize()

    def forward(self, z: torch.Tensor, E_all: torch.Tensor):
        """
        Args:
            z: Graph embeddings [batch_size, z_dim]
            E_all: All node embeddings [num_all_nodes, h_dim]

        Returns:
            h_s: Refined node embeddings [batch_size, num_all_nodes, h_dim]
            h_: Reconstructed features [batch_size, num_all_nodes, output_dim]
        """
        batch_size = z.size(0)
        num_all_nodes = E_all.size(0)

        chunk_size_fuse = self.adap_chunk_size.calc((num_all_nodes, batch_size, self.z_dim + self.h_dim))
        global_fused_chunks = []
        for i in range(0, num_all_nodes, chunk_size_fuse):
            end_idx = min(i + chunk_size_fuse, num_all_nodes)
            chunk_z = z.unsqueeze(1).expand(-1, end_idx - i, -1)
            chunk_E = E_all[i:end_idx].unsqueeze(0).expand(batch_size, -1, -1)
            global_fused_chunks.append(self.fusion(torch.cat([chunk_z, chunk_E], dim=-1)))
        global_fused = torch.cat(global_fused_chunks, dim=1)
        del global_fused_chunks, chunk_z, chunk_E

        head_dim = self.h_dim // self.n_heads
        chunk_size_attn = self.adap_chunk_size.calc(tensor_shape=(num_all_nodes, batch_size, self.h_dim, self.n_heads, head_dim), dim=1)
        h_chunks = []
        for j in range(0, num_all_nodes, chunk_size_attn):
            end_idx = min(j + chunk_size_attn, num_all_nodes)
            seq_chunk = global_fused[:, j:end_idx]

            # Cross-attention
            attn_chunk, _ = self.cross_attn(query=seq_chunk, key=seq_chunk, value=seq_chunk)

            # Apply residual blocks
            for block in self.res_blocks:
                attn_chunk = attn_chunk + block(attn_chunk)

            h_chunks.append(attn_chunk)

        # Combine all chunks
        attn_out = torch.cat(h_chunks, dim=1)

        # Final reconstruction
        h_ = self.ffn_q(attn_out)

        return attn_out, h_


# class GLabelPredictor(nn.Module):
#     r"""
#     A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
#     """

#     def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float, n_heads: int):
#         super().__init__()

#         _in_dim = input_dim
#         layers = []
#         for _dim in hidden_dims:
#             layers += [
#                 nn.Linear(_in_dim, _dim),
#                 nn.SiLU(),
#                 nn.BatchNorm1d(_dim),
#                 nn.Dropout(dropout),
#             ]
#             _in_dim = _dim
#         layers.append(nn.LayerNorm(_in_dim))
#         layers.append(nn.Linear(_in_dim, output_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim == 1:
#             x = x.unsqueeze(0)
#         elif x.ndim == 3:
#             x = x.squeeze(2)
#         return self.net(x)


class GLabelPredictor(nn.Module):
    r"""
    A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float, n_heads: int):
        super().__init__()

        self.blocks = nn.ModuleList()
        in_dim = input_dim
        residual_projs = []  # Store residual projections separately

        for i, out_dim in enumerate(hidden_dims):
            # Create linear layer for residual connection if dimensions don't match
            proj = nn.Identity()
            if in_dim != out_dim and i > 0:
                proj = nn.Linear(in_dim, out_dim)
            residual_projs.append(proj)

            # Create main block components
            attn_layer = MultiHeadSelfAttention(out_dim, n_heads)
            layer = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.BatchNorm1d(out_dim), nn.Dropout(dropout), attn_layer)
            self.blocks.append(layer)
            in_dim = out_dim

        # Save residual projections as parameters
        self.residual_projs = nn.ModuleList(residual_projs)
        self.layer_norm = nn.LayerNorm(in_dim)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.squeeze(2)

        for i, layer in enumerate(self.blocks):
            res = x
            x = layer(x)
            # Handle dimension mismatch through projection
            if i > 0:
                proj = self.residual_projs[i]
                res = proj(res)  # Project residue if dimensions don't match
                # Only add residual connection when projected shape matches
                if res.shape == x.shape:
                    x = x + res

        x = self.layer_norm(x)
        return self.output_layer(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x_3d = x.unsqueeze(0)
            attn_out, _ = self.attn(x_3d, x_3d, x_3d)
            return self.norm(x + attn_out.squeeze(0))
        else:
            attn_out, _ = self.attn(x, x, x)
            return self.norm(x + attn_out)


class SelfAtt_(nn.Module):
    r"""
    Self-attention (pooling) layer.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = const.default.dropout,
        pool: bool = False,
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.dropout else 0.0, scale=self.scale)
        if self.pool:
            x = x.mean(dim=0)
        x = self.proj(x)
        return x

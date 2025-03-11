r"""
Modules for DeepTAN.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data as GData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_undirected

import deeptan.constants as const


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
            # nn.GELU(),
        )

        # WGAT layers with skip connections
        self.layers = nn.ModuleList(
            [
                WGATLayer(
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

        return emb, E_i, E_all


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

    def _init_pooling_layers(self, input_dim, output_dim, heads):
        # Local subgraph pooling
        self.xgat_pool = WGATLayer(input_dim, output_dim, heads, self.dropout, self.chunk_size)
        self.att_pool = SelfAtt_(output_dim, self.dropout, True)

        # Global graph pooling
        self.global_xgat_pool = WGATLayer(output_dim, output_dim, heads, self.dropout, self.chunk_size)
        self.global_att_pool = SelfAtt_(output_dim, self.dropout, True)

    def _forward_embedding(self, node_names, x, edge_attr, edge_index):
        return self.node_embedding_layers(node_names, x, edge_attr, edge_index)

    @torch._dynamo.disable
    def forward(self, node_names, x, edge_attr, edge_index, batch):
        # Node embedding with layer norm
        h, E_i, E_all = self._forward_embedding(node_names, x, edge_attr, edge_index)

        # Graph embedding
        unique_batches = torch.unique(batch)
        graph_embs = []

        for graph_id in unique_batches:
            # Extract node mask for the current graph
            mask = batch == graph_id
            node_indices = torch.where(mask)[0]

            # Skip empty graphs
            if node_indices.numel() == 0:
                print("Warning: Empty graph detected")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=x.device))
                continue

            # Extract edges for the current graph
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask]

            # Adjust edge indices to local indices
            local_node_ids = torch.arange(mask.sum(), device=x.device)
            global_to_local = torch.zeros_like(mask, dtype=torch.long, device=x.device)
            global_to_local[node_indices] = local_node_ids

            # Apply global-to-local mapping
            sub_edge_index = global_to_local[sub_edge_index]

            # Filter out invalid indices
            valid_mask = (sub_edge_index[0] >= 0) & (sub_edge_index[1] >= 0) & (sub_edge_index[0] < mask.sum()) & (sub_edge_index[1] < mask.sum())
            sub_edge_index = sub_edge_index[:, valid_mask]

            # Validate subgraph edge indices
            if sub_edge_index.numel() == 0:
                print("Warning: Empty subgraph detected")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=x.device))
                continue

            # Compute dynamic centrality
            h_mask = h[mask]
            filtered_edge_index, centrality = self._calculate_dynamic_centrality(
                h_mask,
                edge_attr[edge_mask] if edge_attr is not None else None,
                sub_edge_index,
            )

            # Generate multiscale subgraphs
            subgraphs = self._generate_multiscale_subgraphs(filtered_edge_index, centrality, h_mask)

            # Create graph embeddings
            if subgraphs:
                g_emb = self._create_graph_embeddings(subgraphs)
                graph_embs.append(g_emb)
            else:
                # Process empty subgraph case
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=x.device))

        # Stack all graph embeddings
        return torch.stack(graph_embs), E_i, E_all

    def _calculate_dynamic_centrality(self, h, edge_attr, edge_index):
        with torch.no_grad():
            row, col = edge_index
            num_edges = edge_index.size(1)
            sim_ = []

            # Calculate similarity for each edge in batches to reduce peak memory usage.
            for i in range(0, num_edges, self.chunk_size):
                idx = slice(i, min(i + self.chunk_size, num_edges))
                h_i = h[row[idx]]
                h_j = h[col[idx]]

                sim_.append((h_i * h_j).sum(dim=1).abs())

            sim_ = torch.cat(sim_, dim=0)

            min_sim_ = sim_.min()
            max_sim_ = sim_.max()
            sim_ = (sim_ - min_sim_) / (max_sim_ - min_sim_) if max_sim_ - min_sim_ != 0 else sim_
            sim_ = sim_.clamp(0, 1)

            # Combine with edge attributes
            edge_weight = sim_ * edge_attr.view(-1) if edge_attr is not None else sim_

            # Filter edges
            mask = edge_weight > self.thre_edge_exist

        filtered_edge = edge_index[:, mask]
        filtered_weight = edge_weight[mask]

        # Compute node centrality
        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.scatter_add_(0, filtered_edge[0], filtered_weight)
        centrality.scatter_add_(0, filtered_edge[1], filtered_weight)

        return filtered_edge, centrality

    @torch._dynamo.disable
    def _generate_multiscale_subgraphs(self, edge_index, centrality, h):
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
        _, edges = self.hist_fd(centrality)
        num_bins = len(edges) - 1
        covered_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # Use tensor-based storage for subgraphs
        subgraph_masks = torch.zeros((0, num_nodes), dtype=torch.bool, device=device)
        subgraph_centers = torch.zeros(0, dtype=torch.long, device=device)

        for bin_idx in reversed(range(num_bins)):
            bin_mask = (centrality >= edges[bin_idx]) & (centrality <= edges[bin_idx + 1])
            current_nodes = torch.where(bin_mask)[0]

            if current_nodes.numel() == 0:
                continue

            # Batch process nodes in chunks
            for nodes in current_nodes.split(min(self.chunk_size, current_nodes.numel(), const.default.subg_chunk_size)):
                subsets, subg_edge_indices = self._batch_k_hop_subgraph(nodes, self.n_hop, edge_index, num_nodes)

                for subset, subg_edge_idx, center_node in zip(subsets, subg_edge_indices, nodes.tolist()):
                    if subset.numel() < 2 or subg_edge_idx.shape[1] == 0:
                        continue

                    # Create new subgraph mask
                    new_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                    new_mask[subset] = True

                    # Vectorized overlap check
                    if subgraph_masks.shape[0] > 0:
                        intersections = (subgraph_masks & new_mask).sum(dim=1)
                        min_sizes = torch.minimum(subgraph_masks.sum(dim=1), new_mask.sum())
                        overlaps = intersections / (min_sizes + 1e-8)
                        overlapping_chunks = overlaps > self.thre_sg_overlap

                        del intersections, min_sizes, overlaps
                    else:
                        overlapping_chunks = torch.zeros(0, dtype=torch.bool, device=device)

                    # Merge overlapping subgraphs
                    if overlapping_chunks.any():
                        merged_mask = subgraph_masks[overlapping_chunks].any(dim=0) | new_mask
                        subgraph_masks = torch.cat([subgraph_masks[~overlapping_chunks], merged_mask.unsqueeze(0)], dim=0)
                        subgraph_centers = torch.cat([subgraph_centers[~overlapping_chunks], torch.tensor([center_node], device=device)])
                    else:
                        # Add new subgraph
                        subgraph_masks = torch.cat([subgraph_masks, new_mask.unsqueeze(0)], dim=0)
                        subgraph_centers = torch.cat([subgraph_centers, torch.tensor([center_node], device=device)])

                    # Update coverage tracking
                    covered_nodes |= new_mask

                    del new_mask

                    if covered_nodes.all():
                        break

        # print(f"Percent of nodes covered: {covered_nodes.float().mean():.2%} ({covered_nodes.sum()}/{num_nodes})")

        # Convert tensor masks back to GData objects
        subgraphs = []
        for mask, center in zip(subgraph_masks, subgraph_centers):
            node_idx = torch.where(mask)[0]
            num_sub_nodes = mask.sum().item()

            # Validate edge indices
            local_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device)
            local_mapping[node_idx] = torch.arange(num_sub_nodes, device=device)

            # Get edges within this subgraph
            sub_edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            sub_edge_global = edge_index[:, sub_edge_mask]

            # Map global edge indices to local indices
            sub_edge_local = local_mapping[sub_edge_global]

            # Create subgraph data object
            subgraphs.append(GData(x=h[node_idx], edge_index=sub_edge_local, center_node=center, node_idx=node_idx, mask=mask))

        # Free up memory
        del subgraph_masks, subgraph_centers, covered_nodes

        subgraphs = sorted(subgraphs, key=lambda x: -x.num_nodes)

        # print(f"\nNumber of subgraphs created: {len(subgraphs)}")
        # print(f"Quantiles of subgraph sizes:\n    {torch.quantile(torch.tensor([s.num_nodes for s in subgraphs], dtype=torch.float32), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])).int()}")

        return subgraphs

    @staticmethod
    @torch._dynamo.disable
    def _batch_k_hop_subgraph(nodes, num_hops, edge_index, num_nodes):
        """Custom implementation of batch k-hop subgraph extraction"""

        subsets = []
        edge_indices = []

        for node in nodes:
            subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx=node.item(), num_hops=num_hops, edge_index=edge_index, num_nodes=num_nodes, relabel_nodes=True)
            subsets.append(subset)
            edge_indices.append(sub_edge_index)
        return subsets, edge_indices

    @staticmethod
    @torch._dynamo.disable
    def hist_fd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freedman-Diaconis histogram calculation."""
        q75, q25 = torch.quantile(tensor, torch.tensor([0.75, 0.25], device=tensor.device))
        iqr = q75 - q25
        bin_width = 2 * iqr * (tensor.numel() ** (-1 / 3))
        num_bins = max(1, int((tensor.max() - tensor.min()) / (bin_width + 1e-8)))
        counts = torch.histc(tensor, bins=num_bins)
        edges = torch.linspace(tensor.min(), tensor.max(), num_bins + 1, device=tensor.device)
        return counts, edges

    def _create_graph_embeddings(self, subgraphs):
        chunk_size = self.chunk_size
        all_embs = []
        for i in range(0, len(subgraphs), chunk_size):
            subgraph_chunk = subgraphs[i : i + chunk_size]
            chunk_embs = [self._process_subgraph(g) for g in subgraph_chunk]
            chunk_embs = torch.cat(chunk_embs, dim=0) if chunk_embs else torch.empty(0, self.output_dim_g_emb, device=self.x.device)
            all_embs.append(chunk_embs)
        embs = torch.cat(all_embs, dim=0) if all_embs else torch.zeros(0, self.output_dim_g_emb, device=self.x.device)

        # Build super graph
        num_subgraphs = len(subgraphs)
        if num_subgraphs == 0:
            return torch.zeros(self.output_dim_g_emb, device=self.x.device)

        edge_index = to_undirected(torch.combinations(torch.arange(num_subgraphs, device=embs.device)).t())
        super_nodes = GData(x=embs, edge_index=edge_index)

        # Global pooling with chunked edge processing
        global_emb = self.global_xgat_pool(super_nodes.x, super_nodes.edge_index)
        return self.global_att_pool(global_emb)

    def _process_subgraph(self, subgraph):
        h = self.xgat_pool(subgraph.x, subgraph.edge_index)
        return self.att_pool(h.unsqueeze(0))


class WGATLayer(MessagePassing):
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

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_i.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.W_j.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.normal_(self.attn, mean=0, std=0.1)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        num_edges = x_i.size(0)
        h_chunks = []

        # Process in chunks to reduce peak memory
        for i in range(0, num_edges, self.chunk_size):
            idx = slice(i, min(i + self.chunk_size, num_edges))
            _chunk_size = idx.stop - idx.start
            # Split computation to avoid concatenation
            h_i = self.W_i(x_i[idx]).view(_chunk_size, self.num_heads, self.output_dim)
            h_j = self.W_j(x_j[idx]).view(_chunk_size, self.num_heads, self.output_dim)
            h = h_i + h_j

            # Calculate attention coefficients
            a = torch.einsum("ehd,hd->eh", h, self.attn)

            if edge_attr is not None:
                a = a * edge_attr[idx].view(-1, 1).expand(-1, self.num_heads)

            # Process attention scores
            a = F.softmax(a, dim=0)

            # Transform and weight features
            x_trans = self.trans(x_j[idx]).view(_chunk_size, self.num_heads, self.output_dim)
            x_trans = torch.einsum("ehd,eh->ed", x_trans, a)
            h_chunks.append(x_trans)

        h = torch.cat(h_chunks, dim=0)  # if h_chunks else torch.tensor([], device=x_i.device)
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

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=h_dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.GELU(),
            nn.LayerNorm(h_dim),
            nn.Dropout(dropout),
        )

        # Residual blocks
        self.res_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(h_dim, 2 * h_dim),
                    nn.GELU(),
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
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor, E_i: torch.Tensor, E_all: torch.Tensor):
        """
        Args:
            z: Graph embeddings [batch_size, z_dim]
            E_i: Initial node embeddings [batch_size, num_nodes, h_dim]
            E_all: All node embeddings [num_all_nodes, h_dim]

        Returns:
            h_s: Refined node embeddings [batch_size, num_all_nodes, h_dim]
            h_: Reconstructed features [batch_size, num_all_nodes, output_dim]
        """
        batch_size = z.size(0)
        num_all_nodes = E_all.size(0)

        h_chunks = []

        # Process all data in chunks
        for i in range(0, num_all_nodes, self.chunk_size):
            end_idx = min(i + self.chunk_size, num_all_nodes)
            current_chunk = E_all[i:end_idx]

            # Expand z and current chunk to batch dimension
            z_chunk = z.unsqueeze(1).expand(-1, end_idx - i, -1)
            E_chunk = current_chunk.unsqueeze(0).expand(batch_size, -1, -1)

            # Feature fusion
            fused_chunk = self.fusion(torch.cat([z_chunk, E_chunk], dim=-1))

            # Cross-attention
            attn_chunk, _ = self.cross_attn(query=fused_chunk, key=fused_chunk, value=fused_chunk)

            # Apply residual blocks
            for block in self.res_blocks:
                attn_chunk = attn_chunk + block(attn_chunk)

            h_chunks.append(attn_chunk)

        # Combine all chunks
        attn_out = torch.cat(h_chunks, dim=1)

        # Final reconstruction
        h_ = self.ffn_q(attn_out)

        return attn_out, h_


class GLabelPredictor(nn.Module):
    r"""
    A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float, n_heads: int):
        super().__init__()

        layers = [
            SelfAtt_(input_dim, dropout),
            nn.LayerNorm(input_dim),
        ]
        for dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, dim),
                nn.GELU(),
            ]
            input_dim = dim
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.GELU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class SelfAtt_(nn.Module):
    r"""
    Self-attention (pooling) layer.
    """

    def __init__(self, dim: int, dropout: float = const.default.dropout, pool: bool = False):
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

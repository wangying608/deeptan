r"""
Modules for DeepTAN.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch_geometric.data import Data as GData
from torch_geometric.nn import GATv2Conv, SAGPooling
from torch_geometric.utils import k_hop_subgraph, subgraph, to_undirected
from torch.utils.checkpoint import checkpoint

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
        """
        super().__init__()
        self.dict_node_names = dict_node_names
        self.output_dim_g_emb = output_dim_g_emb
        self.n_hop = n_hop
        self.thre_edge_exist = threshold_edge_exist
        self.thre_sg_overlap = threshold_subgraph_overlap
        self.dropout = dropout
        self.node_emb_dim = node_emb_dim
        self.n_heads_node_emb = n_heads_node_emb
        self.n_heads_pooling = n_heads_pooling

        # Node embedding
        self.node_embedding_layers = NodeEmbedding(
            node_emb_dim,
            fusion_dims_node_emb,
            dict_node_names,
            n_heads_node_emb,
            dropout,
        )

        # Multi-scale pooling architecture
        self._init_pooling_layers(fusion_dims_node_emb[-1], output_dim_g_emb, n_heads_pooling)

        # Adaptive chunk size calculation
        # self.adap_chunk_size_dyn_cent = GetAdaptiveChunkSize()
        self.adap_chunk_size_mul_subg = GetAdaptiveChunkSize()

    def _init_pooling_layers(self, input_dim, output_dim, heads):
        self.sagpool1 = SAGPooling(input_dim, ratio=1)
        self.global_xgat_pool1 = GATv2Conv(input_dim, output_dim, heads=heads, concat=False)
        self.global_sagpool1 = SAGPooling(output_dim, ratio=1)

    def forward(self, node_names, x, edge_index, batch):
        # Graph embedding
        unique_batches, inverse, counts = torch.unique(batch, return_inverse=True, return_counts=True)
        # Split nodes into subgraphs based on batch indices
        node_indices_list = torch.split(torch.arange(batch.size(0), device=batch.device), counts.tolist())

        # Process each graph in the batch separately
        all_h = []
        all_ids = []

        # Split node_names according to batch
        if isinstance(node_names[0], list):
            node_names_flat = [n for sublist in node_names for n in sublist]
        else:
            node_names_flat = node_names

        node_names_splits = []
        start_idx = 0
        for count in counts.tolist():
            node_names_splits.append(node_names_flat[start_idx : start_idx + count])
            start_idx += count

        for i, (node_indices, graph_node_names) in enumerate(zip(node_indices_list, node_names_splits)):
            if node_indices.numel() == 0:
                logger.warning("Empty batch detected!")
                continue

            # Extract subgraph for this batch
            sub_x = x[node_indices]
            sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)

            if sub_edge_index.numel() == 0:
                logger.warning("Empty subgraph detected!")
                continue

            # Process this subgraph with NodeEmbedding using gradient checkpointing
            def node_embedding_forward(names, x_data, edge_idx):
                return self.node_embedding_layers(names, x_data, edge_idx)

            h_sub, ids_sub = checkpoint(
                node_embedding_forward, graph_node_names, sub_x, sub_edge_index, use_reentrant=False
            )
            all_h.append(h_sub)
            all_ids.append(ids_sub)

        # Concatenate all processed subgraphs
        if all_h.__len__() == 0:
            raise ValueError("No valid subgraphs found!")

        h = torch.cat(all_h, dim=0)
        ids = torch.cat(all_ids, dim=0)

        #
        graph_embs = []
        self._device = x.device

        for node_indices in node_indices_list:
            if node_indices.numel() == 0:
                logger.warning("Empty batch detected!")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)

            # Validate subgraph edge indices
            if sub_edge_index.numel() == 0:
                logger.warning("Empty subgraph detected!")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            # Compute dynamic centrality
            h_masked = h[node_indices]
            filtered_edge_index, centrality = self._calculate_dynamic_centrality(h_masked, sub_edge_index)

            # Generate multiscale subgraphs
            subgraphs = self._generate_multiscale_subgraphs(filtered_edge_index, centrality, h_masked)

            # Create graph embeddings
            if subgraphs:
                g_emb = self._create_graph_embedding(subgraphs)
                graph_embs.append(g_emb)
            else:
                # Process empty subgraph case
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))

        # Stack all graph embeddings
        return torch.stack(graph_embs), ids

    def _calculate_dynamic_centrality(self, h, edge_index):
        row, col = edge_index
        h_row = h[row]
        h_col = h[col]

        num_edges = h_row.shape[0]
        num_nodes = h.shape[0]

        # Calculate graph density for adaptive threshold
        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        graph_density = num_edges / max(max_possible_edges, 1)

        # Adaptive threshold based on graph density
        # Denser graphs need higher thresholds to avoid over-connectivity
        adaptive_threshold = self.thre_edge_exist * (1 + 2 * graph_density)

        if num_edges <= 1000:
            sim_ = (h_row * h_col).sum(dim=1).abs()
        else:
            chunk_size = self.adap_chunk_size_mul_subg.calc(
                tensor_shape=(num_edges, edge_index.shape[1], h.shape[1]),
                dim=0,
                dtype=h.dtype,
            )
            chunk_size = max(16, chunk_size)

            sim_chunks = []
            for i in range(0, num_edges, chunk_size):
                end = min(i + chunk_size, num_edges)
                h_row_chunk = h_row[i:end]
                h_col_chunk = h_col[i:end]
                sim_chunk = (h_row_chunk * h_col_chunk).sum(dim=1).abs()
                sim_chunks.append(sim_chunk)

            sim_ = torch.cat(sim_chunks, dim=0)

        min_sim_ = sim_.min()
        max_sim_ = sim_.max()
        if max_sim_ - min_sim_ > 1e-6:
            sim_.sub_(min_sim_).div_(max_sim_ - min_sim_)
        else:
            sim_.clamp_(min=0, max=1)

        # Use adaptive threshold instead of fixed threshold
        mask = sim_ > adaptive_threshold
        filtered_edge = edge_index[:, mask]
        filtered_weight = sim_[mask]

        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.scatter_add_(0, filtered_edge[0], filtered_weight)
        centrality.scatter_add_(0, filtered_edge[1], filtered_weight)
        return filtered_edge, centrality

    def _generate_multiscale_subgraphs(
        self,
        edge_index: torch.Tensor,
        centrality: torch.Tensor,
        h: torch.Tensor,
    ) -> List[GData]:
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

            chunk_size = self.adap_chunk_size_mul_subg.calc(
                tensor_shape=(n_current, edge_index.shape[1], h.shape[1]),
                dim=0,
                dtype=h.dtype,
            )

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
                        min_sizes = torch.min(
                            existing_masks.sum(dim=1), torch.full_like(intersections, new_mask.sum().item())
                        )
                        overlaps = intersections / (min_sizes + 1e-6)
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
    def _batch_k_hop_subgraph(
        nodes: torch.Tensor, num_hops: int, edge_index: torch.Tensor, num_nodes: int
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Custom implementation of batch k-hop subgraph extraction."""
        nodes_cpu = nodes.cpu().numpy().tolist()
        subsets = []
        edge_indices = []

        for _node in nodes_cpu:
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                node_idx=_node, num_hops=num_hops, edge_index=edge_index, num_nodes=num_nodes, relabel_nodes=True
            )
            subsets.append(subset)
            edge_indices.append(sub_edge_index)
        return subsets, edge_indices

    @staticmethod
    def hist_fd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Freedman-Diaconis rule with memory optimization."""

        # Calculate statistics on GPU first
        min_val = tensor.min()
        max_val = tensor.max()
        q75, q25 = torch.quantile(tensor, torch.tensor([0.75, 0.25], device=tensor.device))
        iqr = q75 - q25

        # Move scalar values to CPU for bin calculation
        min_cpu = min_val.cpu().item()
        max_cpu = max_val.cpu().item()
        iqr_cpu = iqr.cpu().item()

        # Calculate bin parameters on CPU
        bin_width = 2 * iqr_cpu / (tensor.numel() ** (1 / 3)) if iqr_cpu > 0 else (max_cpu - min_cpu)
        num_bins = int((max_cpu - min_cpu) / bin_width) if bin_width > 0 else 1
        num_bins = max(1, min(num_bins, 1000))  # Cap bins to 1000

        # Create edges on CPU first
        edges = torch.linspace(min_cpu, max_cpu, num_bins + 1, device=tensor.device)

        # Calculate histogram on GPU
        hist = torch.histc(tensor.cpu(), bins=num_bins, min=min_cpu, max=max_cpu)
        return hist.to(tensor.device), edges

    def _create_graph_embedding(self, subgraphs):
        num_subgraphs = len(subgraphs)
        if num_subgraphs == 0:
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        # Filter out empty subgraphs
        subgraphs = [g for g in subgraphs if g.x.numel() > 0]
        num_subgraphs = len(subgraphs)
        if num_subgraphs == 0:
            logger.warning("All subgraphs were empty, returning zero embeddings.")
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        embs = torch.cat([self._process_subgraph(g) for g in subgraphs])

        # Build super graph and perform global pooling
        super_nodes_edge_index = to_undirected(torch.combinations(torch.arange(num_subgraphs, device=embs.device)).t())
        global_emb = self.global_xgat_pool1(embs, super_nodes_edge_index)
        _outputs = self.global_sagpool1(global_emb, super_nodes_edge_index)
        global_emb = _outputs[0]

        return global_emb.squeeze(0)

    def _process_subgraph(self, subgraph):
        hx = self.sagpool1(subgraph.x, subgraph.edge_index)
        return hx[0]


class NodeEmbedding(nn.Module):
    r"""
    Embedding nodes in a graph like embedding words in a sentence.
    """

    def __init__(
        self,
        embedding_dim: int,
        fusion_dims: List[int],
        dict_node_names: Dict[str, int],
        n_heads: int,
        dropout: float = const.default.dropout,
    ):
        r"""
        Embedding nodes in a graph like embedding words in a sentence.

        Args:
            embedding_dim: Dimension of the embedding.
            fusion_dims: Dimensions for the fusion (fusing observation value and inherent feature) layers.
            dict_node_names: Dictionary mapping node names to indices.
            n_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.fusion_dims = fusion_dims
        self.dict_node_names = dict_node_names
        self.n_heads = n_heads

        self.embed = nn.Embedding(len(dict_node_names), embedding_dim, scale_grad_by_freq=False, sparse=True)

        # Embedding features
        self.feature_proj = nn.Linear(1, embedding_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(embedding_dim)

        self._layers = nn.ModuleList(
            [
                GATv2Conv(
                    in_channels=embedding_dim if i == 0 else fusion_dims[i - 1],
                    out_channels=dim_out,
                    heads=n_heads,
                    concat=False,
                )
                for i, dim_out in enumerate(fusion_dims)
            ]
        )

        self.norm = nn.LayerNorm(fusion_dims[-1])

    def forward(
        self,
        node_names,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(node_names[0], list):
            node_names = [n for sublist in node_names for n in sublist]

        # Initial embeddings
        ids = torch.tensor(
            [self.dict_node_names[n] for n in node_names],
            dtype=torch.long,
            device=x.device,
        )

        # Get node embeddings
        node_embeddings = self.embed(ids)

        # Concatenate and fuse features
        x_proj = self.feature_proj(x.unsqueeze(-1))
        node_emb = node_embeddings.unsqueeze(1)
        # Cross attention
        fused, _ = self.cross_attn(x_proj, node_emb, node_emb)
        fused = self.fusion_norm(fused.squeeze(1) + node_embeddings)

        # Multi-scale processing
        for _layer in self._layers:
            fused = _layer(fused, edge_index)
        fused = self.norm(fused)

        return fused, ids


class GE_Decoder(nn.Module):
    r"""
    Efficient Graph Embedding Decoder without attention mechanisms.
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = const.default.dropout,
        n_layers: int = 4,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.output_dim = output_dim

        self.fusion = nn.Sequential(
            nn.Linear(z_dim + h_dim, h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.feature_processor = self._build_mlp(h_dim, hidden_dim, n_layers, dropout)

        self.final_projection = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, output_dim),
        )

        self.adap_chunk_size = GetAdaptiveChunkSize(min_chunk_size=16)

    def forward(self, z: torch.Tensor, E_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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

        # chunk_size = self.adap_chunk_size.calc(
        #     tensor_shape=(num_all_nodes, batch_size, self.z_dim + self.h_dim, self.h_dim),
        #     dim=0,
        #     dtype=z.dtype,
        # )
        # logger.info(f"Using chunk size: {chunk_size}")
        chunk_size = 16

        refined_features = torch.zeros(batch_size, num_all_nodes, self.h_dim, device=z.device)
        reconstructed_features = torch.zeros(batch_size, num_all_nodes, self.output_dim, device=z.device)

        for i in range(0, num_all_nodes, chunk_size):
            end_idx = min(i + chunk_size, num_all_nodes)

            current_chunk = E_all[i:end_idx]
            chunk_size_actual = end_idx - i

            z_expanded = z.unsqueeze(1).expand(-1, chunk_size_actual, -1)
            current_chunk_expanded = current_chunk.unsqueeze(0).expand(batch_size, -1, -1)

            fused = torch.cat([z_expanded, current_chunk_expanded], dim=-1)
            fused = self.fusion(fused)

            processed = self.feature_processor(fused)

            refined_features[:, i:end_idx] = processed

            reconstructed = self.final_projection(processed)
            reconstructed_features[:, i:end_idx] = reconstructed

        return refined_features, reconstructed_features

    def _build_mlp(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float) -> nn.Module:
        layers = []
        for i in range(n_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
            if i < n_layers - 1:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, input_dim))
        return nn.Sequential(*layers)


class GLabelPredictor(nn.Module):
    r"""
    Graph-level label predictor.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers = []
        in_dim = input_dim

        for out_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, out_dim),
                nn.SiLU(),
                nn.LayerNorm(out_dim),  # Use LayerNorm instead of BatchNorm1d
                nn.Dropout(dropout),
            ]
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(in_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle input shape variations
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.squeeze(2)

        x = self.mlp(x)
        return self.output_layer(x)


class FocalLoss(torch.nn.Module):
    r"""Multi-class Focal Loss
    Formula: loss = -alpha * (1-p)^gamma * log(p)
    """

    def __init__(
        self,
        gamma: float = 0.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=None,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_weight = alpha[targets]
            loss = alpha_weight * loss

        return loss.mean() if self.reduction == "mean" else loss

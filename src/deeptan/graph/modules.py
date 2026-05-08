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
import torch.nn.functional as F 

import deeptan.constants as const
from deeptan.utils.uni import GetAdaptiveChunkSize
from torch_scatter import scatter_add


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
            # logger.debug(f"Batch {i}: {len(node_indices)} nodes")
            assert (node_indices >= 0).all()
            assert node_indices.max().item() < x.size(0)
            if node_indices.numel() == 0:
                # logger.warning("Empty batch detected!")
                continue
            # Extract subgraph for this batch
            sub_x = x[node_indices]
            sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True, num_nodes=x.size(0))

            if sub_edge_index.numel() == 0:
                # logger.warning("Empty subgraph detected! Using fallback node embedding.")
                ids_sub = torch.tensor(
                    [self.node_embedding_layers.dict_node_names[n] for n in graph_node_names],
                    device=x.device,
                    dtype=torch.long,
                )
                node_embeddings = self.node_embedding_layers.embed(ids_sub)
                x_proj = self.node_embedding_layers.feature_proj(sub_x.unsqueeze(-1)).squeeze(1)
                fused = torch.cat([node_embeddings, x_proj], dim=-1)
                fused = self.node_embedding_layers.fusion_mlp(fused)
                h_sub = self.node_embedding_layers.norm(fused)

                all_h.append(h_sub)
                all_ids.append(ids_sub)
                continue

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
        
        graph_embs = []
        self._device = x.device

        for node_indices in node_indices_list:
            if node_indices.numel() == 0:
                # logger.warning("Empty batch detected!")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True, num_nodes=x.size(0))

            # Validate subgraph edge indices
            if sub_edge_index.numel() == 0:
                # logger.warning("Empty subgraph detected!")
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            # Compute dynamic centrality
            h_masked = h[node_indices] 
            filtered_edge_index, centrality = self._calculate_dynamic_centrality(h_masked, sub_edge_index)

            # Generate multiscale subgraphs
            subgraphs = self._generate_multiscale_subgraphs(filtered_edge_index, centrality, h_masked)
            
            if len(subgraphs) == 0:
                logger.warning(f"No subgraphs generated for graph with nodes and {edge_index.size(1)} edges.")

            # Create graph embeddings
            if subgraphs:
                g_emb = self._create_graph_embedding(subgraphs)
                graph_embs.append(g_emb)
            else:
                # Process empty subgraph case
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
        result = torch.stack(graph_embs)
        # Stack all graph embeddings
        return result, ids

    @torch.no_grad()
    def _calculate_dynamic_centrality(self, h, edge_index):
        row, col = edge_index 
        num_edges = row.size(0)
        num_nodes = h.size(0)

        if num_edges == 0:
            device = h.device
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            centrality = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            return filtered_edge_index, centrality

        max_possible_edges = num_nodes * (num_nodes - 1) // 2
        graph_density = num_edges / max(max_possible_edges, 1)
        adaptive_threshold = self.thre_edge_exist * (1 + 2 * graph_density)

        chunk_size = self.adap_chunk_size_mul_subg.calc(
            tensor_shape=(num_edges, h.shape[1]),
            dim=0,
            dtype=h.dtype
        )
        chunk_size = max(16, min(chunk_size, 512))

        device = h.device
        filtered_rows = []
        filtered_cols = []
        filtered_weights = []

        for start in range(0, num_edges, chunk_size):
            end = min(start + chunk_size, num_edges)
            r_chunk = row[start:end]
            c_chunk = col[start:end]

            h_r = h[r_chunk]  # [C, D]
            h_c = h[c_chunk]  # [C, D]
            sim = torch.sum(h_r * h_c, dim=1).abs()  # [C]
            if adaptive_threshold > sim.max():
                adaptive_threshold = sim.quantile(0.5)  

            mask = sim > adaptive_threshold
            if mask.any():
                filtered_rows.append(r_chunk[mask])
                filtered_cols.append(c_chunk[mask])
                filtered_weights.append(sim[mask])

            del h_r, h_c, sim, mask

        # Concatenate only valid edges
        if not filtered_rows:
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            centrality = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            return filtered_edge_index, centrality

        try:
            filtered_row = torch.cat(filtered_rows, dim=0)
            filtered_col = torch.cat(filtered_cols, dim=0)
            filtered_weight = torch.cat(filtered_weights, dim=0)
            filtered_edge_index = torch.stack([filtered_row, filtered_col], dim=0)
        except RuntimeError as e:
            logger.warning(f"Failed to concatenate filtered edges: {e}")
            filtered_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            centrality = torch.zeros(num_nodes, dtype=h.dtype, device=device)
            return filtered_edge_index, centrality

        # Centrality accumulation
        centrality = torch.zeros(num_nodes, dtype=h.dtype, device=device)
        if filtered_weight.numel() > 0:
            # Scatter both directions
            scatter_add(filtered_weight, filtered_row, out=centrality, dim=0)
            scatter_add(filtered_weight, filtered_col, out=centrality, dim=0)

        return filtered_edge_index, centrality

    def _generate_multiscale_subgraphs(
        self,
        edge_index: torch.Tensor,
        centrality: torch.Tensor,
        h: torch.Tensor,
    ) -> List[GData]:
        device = h.device
        num_nodes = h.size(0)

        if num_nodes == 0:
            return []

        # Compute FD bins
        _, edges = self.hist_fd(centrality)
        num_bins = len(edges) - 1
        covered_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        subgraph_masks = []   # Store final masks only
        subgraph_centers = []

        adj_csr = None  # Will be built lazily

        for bin_idx in reversed(range(num_bins)):
            if covered_nodes.all():
                break

            bin_mask = (centrality >= edges[bin_idx]) & (centrality <= edges[bin_idx + 1])
            center_candidates = torch.where(bin_mask)[0]
            if center_candidates.numel() == 0:
                continue

            chunk_size = max(1, 512 // (self.n_hop + 1))
            for start in range(0, len(center_candidates), chunk_size):
                batch_centers = center_candidates[start:start+chunk_size]

                if adj_csr is None and edge_index.numel() > 0:
                    row, col = edge_index
                    if row.numel() > 0:
                        adj = torch.sparse_coo_tensor(
                            torch.stack([row, col]), torch.ones_like(row, dtype=h.dtype),
                            size=(num_nodes, num_nodes)
                        ).coalesce().to_sparse_csr()
                        adj_csr = adj

                if adj_csr is None:
                    continue  

                node_masks = self._batch_k_hop_subgraph(
                    batch_centers, self.n_hop, edge_index, num_nodes
                )  

                for i, center_node in enumerate(batch_centers.tolist()):
                    mask = node_masks[i]  
                    if mask.sum() < 2:
                        continue

                    # Overlap detection
                    overlapping_indices = []
                    if subgraph_masks:
                        existing_stacked = torch.stack(subgraph_masks)  
                        intersection = (existing_stacked & mask).sum(dim=1)
                        min_sizes = torch.min(existing_stacked.sum(dim=1), mask.sum())
                        overlaps = intersection / (min_sizes + 1e-8)
                        overlapping_indices = (overlaps > self.thre_sg_overlap).nonzero(as_tuple=False).view(-1).tolist()

                    # Collect masks to merge
                    merged_mask = mask.clone()
                    masks_to_merge = []
                    indices_to_remove = sorted(overlapping_indices, reverse=True)  

                    # Validate indices before accessing
                    valid_indices = [idx for idx in indices_to_remove if 0 <= idx < len(subgraph_masks)]
                    for idx in valid_indices:
                        masks_to_merge.append(subgraph_masks[idx])

                    # Remove invalid ones safely
                    for idx in reversed(range(len(subgraph_masks))):
                        if idx in valid_indices:
                            del subgraph_masks[idx]
                            del subgraph_centers[idx]

                    # Merge all overlapping regions into one
                    for m in masks_to_merge:
                        merged_mask |= m

                    subgraph_masks.append(merged_mask)
                    subgraph_centers.append(center_node)
                    covered_nodes |= merged_mask

                    if covered_nodes.all():
                        break
                if covered_nodes.all():
                    break
            if covered_nodes.all():
                break

        # Build GData objects
        subgraphs = []
        valid_pairs = [(m, c) for m, c in zip(subgraph_masks, subgraph_centers) if m.sum() > 1]

        for mask, center in valid_pairs:
            subset_nodes = torch.where(mask)[0]
            if subset_nodes.numel() == 0:
                continue

            try:
                sub_edge_index, _ = subgraph(
                    subset=subset_nodes,
                    edge_index=edge_index,
                    relabel_nodes=True,
                    num_nodes=num_nodes
                )
                sub_x = h[subset_nodes]
                gdata = GData(
                    x=sub_x,
                    edge_index=sub_edge_index,
                    center_node=center,
                    node_idx=subset_nodes,
                    mask=mask
                )
                subgraphs.append(gdata)
            except Exception as e:
                logger.warning(f"Failed to build subgraph around node {center}: {e}")
                continue

        # Sort by size descending
        subgraphs.sort(key=lambda g: g.num_nodes, reverse=True)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batched k-hop subgraph extraction using vectorized BFS via sparse matrix multiplication.
        Returns:
            - node_masks: [len(nodes), num_nodes], bool tensor indicating membership
            - edge_indices_list not returned directly; use later masking
        """
        device = edge_index.device
        row, col = edge_index

        # Build adjacency matrix: sparse (num_nodes, num_nodes)
        adj = torch.sparse_coo_tensor(
            indices=torch.stack([row, col]),
            values=torch.ones(row.size(0), device=device),
            size=(num_nodes, num_nodes)
        ).coalesce()

        # Convert to CSR for faster matmul
        adj_csr = adj.to_sparse_csr()

        # Initialize mask: one-hot for each seed node
        node_masks = torch.zeros((nodes.size(0), num_nodes), dtype=torch.bool, device=device)
        node_masks.scatter_(1, nodes.unsqueeze(-1), True)

        current_mask = node_masks.float()
        for _ in range(num_hops):
            # Sparse-Dense matmul: propagate neighbors
            next_mask_vals = torch.matmul(adj_csr, current_mask.t()).t().bool()
            new_neighbors = next_mask_vals & (~node_masks)
            if not new_neighbors.any():
                break
            node_masks |= next_mask_vals
            current_mask = new_neighbors.float()

        return node_masks      

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

    def _create_graph_embedding(self, subgraphs: List[GData]):
        if not subgraphs:
            logger.warning("❌ No subgraphs provided to create graph embedding.")
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        # Filter valid non-empty subgraphs
        valid_subgraphs = [g for g in subgraphs if hasattr(g, 'x') and g.x.numel() > 0]
        if not valid_subgraphs:
            logger.warning("❌ All subgraphs are empty after filtering.")
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        try:
            embs = torch.stack([self._process_subgraph(g)[0] for g in valid_subgraphs])  # [S, D]
        except Exception as e:
            logger.error(f"💥 Error processing subgraphs: {e}")
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        num_subgraphs = embs.size(0)
        if num_subgraphs == 1:
            # Can't form edge_index with single node → skip pooling
            global_emb = self.global_xgat_pool1(embs, torch.empty(2, 0, dtype=torch.long, device=embs.device))
            return global_emb.squeeze(0)

        # Create fully connected super-node graph
        super_edge_index = to_undirected(torch.combinations(torch.arange(num_subgraphs, device=embs.device), r=2).t())

        try:
            global_emb = self.global_xgat_pool1(embs, super_edge_index)
            pooled = self.global_sagpool1(global_emb, super_edge_index)
            return pooled[0].squeeze(0)
        except Exception as e:
            logger.error(f"Global pooling failed: {e}")
            return torch.mean(embs, dim=0)  # Fallback: mean pooling


    def _process_subgraph(self, subgraph):
        hx = self.sagpool1(subgraph.x, subgraph.edge_index)
        return hx[0]

class NodeEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        fusion_dims: List[int],
        dict_node_names: Dict[str, int],
        n_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.dict_node_names = dict_node_names

        self.embed = nn.Embedding(len(dict_node_names), embedding_dim)
        self.feature_proj = nn.Linear(1, embedding_dim)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

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
        self.float()

    def forward(self, node_names, x, edge_index):
        if isinstance(node_names[0], list):
            node_names = [n for sublist in node_names for n in sublist]

        ids = torch.tensor([self.dict_node_names[n] for n in node_names], device=x.device)
        node_embeddings = self.embed(ids)
        
        x = x.float()  # float32
        x_proj = self.feature_proj(x.unsqueeze(-1)).squeeze(1)

        # Simple splicing and fusion
        fused = torch.cat([node_embeddings, x_proj], dim=-1)
        fused = self.fusion_mlp(fused)

        for layer in self._layers:
            fused = layer(fused, edge_index)

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

        chunk_size = self.adap_chunk_size.calc(
            tensor_shape=(num_all_nodes, batch_size, self.z_dim + self.h_dim, self.h_dim),
            dim=0,
            dtype=z.dtype,
        )
        chunk_size = max(16, min(chunk_size, 128))
        # logger.info(f"Using chunk size: {chunk_size}")

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
                nn.LayerNorm(out_dim), 
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

r"""
Modules for DeepTAN.
"""

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import k_hop_subgraph, to_undirected
from torch_geometric.data import Data as GData
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
        self.fusion_dims = fusion_dims
        self.dict_node_names = dict_node_names
        self.n_heads = n_heads

        self.embed = nn.Embedding(
            len(dict_node_names), embedding_dim, scale_grad_by_freq=True, sparse=True
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
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
                for i, (dim_in, dim_out) in enumerate(
                    zip([embedding_dim] + fusion_dims[:-1], fusion_dims)
                )
            ]
        )

        # Skip connections
        self.skips = (
            nn.ModuleList([nn.Linear(dim, fusion_dims[-1]) for dim in fusion_dims])
            if len(fusion_dims) > 1
            else None
        )

    def forward(self, node_names, x, edge_attr, edge_index):
        if isinstance(node_names[0], list):
            node_names = [n for sublist in node_names for n in sublist]

        # Verify node indices in edge_index
        # num_nodes = x.size(0)
        # assert torch.all(edge_index >= 0) and torch.all(edge_index < num_nodes), (
        #     "Invalid edge indices detected"
        # )

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

        # x_increased = self.mlp1(x)
        # Do not use mlp1 for increasing input feature values
        # But use repeating method
        x_increased = x.repeat(1, self.embedding_dim)

        combined = torch.cat([x_increased, E_i], dim=-1)
        # combined = self.dropout(combined)
        x_mlp2 = self.mlp2(combined)
        emb = x_mlp2 + E_i

        # Multi-scale processing
        skips = []
        if self.skips:
            for i, layer in enumerate(self.layers):
                # emb = layer(emb, edge_index, edge_attr)
                emb = checkpoint(layer, emb, edge_index, edge_attr, use_reentrant=False)
                if i < len(self.skips):
                    skips.append(self.skips[i](emb))

            # Skip fusion
            emb = emb + torch.stack(skips).mean(dim=0)

        # emb = F.layer_norm(emb, emb.shape)

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
        self.threshold_edge_exist = threshold_edge_exist
        self.threshold_subgraph_overlap = threshold_subgraph_overlap
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
        self._init_pooling_layers(
            fusion_dims_node_emb[-1], output_dim_g_emb, n_heads_pooling
        )

    def _init_pooling_layers(self, input_dim, output_dim, heads):
        # Local subgraph pooling
        self.xgat_pool = WGATLayer(
            input_dim, output_dim, heads, self.dropout, self.chunk_size
        )
        self.att_pool = SelfAtt_(output_dim, self.dropout, True)

        # Global graph pooling
        self.global_xgat_pool = WGATLayer(
            output_dim, output_dim, heads, self.dropout, self.chunk_size
        )
        self.global_att_pool = SelfAtt_(output_dim, self.dropout, True)

    def _forward_embedding(self, node_names, x, edge_attr, edge_index):
        return self.node_embedding_layers(node_names, x, edge_attr, edge_index)

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

            # Extract edges for the current graph
            edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            sub_edge_index = edge_index[:, edge_mask]
            # Adjust edge indices to local indices
            local_node_ids = torch.arange(mask.sum(), device=x.device)
            global_to_local = torch.zeros_like(mask, dtype=torch.long, device=x.device)
            global_to_local[node_indices] = local_node_ids
            sub_edge_index = global_to_local[sub_edge_index]

            # Compute dynamic centrality
            filtered_edge_index, centrality = self._calculate_dynamic_centrality(
                h[mask],
                edge_attr[edge_mask] if edge_attr is not None else None,
                sub_edge_index,
            )

            # Generate multiscale subgraphs
            subgraphs = self._generate_multiscale_subgraphs(
                filtered_edge_index, mask.sum(), centrality, h[mask], x.device
            )

            # Create graph embeddings
            if subgraphs:
                g_emb = self._create_graph_embeddings(subgraphs)
                # g_emb = checkpoint(
                #     self._create_graph_embeddings, subgraphs, use_reentrant=False
                # )
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
            cos_sim = []

            # Calculate cosine similarity for each edge in batches to reduce peak memory usage.
            for i in range(0, num_edges, self.chunk_size):
                idx = slice(i, min(i + self.chunk_size, num_edges))
                h_i = h[row[idx]]
                h_j = h[col[idx]]
                # Cosine similarity and immediately release intermediate tensors
                batch_cos = F.cosine_similarity(h_i, h_j, dim=1).abs()
                cos_sim.append(batch_cos)

                # del h_i, h_j, batch_cos

            cos_sim = torch.cat(cos_sim, dim=0)

            # Combine with edge attributes
            edge_weight = (
                cos_sim * edge_attr.view(-1) if edge_attr is not None else cos_sim
            )

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
        q_high = torch.quantile(centrality, const.default.threshold_centrality)
        central_nodes = torch.where(centrality > q_high)[0]

        node_degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        central_nodes = central_nodes[(node_degrees[central_nodes] >= 2)]

        subsets, subg_edge_indices, _, _ = zip(
            *[
                k_hop_subgraph(
                    int(node_idx),
                    self.n_hop,
                    edge_index,
                    num_nodes,
                    flow="source_to_target",
                )
                for node_idx in central_nodes
            ]
        )

        subgraph_pool = []
        for i, (subset, subg_edge_idx) in enumerate(zip(subsets, subg_edge_indices)):
            if len(subset) >= 2 and subg_edge_idx.size(1) > 0:
                node_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device)
                node_mapping[subset] = torch.arange(len(subset), device=device)
                subg_edge_local = node_mapping[subg_edge_idx]

                subgraph_pool.append(
                    GData(
                        x=h[subset],
                        edge_index=subg_edge_local,
                        center_node=central_nodes[i],
                        node_idx=subset,
                    )
                )

        subgraphs = sorted(subgraph_pool, key=lambda x: -x.num_nodes)

        coverage_bits = torch.zeros(num_nodes, dtype=torch.int32, device=device)
        merged = []
        for subg in subgraphs:
            subg_bits = torch.zeros(num_nodes, dtype=torch.int32, device=device)
            subg_bits[subg.node_idx] = 1

            overlap_ratio = (coverage_bits & subg_bits).sum().float() / subg.num_nodes
            if overlap_ratio < self.threshold_subgraph_overlap:
                merged.append(subg)
                coverage_bits |= subg_bits

        return merged

    def _create_graph_embeddings(self, subgraphs):
        # Process each subgraph
        embs = torch.cat([self._process_subgraph(g) for g in subgraphs], dim=0)

        # Build super graph
        super_nodes = GData(
            x=embs,
            edge_index=to_undirected(
                torch.combinations(torch.arange(len(subgraphs), device=embs.device)).t()
            ),
        )

        # Global pooling
        global_emb = self.global_xgat_pool(super_nodes.x, super_nodes.edge_index)
        return self.global_att_pool(global_emb)

    def _process_subgraph(self, subgraph):
        assert (subgraph.edge_index.numel() == 0) or (
            subgraph.edge_index.max() < subgraph.x.size(0)
        ), "Edge index exceeds node count"
        h = self.xgat_pool(subgraph.x, subgraph.edge_index)
        return self.att_pool(h.unsqueeze(0))  # .squeeze(0)

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

        self.trans = nn.Linear(input_dim, output_dim * num_heads)
        # Adjusted for per-head attention computation
        self.W = nn.Parameter(torch.empty(2 * input_dim, num_heads * output_dim))
        self.attn = nn.Parameter(torch.empty(num_heads, output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W.data)
        nn.init.kaiming_uniform_(self.attn.data, a=0.2, mode="fan_in")

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
        # return checkpoint(
        #     self.propagate, edge_index, x=x, edge_attr=edge_attr, use_reentrant=False
        # )

    def message(self, x_i, x_j, edge_attr):
        num_edges = x_i.size(0)
        # if num_edges < 7:
        # raise ValueError("\nNumber of edges is too small.")
        # raise Warning("\nNumber of edges is too small.")
        # print("\nNumber of edges is too small.")
        # return torch.zeros_like(x_i)

        if num_edges > self.chunk_size:
            h_chunks = []
            e_chunks = []

            for i in range(0, num_edges, self.chunk_size):
                idx = slice(i, min(i + self.chunk_size, num_edges))

                x_i_chunk = x_i[idx]
                x_j_chunk = x_j[idx]
                edge_attr_chunk = edge_attr[idx] if edge_attr is not None else None

                h = (torch.cat([x_i_chunk, x_j_chunk], -1) @ self.W).view(
                    -1, self.num_heads, self.output_dim
                )
                e = torch.einsum("ehd,hd->eh", h, self.attn)

                if edge_attr_chunk is not None:
                    e = e * edge_attr_chunk.view(-1, 1).expand(-1, self.num_heads)

                h_chunks.append(h)
                e_chunks.append(e)

            e = torch.cat(e_chunks, dim=0)
            h = torch.cat(h_chunks, dim=0)

        else:
            # Compute attention scores per head
            h = (torch.cat([x_i, x_j], -1) @ self.W).view(
                -1, self.num_heads, self.output_dim
            )

            # Calculate attention coefficients [E, num_heads]
            # e = (h * self.attn.unsqueeze(0)).sum(dim=-1)  # Dot product per head
            e = torch.einsum("ehd,hd->eh", h, self.attn)

            # Integrate edge attributes
            if edge_attr is not None:
                # Expand edge_attr to match num_heads [E, num_heads]
                e = e * edge_attr.view(-1, 1).expand(-1, self.num_heads)

        a = F.gelu(e)
        a = F.softmax(a, dim=0)
        x_trans = self.trans(x_j).view(-1, self.num_heads, self.output_dim)
        return torch.einsum("ehd,eh->ed", x_trans, a)

    # def message(self, x_i, x_j, edge_attr):
    #     # Compute attention scores per head
    #     h = (torch.cat([x_i, x_j], -1) @ self.W).view(
    #         -1, self.num_heads, self.output_dim
    #     )

    #     # Calculate attention coefficients [E, num_heads]
    #     # e = (h * self.attn.unsqueeze(0)).sum(dim=-1)  # Dot product per head
    #     e = torch.einsum("ehd,hd->eh", h, self.attn)

    #     # Integrate edge attributes
    #     if edge_attr is not None:
    #         # Expand edge_attr to match num_heads [E, num_heads]
    #         e = e * edge_attr.view(-1, 1).expand(-1, self.num_heads)

    #     # Normalize attention scores
    #     a = F.gelu(e)
    #     a = F.softmax(a, dim=0)
    #     # a = F.dropout(a, self.dropout, training=self.training)

    #     # Transform features and prepare multi-head output
    #     # [E, num_heads, output_dim]
    #     x_trans = self.trans(x_j).view(-1, self.num_heads, self.output_dim)

    #     # Weight features by attention scores
    #     # Average features across heads
    #     # h = (x_trans * a.unsqueeze(-1)).sum(dim=1)  # [E, output_dim]
    #     h = torch.einsum("ehd,eh->ed", x_trans, a)

    #     return h


class GE_Decoder(nn.Module):
    r"""
    Enhanced Graph Embedding Decoder with cross-attention and residual blocks
    """

    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        output_dim: int,
        hidden_dim: int,
        dropout: float = const.default.dropout,
        chunk_size: int = const.default.chunk_size,
        n_heads: int = 4,
        n_res_blocks: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size

        # Cross-attention layer
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=h_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

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

        if num_all_nodes <= self.chunk_size:
            # Create cross-attention input [batch_size, num_all_nodes, z_dim + h_dim]
            # [B, N, z_dim]
            z_expanded = z.unsqueeze(1).expand(-1, num_all_nodes, -1)
            # [B, N, h_dim]
            E_all_expanded = E_all.unsqueeze(0).expand(batch_size, -1, -1)

            # Feature fusion
            # [B, N, 2h_dim]
            fused = self.fusion(torch.cat([z_expanded, E_all_expanded], dim=-1))

            # Cross-attention
            # [B, N, h_dim]
            attn_out, _ = self.cross_attn(query=fused, key=fused, value=fused)

            # Residual learning
            for block in self.res_blocks:
                attn_out = attn_out + block(attn_out)

        else:
            h_chunks = []

            for i in range(0, num_all_nodes, self.chunk_size):
                end_idx = min(i + self.chunk_size, num_all_nodes)
                current_chunk = E_all[i:end_idx]

                z_chunk = z.unsqueeze(1).expand(-1, end_idx - i, -1)
                E_chunk = current_chunk.unsqueeze(0).expand(z.size(0), -1, -1)

                fused_chunk = self.fusion(torch.cat([z_chunk, E_chunk], dim=-1))
                attn_chunk, _ = self.cross_attn(fused_chunk, fused_chunk, fused_chunk)

                for block in self.res_blocks:
                    attn_chunk = attn_chunk + block(attn_chunk)

                h_chunks.append(attn_chunk)

            attn_out = torch.cat(h_chunks, dim=1)

        # Final reconstruction
        # [B, N, output_dim]
        h_ = self.ffn_q(attn_out)

        return attn_out, h_


class GLabelPredictor(nn.Module):
    r"""
    A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
    """

    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float
    ):
        super().__init__()

        layers = [
            SelfAtt_(input_dim, dropout),
            nn.GELU(),
        ]
        for dim in hidden_dims:
            layers += [
                nn.Linear(input_dim, dim),
                nn.GELU(),
            ]
            input_dim = dim
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.Linear(input_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.net(x)


class SelfAtt_(nn.Module):
    r"""
    Self-attention (pooling) layer.
    """

    def __init__(
        self, dim: int, dropout: float = const.default.dropout, pool: bool = False
    ):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.scale = dim**-0.5
        self.dropout = nn.Dropout(dropout)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout.p if self.dropout else 0.0, scale=self.scale
        )
        if self.pool:
            x = x.mean(dim=0)
        x = self.proj(x)
        return x

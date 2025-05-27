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
                # print("Warning: Empty graph detected")
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

    def _generate_multiscale_subgraphs(self, edge_index, centrality, h) -> List[GData]:
        """优化后的多尺度子图生成方法"""
        device = h.device
        num_nodes = h.size(0)
        _, edges = self.hist_fd(centrality)

        # 批量处理所有候选节点
        valid_centers = torch.where(centrality > edges[0])[0]
        if valid_centers.numel() == 0:
            return []

        # 批量生成k-hop子图
        batch_subsets, batch_edges = self._batch_k_hop_subgraph(valid_centers, self.n_hop, edge_index, num_nodes)

        # 构建覆盖矩阵（使用稀疏格式节省内存）
        indices = []
        for i, subset in enumerate(batch_subsets):
            if subset.numel() > 0:
                indices.append(torch.stack([torch.full_like(subset, i), subset]))
        if not indices:
            return []

        coverage_indices = torch.cat(indices, dim=1).to(device)
        coverage_values = torch.ones(coverage_indices.shape[1], device=device)
        coverage_matrix = torch.sparse_coo_tensor(indices=coverage_indices, values=coverage_values, size=(len(batch_subsets), num_nodes)).coalesce()

        # 计算子图相似度矩阵
        overlap_matrix = torch.sparse.mm(coverage_matrix, coverage_matrix.t().to_dense())
        min_sizes = torch.min(
            coverage_matrix._values().sum(dim=0).unsqueeze(0),
            coverage_matrix._values().sum(dim=0),
        )
        overlap_matrix = overlap_matrix / (min_sizes + 1e-8)

        # 寻找需要合并的子图
        merge_mask = overlap_matrix > self.thre_sg_overlap
        components = self._find_connected_components(merge_mask)

        # 合并子图
        merged_subsets = []
        for comp in components:
            merged_mask = coverage_matrix.to_dense()[comp].sum(dim=0) > 0
            merged_subsets.append(merged_mask)

        # 生成最终子图对象
        subgraphs = []
        for mask in merged_subsets:
            node_idx = mask.nonzero(as_tuple=True)[0]
            if node_idx.numel() > 1:  # 过滤单节点子图
                subgraphs.append(
                    GData(
                        x=h[node_idx],
                        edge_index=self._fast_edge_index(edge_index, mask),
                        node_idx=node_idx,
                        mask=mask,
                    )
                )

        return sorted(subgraphs, key=lambda x: -x.num_nodes)

    @staticmethod
    def _find_connected_components(adj_matrix: torch.Tensor) -> List[torch.Tensor]:
        """使用BFS寻找连通分量（优化GPU实现）"""
        n = adj_matrix.size(0)
        visited = torch.zeros(n, dtype=torch.bool, device=adj_matrix.device)
        components = []

        for i in range(n):
            if not visited[i]:
                queue = [i]
                visited[i] = True
                component = []

                while queue:
                    node = queue.pop(0)
                    component.append(node)
                    neighbors = torch.where(adj_matrix[node])[0]
                    valid_neighbors = neighbors[~visited[neighbors]]
                    if valid_neighbors.numel() > 0:
                        visited[valid_neighbors] = True
                        queue.extend(valid_neighbors.tolist())

                if component:
                    components.append(torch.tensor(component, device=adj_matrix.device))

        return components

    @staticmethod
    def hist_fd(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """优化后的分箱策略（完全GPU实现）"""
        n = tensor.numel()
        if n == 0:
            return torch.tensor([]), torch.tensor([])

        q = torch.quantile(tensor, torch.tensor([0.25, 0.75], device=tensor.device))
        iqr = q[1] - q[0]
        bin_width = (2 * iqr) / (n ** (1 / 3)) if iqr > 0 else (tensor.max() - tensor.min())

        num_bins = torch.floor((tensor.max() - tensor.min()) / (bin_width + 1e-8)).int()
        num_bins = torch.clamp(num_bins, min=1, max=1000)

        edges = torch.linspace(tensor.min().item(), tensor.max().item(), num_bins.item() + 1, device=tensor.device)
        hist = torch.histc(tensor, bins=num_bins.item(), min=tensor.min().item(), max=tensor.max().item())
        return hist, edges

    def _calculate_dynamic_centrality(self, h, edge_index):
        """优化后的动态中心性计算"""
        row, col = edge_index
        if row.numel() == 0:
            return edge_index, torch.zeros(h.size(0), device=h.device)

        # 向量化相似度计算
        h_i = h[row]
        h_j = h[col]
        sim = (h_i * h_j).sum(dim=1).abs()

        # 标准化处理
        sim_min, sim_max = sim.min(), sim.max()
        sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-8)

        # 构建稀疏邻接矩阵
        mask = sim_norm > self.thre_edge_exist
        filtered_edge = edge_index[:, mask]
        sim_filtered = sim_norm[mask]

        # 计算中心性得分
        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.index_add_(0, filtered_edge[0], sim_filtered)
        centrality.index_add_(0, filtered_edge[1], sim_filtered)

        return filtered_edge, centrality

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
            # nn.GELU(),
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
        # self.chunk_size = chunk_size

        # Split weight matrix into two parts to avoid concatenation
        self.W_i = nn.Linear(input_dim, output_dim * num_heads)
        self.W_j = nn.Linear(input_dim, output_dim * num_heads)

        self.trans = nn.Linear(input_dim, output_dim * num_heads)
        self.attn = nn.Parameter(torch.empty(num_heads, output_dim))
        self.reset_parameters()

        # Adaptive chunk size calculation
        self.adap_chunk_size = GetAdaptiveChunkSize()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W_i.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.W_j.weight, gain=nn.init.calculate_gain("relu"))
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
                a = a * edge_attr[idx].view(-1, 1)

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


class GLabelPredictor(nn.Module):
    r"""
    A graph-level label predictor that predicts the label of graphs based on the graph embeddings.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float, n_heads: int):
        super().__init__()

        # layers = [
        #     SelfAtt_(input_dim, dropout),
        #     nn.LayerNorm(input_dim),
        # ]
        _in_dim = input_dim
        layers = []
        for _dim in hidden_dims:
            layers += [
                nn.Linear(_in_dim, _dim),
                nn.GELU(),
            ]
            _in_dim = _dim
        layers.append(nn.LayerNorm(_in_dim))
        layers.append(nn.Linear(_in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            x = x.squeeze(2)
        return self.net(x)


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

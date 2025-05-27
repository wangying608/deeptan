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

# 禁用非必要警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_geometric")


class AMSGP(torch.nn.Module):
    r"""
    Optimized Adaptive Multi-Scale Graph Pooling for Graph-Level Representation Learning.
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
        super().__init__()
        # 保存基础参数
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

        # 启用内存优化配置
        self.use_sparse_masks = True
        self.enable_amp = True  # 自动混合精度
        self.optimize_for_inference = False  # 推理优化开关

        # 初始化节点嵌入层
        self.node_embedding_layers = NodeEmbedding(
            input_dim,
            node_emb_dim,
            fusion_dims_node_emb,
            dict_node_names,
            n_heads_node_emb,
            dropout,
            chunk_size,
        )

        # 多尺度池化架构优化
        self._init_pooling_layers(fusion_dims_node_emb[-1], output_dim_g_emb, n_heads_pooling)

        # 自适应块大小计算器
        self.adap_chunk_size_dyn_cent = GetAdaptiveChunkSize()
        self.adap_chunk_size_mul_subg = GetAdaptiveChunkSize()

        # 编译关键函数
        if torch.__version__ >= "2.0":
            self._calculate_dynamic_centrality = torch.compile(self._calculate_dynamic_centrality, mode="reduce-overhead")
            self._generate_multiscale_subgraphs = torch.compile(self._generate_multiscale_subgraphs, mode="reduce-overhead")

    def _init_pooling_layers(self, input_dim, output_dim, heads):
        """初始化多尺度池化层（优化版）"""
        # 多尺度GAT池化
        self.multi_scale_pool = nn.ModuleList(
            [
                WGATLayer_chunked(input_dim, output_dim, heads, self.dropout, self.chunk_size, kernel_size=k)
                for k in [1, 2, 3]  # 不同感受野尺寸
            ]
        )

        # 层次注意力融合
        self.hierarchical_att = SelfAtt_(output_dim * len(self.multi_scale_pool))

        # 全局池化优化
        self.global_pool = nn.ModuleList(
            [
                WGATLayer_chunked(output_dim, output_dim, heads, self.dropout, self.chunk_size)
                for _ in range(2)  # 双阶段全局池化
            ]
        )
        self.global_att_pool = SelfAtt_(output_dim, self.dropout, True)

    def forward(self, node_names, x, edge_attr, edge_index, batch):
        """前向传播（优化版）"""
        # 节点嵌入
        h, E_all, ids = self.node_embedding_layers(node_names, x, edge_attr, edge_index)

        # 图嵌入处理
        unique_batches = torch.unique(batch)
        graph_embs = []
        self._device = x.device

        for graph_id in unique_batches:
            # 提取当前图的节点
            mask = batch == graph_id
            node_indices = torch.where(mask)[0]

            # 空图处理优化
            if node_indices.numel() == 0:
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            # 子图提取优化
            try:
                sub_edge_index, _ = subgraph(node_indices, edge_index, relabel_nodes=True)
            except IndexError:
                # 异常情况处理
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))
                continue

            # 动态中心性计算（AMP优化）
            with torch.cuda.amp.autocast(enabled=self.enable_amp):
                h_masked = h[mask]
                filtered_edge_index, centrality = self._calculate_dynamic_centrality(h_masked, sub_edge_index)

            # 多尺度子图生成（稀疏张量优化）
            subgraphs = self._generate_multiscale_subgraphs(filtered_edge_index, centrality, h_masked)

            # 图嵌入生成（混合精度优化）
            if subgraphs:
                with torch.cuda.amp.autocast(enabled=self.enable_amp):
                    g_emb = self._create_graph_embeddings(subgraphs)
                graph_embs.append(g_emb)
            else:
                graph_embs.append(torch.zeros(self.output_dim_g_emb, device=self._device))

        return torch.stack(graph_embs), E_all, ids

    def _calculate_dynamic_centrality(self, h, edge_index):
        """动态中心性计算优化（向量化实现）"""
        chunk_size = self.adap_chunk_size_dyn_cent.calc(h.size(), 0)

        # 预分配存储空间
        sim_ = torch.empty(edge_index.size(1), device=h.device)

        # 向量化计算相似度
        row, col = edge_index
        for i in range(0, edge_index.size(1), chunk_size):
            idx = slice(i, min(i + chunk_size, edge_index.size(1)))
            h_i = h[row[idx]]
            h_j = h[col[idx]]
            sim_[idx] = (h_i * h_j).sum(dim=1).abs()

        # 归一化处理
        sim_min, sim_max = sim_.min(), sim_.max()
        if sim_max > sim_min:
            sim_ = (sim_ - sim_min) / (sim_max - sim_min)

        # 边过滤优化
        mask = sim_ > self.thre_edge_exist
        filtered_edge = edge_index[:, mask]
        filtered_weight = sim_[mask]

        # 中心性计算优化
        centrality = torch.zeros(h.size(0), device=h.device)
        centrality.scatter_add_(0, filtered_edge[0], filtered_weight)
        centrality.scatter_add_(0, filtered_edge[1], filtered_weight)

        return filtered_edge, centrality

    def _generate_multiscale_subgraphs(self, edge_index, centrality, h) -> List[GData]:
        """生成多尺度子图（稀疏张量优化版）"""
        device = h.device
        num_nodes = h.size(0)

        # 使用稀疏张量存储
        if self.use_sparse_masks:
            subgraph_masks = torch.sparse_coo_tensor(size=(0, num_nodes), dtype=torch.bool, device=device)
        else:
            subgraph_masks = torch.zeros((0, num_nodes), dtype=torch.bool, device=device)

        subgraph_centers = torch.tensor([], dtype=torch.long, device=device)
        covered_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=device)

        # 节点排序优化
        node_order = torch.argsort(centrality, descending=True)
        sorted_nodes = node_order[centrality > self.thre_edge_exist]

        # 自适应分块处理
        chunk_size = self.adap_chunk_size_mul_subg.calc((sorted_nodes.shape[0], edge_index.shape[1], h.shape[1]), 0)

        # 向量化k-hop计算
        for chunk in sorted_nodes.split(chunk_size):
            if chunk.numel() == 0:
                continue

            # 批量计算k-hop子图
            try:
                subsets = k_hop_subgraph(chunk.tolist(), self.n_hop, edge_index, num_nodes=num_nodes, relabel_nodes=True, directed=False)[0]
            except Exception:
                continue

            # 处理每个子图
            for i, subset in enumerate(subsets):
                if subset.numel() < 2:
                    continue

                # 创建子图mask
                new_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
                new_mask[subset] = True

                # 重叠检测优化
                if self.use_sparse_masks and subgraph_masks.size(0) > 0:
                    # 稀疏张量快速求交
                    intersections = torch.sparse.mm(subgraph_masks.float(), new_mask.float())
                    overlaps = intersections / (subgraph_masks.to_dense().sum(dim=1) + 1e-8)
                    overlapping = overlaps > self.thre_sg_overlap
                else:
                    # 常规检测
                    overlapping = torch.any((subgraph_masks & new_mask).sum(dim=1) / torch.minimum(subgraph_masks.sum(dim=1), new_mask.sum()) > self.thre_sg_overlap, dim=1)

                # 合并重叠子图
                if torch.any(overlapping):
                    merged_mask = subgraph_masks[overlapping].any(dim=0) | new_mask
                    subgraph_masks = torch.cat([subgraph_masks[~overlapping], merged_mask.unsqueeze(0)], dim=0)
                    subgraph_centers = torch.cat([subgraph_centers[~overlapping], torch.tensor([chunk[i]], device=device)])
                else:
                    # 添加新子图
                    subgraph_masks = torch.cat([subgraph_masks, new_mask.unsqueeze(0)], dim=0)
                    subgraph_centers = torch.cat([subgraph_centers, torch.tensor([chunk[i]], device=device)])

                covered_nodes |= new_mask

                # 提前终止条件
                if covered_nodes.all():
                    break

        # 转换为GData对象
        subgraphs = []
        for mask, center in zip(subgraph_masks, subgraph_centers):
            node_idx = torch.where(mask)[0]
            subgraphs.append(
                GData(
                    x=h[node_idx],
                    edge_index=self._fast_edge_index(edge_index, mask),
                    center_node=center,
                    node_idx=node_idx,
                    mask=mask,
                )
            )

        # 按大小排序
        return sorted(subgraphs, key=lambda x: -x.num_nodes)

    def _create_graph_embeddings(self, subgraphs):
        """创建图嵌入（优化版）"""
        if not subgraphs:
            return torch.zeros(self.output_dim_g_emb, device=self._device)

        # 多尺度池化
        multi_scale_embs = []
        for pool_layer in self.multi_scale_pool:
            embs = torch.cat([self._process_subgraph(g, pool_layer) for g in subgraphs], dim=0)
            multi_scale_embs.append(embs)

        # 拼接多尺度特征
        combined_embs = torch.cat(multi_scale_embs, dim=1)

        # 构建超级图
        num_subgraphs = len(subgraphs)
        edge_index = to_undirected(torch.combinations(torch.arange(num_subgraphs, device=combined_embs.device)).t())

        # 全局池化优化
        global_emb = combined_embs
        for pool in self.global_pool:
            global_emb = pool(global_emb, edge_index)

        return self.global_att_pool(global_emb)

    def _process_subgraph(self, subgraph, pool_layer=None):
        """处理单个子图（优化版）"""
        if pool_layer is None:
            pool_layer = self.xgat_pool

        h = pool_layer(subgraph.x, subgraph.edge_index)
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

r"""
Modules for multi-task graph learning.
"""
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data as GData
import graph_tool.all as gt
import frn.constants as const


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

        if g.edge_attr is None:
            out = self.propagate(g.edge_index, x=g.x, edge_attr=None)
        else:
            out = self.propagate(g.edge_index, x=g.x, edge_attr=g.edge_attr.view(-1,1))
        return out
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: Optional[torch.Tensor]):
        # edge_attr has shape: torch.Size([E, 1])
        h_cat = torch.cat([x_i, x_j], dim=-1)
        # Shape of h_cat: (E, 2 * input_dim)
        Wh = (h_cat @ self.W)
        # Shape of Wh: (E, output_dim)
        e_ij: torch.Tensor = self.activation(Wh.matmul(self.attn))
        if edge_attr is None:
            e_ij = e_ij.softmax(dim=0)
        else:
            # Mask attention scores with adjacency matrix (MIC)
            e_ij = e_ij.mul(edge_attr).softmax(dim=0)
        # Shape of e_ij: (E, 1)
        # Output shape: (E, output_dim)
        # Compute transformed node embeddings
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
        c_BCE_o = torch.tensor(c_BCE, dtype=torch.float32, device=g_pyg.x.device)
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
        # self.attention = nn.Linear(input_dim, 1)
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
        )
    
    def forward(self, g_emb: torch.Tensor):
        r"""
        Returns:
            Pooled representation of shape ``(output_dim_g_emb,)``.
        """
        weight_per_g = self.attention(g_emb).softmax(dim=0)
        weighted = g_emb.mul(weight_per_g)
        # pooled = weighted.sum(dim=0)

        # Concatenate the pooled representation with the mean and std of the graphs
        out_emb = torch.cat([weighted.sum(dim=0), weighted.mul(weighted.shape[0]).std(dim=0), g_emb.mean(dim=0), g_emb.std(dim=0)])

        # Shape of the output: (graph_embedding_dim * 4)
        return out_emb


class VGAE_Decoder(nn.Module):
    def __init__(self, graph_embedding_dim: int, node_embedding_dim: int):
        super().__init__()
        self.graph_embedding_dim = graph_embedding_dim
        self.node_embedding_dim = node_embedding_dim
        
        self.guess_num_nodes = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.Sigmoid(),
            nn.Linear(graph_embedding_dim // 2, 1),
            nn.Softplus(),
        )
        self.fc_mu = nn.Linear(graph_embedding_dim, graph_embedding_dim)
        self.fc_var = nn.Linear(graph_embedding_dim, graph_embedding_dim)

        self.decoder = nn.Sequential(
            nn.Linear(graph_embedding_dim, graph_embedding_dim // 2),
            nn.LeakyReLU(),
            nn.Linear(graph_embedding_dim // 2, node_embedding_dim * 4),
            nn.Tanh(),
            nn.Linear(node_embedding_dim * 4, node_embedding_dim),
            nn.LeakyReLU(),
        )
    
    def forward(self, graph_embedding: torch.Tensor):
        num_nodes = torch.exp(self.guess_num_nodes(graph_embedding)).round().long()
        graph_embedding_expanded = graph_embedding.unsqueeze(0).expand(int(num_nodes.item()), -1)
        # Shape of graph_embedding_expanded: (num_nodes, graph_embedding_dim)

        # mu, var
        mu = self.fc_mu(graph_embedding_expanded)
        log_var = self.fc_var(graph_embedding_expanded)
        z = self.reparameterize(mu, log_var)
        
        node_embeddings = self.decoder(z)
        
        return node_embeddings

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu


class GLabelPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.predictor(x)

r"""
GAT model
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.pool import SAGPooling
import frn.constants as const


torch.set_float32_matmul_precision(const.default.float32_matmul_precision)


class Backbone(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden_dim: int,
            # edge_dim: int,
            heads: int,
            dropout: float,
            negative_slope: float,
        ):
        super().__init__()

        # Define the GAT layer 1
        self.gatv2_1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            concat=True,
            negative_slope=negative_slope,
            dropout=dropout,
            # edge_dim=edge_dim,
            share_weights=False,
        )

        # Define the GAT layer 2
        self.gatv2_2 = GATv2Conv(
            in_channels=hidden_dim * heads,
            out_channels=out_channels,
            heads=heads,
            concat=False,
            negative_slope=negative_slope,
            dropout=dropout,
            # edge_dim=edge_dim,
            share_weights=False,
        )
        
        # Define the pooling layers
        self.pool_1 = SAGPooling(hidden_dim * heads, ratio=0.5)
        self.pool_2 = SAGPooling(out_channels, ratio=0.6)
    
    def forward(self, data_batch):
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch

        x = self.gatv2_1(x, edge_index)
        x = torch.relu(x)
        # raise Warning(f"\nx dimension: {x.shape}\n")
        x, edge_index, edge_attr, batch, _perm, _score = self.pool_1(x, edge_index, batch=batch)
        
        x = self.gatv2_2(x, edge_index)
        x = torch.relu(x)
        # raise Warning(f"\nx dimension: {x.shape}\n")
        x, edge_index, edge_attr, batch, _perm, _score = self.pool_2(x, edge_index, batch=batch)
        
        # raise Warning(f"\nx dimension: {x.shape}\n")
        x = global_mean_pool(x, batch)

        return x


r"""
GAT model
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.nn.pool import SAGPooling


torch.set_float32_matmul_precision('high')


class MyGATModel(nn.Module):
    def __init__(self, in_channels: int, graph_label_dim: int, edge_dim: int, heads: int, dropout: float, negative_slope: float):
        super().__init__()
        out_channels_1 = in_channels
        out_channels_2 = graph_label_dim * 4

        # Define the GAT layer 1
        self.gatv2_1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels_1,
            heads=heads,
            concat=False,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_dim,
            share_weights=True,
        )

        # Define the GAT layer 2
        self.gatv2_2 = GATv2Conv(
            in_channels=out_channels_1,
            out_channels=out_channels_2,
            heads=heads,
            concat=False,
            negative_slope=negative_slope,
            dropout=dropout,
            edge_dim=edge_dim,
            share_weights=True,
        )
        
        # Define the pooling layers
        self.pool_1 = SAGPooling(out_channels_2, ratio=0.5)
        self.pool_2 = SAGPooling(out_channels_2, ratio=0.7)

        # Define the output layer for graph label prediction
        self.output_layer = nn.Linear(out_channels_2, graph_label_dim)
    
    def forward(self, data_batch):
        x, edge_index, edge_attr, batch_idx = data_batch.x, data_batch.edge_index, data_batch.edge_attr, data_batch.batch
        x = self.gatv2_1(x, edge_index, edge_attr)
        x, edge_index, edge_attr, _ = self.pool_1(x, edge_index, edge_attr, batch_idx)
        x = self.gatv2_2(x, edge_index, edge_attr)
        x, edge_index, edge_attr, _ = self.pool_2(x, edge_index, edge_attr, batch_idx)
        x = global_mean_pool(x, batch_idx)
        return self.output_layer(x)


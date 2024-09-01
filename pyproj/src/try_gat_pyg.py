import os
import torch
# import torch_geometric.loader as geom_loader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FakeDataset, TUDataset
# from torch_geometric.data.lightning import LightningDataset as PyG_LightningDataset
from frn.utils.uni import time_string, train_model
from frn.graph.pipeline_general import MyGAT


# n_channels=64
# edge_dim=16
# graph_label_dim=1

# datamodule = PyG_LightningDataset(
#     train_dataset=FakeDataset(num_graphs=300, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     val_dataset=FakeDataset(num_graphs=100, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     # test_dataset=FakeDataset(num_graphs=100, num_channels=64, edge_dim=16, task="graph"),
# )
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_datasets")
BATCH_SIZE = 16

if __name__ == "__main__":
    tu_dataset = TUDataset(root=DATASET_PATH, name="MUTAG")
    torch.manual_seed(42)
    tu_dataset.shuffle()
    graph_train_loader = DataLoader(tu_dataset[:100], batch_size=BATCH_SIZE, shuffle=True)
    graph_val_loader = DataLoader(tu_dataset[100:], batch_size=BATCH_SIZE)

    dim_input = tu_dataset.num_node_features
    dim_output = tu_dataset.num_classes
    # dim_edge_feat = tu_dataset.num_edge_features

    _model = MyGAT(
        in_channels=dim_input,
        graph_label_dim=dim_output,
        regression=False,
        # edge_dim=dim_edge_feat,
        hidden_dim=32,
        heads=8,
        lr=0.0001,
        dropout=0.3,
        negative_slope=0.2,
    )

    loss_min = train_model(
        model=_model,
        dataloader_train=graph_train_loader,
        dataloader_val=graph_val_loader,
        es_patience=15,
        max_epochs=1000,
        min_epochs=10,
        log_dir=f".tmp/runs/{time_string()}",
        in_dev=False,
    )

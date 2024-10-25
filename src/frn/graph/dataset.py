import os
import torch
# import torch_geometric.loader as geom_loader
from torch_geometric.data import DataLoader
from torch_geometric.datasets import FakeDataset, TUDataset
from lightning import LightningDataModule


# n_channels=64
# edge_dim=16
# graph_label_dim=1

# datamodule = PyG_LightningDataset(
#     train_dataset=FakeDataset(num_graphs=300, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     val_dataset=FakeDataset(num_graphs=100, num_channels=n_channels, edge_dim=edge_dim, task="graph"),
#     # test_dataset=FakeDataset(num_graphs=100, num_channels=64, edge_dim=16, task="graph"),
# )


class _tmp_GraphDataModule_MUTAG(LightningDataModule):
    def __init__(self):
        super().__init__()
        # self.DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_datasets")
        self.DATASET_PATH = f".tmp/datasets"
        self.BATCH_SIZE = 32
        
    def setup(self, stage=None):
        tu_dataset = TUDataset(root=self.DATASET_PATH, name="MUTAG")
        self.dim_input = tu_dataset.num_node_features
        self.dim_output = tu_dataset.num_classes
        self.dim_edge_feat = tu_dataset.num_edge_features

        torch.manual_seed(42)
        tu_dataset.shuffle()
        self.graph_train_loader = DataLoader(tu_dataset[:100], batch_size=self.BATCH_SIZE, shuffle=True)
        self.graph_val_loader = DataLoader(tu_dataset[100:], batch_size=self.BATCH_SIZE)
    
    def train_dataloader(self):
        return self.graph_train_loader
    
    def val_dataloader(self):
        return self.graph_val_loader

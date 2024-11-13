r"""
Graph data module.
"""
from typing import Optional
import numpy as np
import torch
from torch_geometric.data import Data as GData
from torch_geometric.data import Dataset as GDataset
from torch_geometric.loader import DataLoader as GDataLoader
# from torch_geometric.data.lightning import LightningDataset as GLightningDataset
from torch_geometric.utils import to_undirected
from lightning import LightningDataModule


def random_graph_data(node_dim: int = 16, num_nodes_max: int = 233):
    num_nodes = np.random.randint(num_nodes_max // 2, num_nodes_max + 1)
    
    # edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))
    # edge_index = to_undirected(edge_index)
    # num_edges = edge_index.shape[1]
    # edge_attr = torch.randn(num_edges).abs()

    adj = torch.triu(torch.randn(num_nodes, num_nodes))
    adj = adj + adj.T
    adj = torch.pow(adj, 2)
    # Simulate sparse connectivity.
    adj = adj.where(adj > 0.8, torch.zeros_like(adj)).to_sparse(layout=torch.sparse_coo)

    x = torch.randn(num_nodes, node_dim)
    y = torch.randn(1)
    return GData(x=x, edge_index=adj.indices(), edge_attr=adj.values(), y=y)


class RandomGraphDataset(GDataset):
    def __init__(self, num_graphs: int = 100, num_nodes_max: int = 100, node_dim: int = 16):
        self.num_graphs = num_graphs
        self.num_nodes_max = num_nodes_max
        self.node_dim = node_dim
        super().__init__()

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return random_graph_data(node_dim=self.node_dim, num_nodes_max=self.num_nodes_max)

def random_datamodule(num_graphs: int, node_dim: int, num_nodes_max: int = 100, batch_size: int = 1):
    train_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim)
    val_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim)
    test_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim)
    # pred_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim)

    ltn_dm = GraphDataModule(train_dataset, val_dataset, test_dataset, batch_size)
    return ltn_dm


class GraphDataModule(LightningDataModule):
    def __init__(
            self,
            train_dataset: GDataset,
            val_dataset: GDataset,
            test_dataset: GDataset,
            batch_size: int = 1,
        ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return GDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return GDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return GDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


# if __name__ == '__main__':
#     print(type(random_datamodule(100, 16)))

#     torch.save(random_graph_data(), "gdata.pt")

# class GraphDataModule(LightningDataModule):
#     def __init__(self, batch_size=1, num_graphs=100, num_nodes_max=100):
#         super().__init__()
#         self.batch_size = batch_size
#         self.num_graphs = num_graphs
#         self.num_nodes_max = num_nodes_max

#     def setup(self, stage=None):
#         # 随机生成训练、验证和测试数据集
#         self.train_dataset = RandomGraphDataset(self.num_graphs, self.num_nodes_max)
#         self.val_dataset = RandomGraphDataset(self.num_graphs // 2, self.num_nodes_max)
#         self.test_dataset = RandomGraphDataset(self.num_graphs // 2, self.num_nodes_max)

#     def train_dataloader(self):
#         return geom_data.DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             follow_batch=['x', 'edge_index']
#         )

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             follow_batch=['x', 'edge_index']
#         )

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             follow_batch=['x', 'edge_index']
#         )


from torch_geometric.datasets import FakeDataset, TUDataset


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
        self.graph_train_loader = GDataLoader(tu_dataset[:100], batch_size=self.BATCH_SIZE, shuffle=True)
        self.graph_val_loader = GDataLoader(tu_dataset[100:], batch_size=self.BATCH_SIZE)
    
    def train_dataloader(self):
        return self.graph_train_loader
    
    def val_dataloader(self):
        return self.graph_val_loader

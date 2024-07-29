from p2_semisupervised import *


# Example usage:
# nfeat = number of features per node
# nhid = number of features in hidden layer
# nclass = number of classes
# dropout = dropout rate
# alpha = negative slope in leaky relu
# nheads = number of attention heads
#
# the adjacency matrix (adj) and feature matrix (x)

def init_graph_rand(max_num_nodes: int, nfeat: int, nclass: int):
    """
    Output:
    - `h`: node features. shape (max_num_nodes, nfeat)
    - `adj`: adjacency matrix. shape (max_num_nodes, max_num_nodes)
    - `y`: graph labels (onehot encoded if nclass > 1).
    """
    h = torch.randn(max_num_nodes, nfeat)
    adj = torch.randint(0, 2, (max_num_nodes, max_num_nodes)).float()
    if nclass == 1:
        y = torch.randn(1)
    else:
        y = torch.randint(0, nclass, (1,))
        y = F.one_hot(y, num_classes=nclass)
    return h, adj, y

# Dummy dataset for demonstration
class DummyGraphDataset(Dataset):
    def __init__(
            self,
            num_graphs: int,
            max_n_node: int,
            node_feat: int,
        ):
        self.data_list = []
        for _ in range(num_graphs):
            # num_nodes = int(torch.randint(10, 30, (1,)).item())
            # x = torch.randn(num_nodes, node_feat)
            # adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            # # y = torch.randint(0, 2, (num_nodes,))
            # y = int(torch.randint(0, 2, (1,)).item())
            # self.data_list.append((x, adj, y))
            self.data_list.append(init_graph_rand(max_n_node, node_feat, 1))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    n_graph = 1013
    max_n_node = 1117
    n_node_feat = 16
    dataset = DummyGraphDataset(n_graph, max_n_node, n_node_feat)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model_0 = ResPGAT(n_node_feat, 1, 3, 0.1, 0.1)
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model_0, dataloader)

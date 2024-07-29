# from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import Dataset, DataLoader

torch.set_float32_matmul_precision('high')


class GATv2Layer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int,
            concat: bool=True,
            dropout: float=0.0,
            negative_slope: float=0.1,
        ):
        super(GATv2Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.n_heads = n_heads
        self.concat = concat

        # Linear transformation
        self.Ws = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_heads)])

        # Shared Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # Concatenation of all attention heads
        if concat:
            self.concat_heads = nn.Linear(out_features * n_heads, out_features)

        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, input, adj):
        """
        Arguments:
        - `input`: (N, F_in)
        - `adj`: (N, N)
        
        These lead to batch size = 1
        """
        """
        Original implementation for one head:
        ```python
        h = torch.mm(input, self.W)
        # Number of nodes
        N = h.size()[0]

        # Compute attention coefficients
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        # Dim of a_input: (N, N, 2*F')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        # Dim of e: (N, N)

        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        # Apply mask to only include non-zero elements in the adjacency matrix
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        # To aggregate the feature information of adjacent nodes through attention mechanism and obtain new feature representations for each node
        h_prime = torch.matmul(attention, h)

        return h_prime
        ```
        """
        head_outputs = []
        for i in range(self.n_heads):
            Wh = self.Ws[i](input).squeeze(0)
            N = Wh.size(0)
            a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
            e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
            #
            print("\n")
            print(f"shape of e: {e.shape}")
            print(f"shape of adj: {adj.shape}")
            print(f"shape of h: {Wh.shape}")
            print(f"shape of a_input: {a_input.shape}")
            print(f"shape of a: {self.a.shape}")
            print("\n")
            """
            shape of e: torch.Size([53, 53])
            shape of adj: torch.Size([53, 53])
            shape of h: torch.Size([53, 1])
            shape of a_input: torch.Size([53, 53, 2])
            shape of a: torch.Size([2, 1])
            """
            #

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, Wh)
            head_outputs.append(h_prime.squeeze(0))
        
        if self.concat:
            output = self.concat_heads(torch.cat(head_outputs, dim=1))
        else:
            output = torch.mean(torch.stack(head_outputs), dim=0)
        
        return output


class GATv2(L.LightningModule):
    def __init__(
            self,
            nfeat: int,
            nhid: int,
            nclass: int,
            nheads: int,
            concat: bool=True,
            dropout: float=0.0,
            negative_slope: float=0.1,
        ):
        super(GATv2, self).__init__()
        self.concat = concat
        self.dropout = dropout

        self.gat_1 = GATv2Layer(nfeat, nhid, nheads, concat, dropout, negative_slope)

        # Output layer
        self.gat_2 = GATv2Layer(nhid, nclass, nheads, concat, dropout, negative_slope)

        # Aggregate graph representation
        # self.graph_aggregator = 

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat_1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat_2(x, adj)
        # x = F.log_softmax(x, dim=1)
        # x = torch.sum(x, dim=0)
        return x

    def training_step(self, batch, batch_idx):
        x, adj, labels = batch
        # output = self(x, adj).unsqueeze(0)
        # Run for each sample in batch
        sample_outputs = []
        for x_i, adj_i in zip(x, adj):
            print(x_i.shape)
            sample_outputs.append(self(x_i, adj_i))
        output = torch.cat(sample_outputs, dim=0)
        print(x.shape)#torch.Size([8, 53, 16])
        print(sample_outputs[0].shape)#torch.Size([53, 1])
        print(output.shape)#torch.Size([424, 1])
        print(labels.shape)#torch.Size([8, 1])
        loss = F.nll_loss(output, labels.squeeze(-1))
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)

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
            node_feat: int=16,
        ):
        self.data_list = []
        for _ in range(num_graphs):
            # num_nodes = int(torch.randint(10, 30, (1,)).item())
            # x = torch.randn(num_nodes, node_feat)
            # adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            # # y = torch.randint(0, 2, (num_nodes,))
            # y = int(torch.randint(0, 2, (1,)).item())
            # self.data_list.append((x, adj, y))
            self.data_list.append(init_graph_rand(53, node_feat, 1))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    dataset = DummyGraphDataset(100, 16)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    model_0 = GATv2(nfeat=16, nhid=8, nclass=1, nheads=3, concat=True, dropout=0.1, negative_slope=0.1)
    trainer = L.Trainer(fast_dev_run=True)
    trainer.fit(model_0, dataloader)

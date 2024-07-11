import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class GATv2Layer(nn.Module):
    def __init__(self, in_features: int, out_features: int, num_heads: int, concat: bool=True):
        super(GATv2Layer, self).__init__()
        self.num_heads = num_heads
        self.concat = concat
        self.attentions = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(num_heads)])
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        heads = []
        for attention in self.attentions:
            h = attention(x)
            f_1 = self.a(torch.cat([h.repeat(1, h.size(0), 1), h.repeat(h.size(0), 1, 1).transpose(0, 1)], dim=2))
            e = F.leaky_relu(f_1.squeeze(2))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=1)
            heads.append(torch.matmul(attention, h))
        if self.concat:
            return torch.cat(heads, dim=1)
        else:
            return torch.mean(torch.stack(heads), dim=0)


class GATv2Model(L.LightningModule):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_heads: int):
        super(GATv2Model, self).__init__()
        self.layer1 = GATv2Layer(in_channels, hidden_channels, num_heads)
        self.layer2 = GATv2Layer(hidden_channels * num_heads, out_channels, 1, concat=False)

    def forward(self, x, adj):
        x = F.elu(self.layer1(x, adj))
        x = self.layer2(x, adj)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, adj, y = batch
        out = self(x, adj)
        loss = F.nll_loss(out, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

# Dummy dataset for demonstration
class DummyGraphDataset(torch.utils.data.Dataset):
    def __init__(self, num_graphs):
        self.data_list = []
        for _ in range(num_graphs):
            num_nodes = int(torch.randint(10, 30, (1,)).item())
            x = torch.randn(num_nodes, 16)
            adj = torch.randint(0, 2, (num_nodes, num_nodes)).float()
            y = torch.randint(0, 2, (num_nodes,))
            self.data_list.append((x, adj, y))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

# Training setup
def train_model():
    dataset = DummyGraphDataset(num_graphs=100)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = GATv2Model(in_channels=16, hidden_channels=8, out_channels=2, num_heads=4)
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, loader)

if __name__ == "__main__":
    train_model()

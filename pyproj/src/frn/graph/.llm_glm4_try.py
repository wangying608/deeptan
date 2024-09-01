import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, SAGPooling, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

# 加载 MUTAG 数据集
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_datasets")
dataset = TUDataset(root=DATASET_PATH, name="MUTAG")
test_dataset = dataset[:20]  # 使用前20个样本作为测试集
train_dataset = dataset[20:]  # 剩余样本作为训练集

# 创建数据加载器
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义模型
class GATv2SAGPooling(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GATv2SAGPooling, self).__init__()
        self.conv1 = GATv2Conv(dataset.num_features, hidden_channels)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels)
        self.pool = SAGPooling(hidden_channels, ratio=0.8)
        self.lin = torch.nn.Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x, batch = self.pool(x, edge_index, batch)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return F.log_softmax(x, dim=1)

# 初始化模型
model = GATv2SAGPooling(hidden_channels=32)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(200):
    model.train()
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型
model.eval()
correct = 0
for data in test_loader:
    data = data.to(device)
    out = model(data.x, data.edge_index, data.batch)
    pred = out.argmax(dim=1)
    correct += pred.eq(data.y).sum().item()
accuracy = correct / len(test_dataset)
print(f'Accuracy: {accuracy:.4f}')

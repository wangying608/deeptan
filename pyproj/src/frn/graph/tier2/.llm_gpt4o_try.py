import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GATv2Conv, SAGPooling, global_mean_pool

# Define the GNN model
class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GATv2Conv(input_dim, hidden_dim, heads=8, dropout=0.6)
        self.pool1 = SAGPooling(hidden_dim * 8, ratio=0.5)
        self.conv2 = GATv2Conv(hidden_dim * 8, hidden_dim, heads=8, dropout=0.6)
        self.pool2 = SAGPooling(hidden_dim * 8, ratio=0.5)
        self.fc = nn.Linear(hidden_dim * 8, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x, edge_index, _, batch, perm, _ = self.pool1(x, edge_index, batch=batch)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x, edge_index, _, batch, perm, _ = self.pool2(x, edge_index, batch=batch)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Load dataset
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".tmp_datasets")
dataset = TUDataset(root=DATASET_PATH, name='MUTAG')

# Create data loaders
train_loader = DataLoader(dataset[:100], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[100:], batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
input_dim = dataset.num_node_features
hidden_dim = 64
output_dim = dataset.num_classes
model = GNNModel(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Training Loss: {total_loss / len(train_loader)}')

# Testing function
def test(loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
    acc = correct / len(dataset)
    print(f'Test Accuracy: {acc:.4f}')

# Train and test the model
for epoch in range(1, 11):
    print(f'Epoch {epoch}')
    train()
    test(test_loader)

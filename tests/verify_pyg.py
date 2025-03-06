import torch
from torch_geometric.data import Data

# Set the device to GPU if available
print(f"GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device=device)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float, device=device)

data = Data(x=x, edge_index=edge_index, y=None)

print("Data:", data)
print("Edge Index:", data.edge_index)
print("X:", data.x)
print(f"device: {data.x.device}")

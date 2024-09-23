import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from torch_sparse import SparseTensor

# Define the fixed indices for the sparse weights
fixed_indices = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long).t()

# Define the number of input and output features
input_features = 3
output_features = 3

# Create a sparse tensor with fixed indices and trainable values
sparse_values = nn.Parameter(torch.randn(fixed_indices.shape[1]))

# Define the sparse linear layer
class SparseLinear(nn.Module):
    def __init__(self, input_features, output_features, fixed_indices, sparse_values):
        super(SparseLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.fixed_indices = fixed_indices
        self.sparse_values = sparse_values

    def forward(self, x):
        # Create a sparse tensor from the fixed indices and trainable values
        sparse_weight = SparseTensor(row=self.fixed_indices[0], col=self.fixed_indices[1], value=self.sparse_values,
                                     sparse_sizes=(self.output_features, self.input_features))
        
        # Perform the sparse matrix multiplication
        return sparse_weight @ x.t()

# Instantiate the sparse linear layer
sparse_linear_layer = SparseLinear(input_features, output_features, fixed_indices, sparse_values)
print("Sparse Values:", sparse_linear_layer.sparse_values)

# Define a simple input tensor
input_tensor = torch.randn(1, input_features)

# Forward pass through the sparse linear layer
output = sparse_linear_layer(input_tensor)
print("Output:", output)

# Define a loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = SGD([sparse_linear_layer.sparse_values], lr=0.01)

# Example training loop
target = torch.randn(1, output_features)

for epoch in range(100):
    optimizer.zero_grad()
    
    # Forward pass
    output = sparse_linear_layer(input_tensor)
    
    # Compute loss
    loss = loss_fn(output.t(), target)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# After training, the sparse_values will have been updated
print("Updated Sparse Values:", sparse_linear_layer.sparse_values)

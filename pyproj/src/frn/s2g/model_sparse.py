from typing import List
import numpy as np
import torch
import torch.nn as nn
from torch_sparse import SparseTensor
from frn.utils.uni import get_map_location


class SparseLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, fixed_indices: List[List[int]]):
        """
        fixed_indices = torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long).t()
        """
        super(SparseLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fixed_indices = torch.tensor(fixed_indices, dtype=torch.long, device=get_map_location()).t()
        self.sparse_values = nn.Parameter(torch.randn(self.fixed_indices.shape[1], device=get_map_location()))

    def forward(self, x):
        # Create a sparse tensor from the fixed indices and trainable values
        sparse_weight = SparseTensor(row=self.fixed_indices[0], col=self.fixed_indices[1], value=self.sparse_values,
                                     sparse_sizes=(self.output_dim, self.input_dim), trust_data=True)
        # Perform the sparse matrix multiplication
        out = sparse_weight @ x.t()
        out = out.t()
        return out


def get_param4sparse(blocks_gt: List[List[int]], len_one_hot_vec: int):
    n_gt = len(np.unique(np.concatenate(blocks_gt)))
    input_dim = n_gt * len_one_hot_vec
    output_dim = len(blocks_gt)

    # Index
    idx_axis_0 = []
    idx_axis_1 = []
    for i_b in range(output_dim):
        x_r = sorted([(i * len_one_hot_vec + nx) for nx in range(len_one_hot_vec) for i in blocks_gt[i_b]])
        idx_axis_0.extend(x_r)
        idx_axis_1.extend([i_b] * len(x_r))
    assert len(idx_axis_0) == len(idx_axis_1)
    indices_ = []
    for i in range(len(idx_axis_0)):
        indices_.append([idx_axis_0[i], idx_axis_1[i]])
    
    return indices_, input_dim, output_dim


class SNPReductionNetModel(nn.Module):
    def __init__(
            self,
            output_dim: int,
            blocks_gt: List[List[int]],
            len_one_hot_vec: int,
            dense_layers_hidden_dims: List[int],
        ):
        super().__init__()
        n_blocks = len(blocks_gt)
        self.n_blocks = n_blocks
        
        s_index, s_input_dim, s_output_dim = get_param4sparse(blocks_gt, len_one_hot_vec)
        self.sparse_layer = SparseLinear(s_input_dim, s_output_dim, s_index)

        # self.sparse_layers = nn.ModuleList()
        # # Define the sparse linear layers that maps SNPs to genome blocks
        # for i_gb in range(n_blocks):
        #     self.sparse_layers.append(nn.Linear(len(blocks_gt[i_gb]) * len_one_hot_vec, 1, bias=False))
        
        # indices_gt = []
        # for i_gb in range(n_blocks):
        #     indices_gt.append(idx_convert(blocks_gt[i_gb], len_one_hot_vec))
        # self.indices_gt = indices_gt

        # Define the dense layers for predicting the phenotype
        self.dense_layers = nn.ModuleList()
        
        # Apply LayerNorm to the input features.
        self.dense_layers.append(nn.LayerNorm(n_blocks))
        
        # First dense layer takes the genome blocks features as input.
        self.dense_layers.append(nn.Linear(n_blocks, dense_layers_hidden_dims[0]))
        for i_dim in range(len(dense_layers_hidden_dims) - 1):
            self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[i_dim], dense_layers_hidden_dims[i_dim + 1]))
            self.dense_layers.append(nn.Sigmoid())
            # self.dense_layers.append(nn.Dropout(p=0.1))
        self.dense_layers.append(nn.Linear(dense_layers_hidden_dims[-1], output_dim))
        # if output_dim > 1:
        #     self.dense_layers.append(nn.Softmax(dim=1))
    
    def forward(self, x):
        # Map SNPs to genome features
        # g_features: list[torch.Tensor] = []
        # for i_gb in range(self.n_blocks):
        #     g_features.append(self.sparse_layers[i_gb](x[:, self.indices_gt[i_gb]]))
        
        # gblocks = torch.cat(g_features, dim=1)

        gblocks = self.sparse_layer(x)
        
        # Predict phenotype based on the low-dimensional features
        for layer in self.dense_layers:
            gblocks = layer(gblocks)
        
        # Return predicted phenotype(s)
        return gblocks#.type(torch.float32)

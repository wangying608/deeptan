r"""
Self-defined improved GAT Layer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_sparse


class GAT1H(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float = 0.4,
            negative_slope: float = 0.1,
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.attn = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.attn.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(negative_slope)

    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        N = h.shape[0]
        Wh = torch.mm(h, self.W)
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        all_combn = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        _e = torch.matmul(self.leakyrelu(all_combn), self.attn).view(N, N, -1).squeeze(2)

        zero_vec = -9e15 * torch.ones_like(_e)
        attention = torch.where(adj > 0, _e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh)
        return h_prime


class GATLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            n_heads: int = 4,
            dropout: float = 0.4,
            leaky_relu_slope: float = 0.1,
        ):
        super(GATLayer, self).__init__()
        # self.in_features = in_features
        # self.out_features = out_features
        # self.n_heads = n_heads
        # self.dropout = dropout
        # self.leaky_relu_slope = leaky_relu_slope

        self.attentions = nn.ModuleList([GAT1H(in_features, out_features, dropout, leaky_relu_slope) for _ in range(n_heads)])

    def forward(self, h, adj):
        # h = F.dropout(h, self.dropout, training=self.training)
        # Apply mean aggregation across heads instead of concatenation
        outputs = [att(h, adj) for att in self.attentions]
        output = torch.mean(torch.stack(outputs), dim=0)
        return output

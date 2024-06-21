import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

class PoodedGraphAttentionLayer(nn.Module):
    def __init__(self, num_node, in_features, out_features, dropout, concat=True):
        super(PoodedGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.concat = concat
        self.Mish = Mish()

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.adj = nn.Parameter(torch.zeros(size=(num_node, num_node)))
        nn.init.xavier_uniform_(self.adj, gain=1.414)

        # self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, inp):
        '''
        inp: input_features [N, in_features], in_features is feature vector of each node
        '''
        h = torch.mm(inp, self.W)
        N = h.size()[0]

        # [N, N, 2*out_features]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*self.out_features)
        # [N, N, 1] => [N, N]
        e = self.Mish(torch.matmul(a_input, self.a).squeeze(2))

        # zero_vec = -9e15*torch.ones_like(e)
        attention = torch.matmul(e, self.adj)
        attention = self.Mish(attention)
        edge_global_average_pooling = torch.mean(e)
        attention *= edge_global_average_pooling
        # attention = torch.where(adj > 0, e, zero_vec)

        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return self.Mish(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class MultiHeadPooledGAT(nn.Module):
    def __init__(self, n_node, n_feat, n_hid, n_class, dropout, n_heads):
        super(MultiHeadPooledGAT, self).__init__()
        self.dropout = dropout
        self.Mish = Mish()

        # MutilHeadGAT layer
        self.attentions = [PoodedGraphAttentionLayer(n_node, n_feat, n_hid, dropout, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = PoodedGraphAttentionLayer(n_node, n_hid*n_heads, n_class, dropout, concat=False)

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Mish(self.out_att(x))
        x = F.log_softmax(x, dim=1)
        return x

if __name__ == '__main__':
    in_features = 1
    out_features = 1
    nb_node = 200
    n_hid = 5
    dropout = 0.1
    n_heads = 2

    # in_features = 101
    # out_features = 7
    # nb_node = 97
    # n_hid = 5
    # dropout = 0.1
    # n_heads = 2

    input = torch.rand(nb_node, in_features)
    # adj = torch.randint(2, (3, 3))

    MGAT = MultiHeadPooledGAT(nb_node, in_features, n_hid, out_features, dropout, n_heads)
    output = MGAT(input)
    print(output)
    print(output.shape)

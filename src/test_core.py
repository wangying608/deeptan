import torch
from frn.graph.core import MSGP


if __name__ == "__main__":
    h = torch.randn(743, 1)
    adj = torch.triu(torch.randn(743, 743))
    adj = adj + adj.T
    adj = torch.pow(adj, 2)
    # Set values to 0 and 1.
    adj = adj.clamp(min=0, max=1)
    # If the value is smaller than 0.1, set it to 0.
    adj = adj.where(adj < 0.7, torch.zeros_like(adj))

    print(f"Shape of h: {h.shape}")
    print(f"Shape of adj: {adj.shape}")
    print(f"Head of adj:\n{adj[:4, :4]}")

    _msgp = MSGP(
        input_dim=1,
        output_dims_nd=[8, 16],
        output_dim_g_emb=32,
        n_heads=[2, 3],
        n_hop=1,
        threshold_subgraph_overlap=0.98,
        dropout=0.3,
        negative_slope=0.2,
        use_all_subgraphs=False,
    )

    g_emb = _msgp(h, adj)

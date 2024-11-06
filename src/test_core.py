import torch
from frn.graph.core import MSGP, GData


if __name__ == "__main__":
    num_nodes = 2003

    h = torch.randn(num_nodes, 1)
    adj = torch.triu(torch.randn(num_nodes, num_nodes))
    adj = adj + adj.T
    adj = torch.pow(adj, 2)
    
    # Simulate sparse connectivity.
    adj = adj.where(adj < 0.7, torch.zeros_like(adj)).to_sparse(layout=torch.sparse_coo)

    print(f"Shape of h: {h.shape}")
    print(f"Shape of adj: {adj.shape}")
    # print(f"Head of adj:\n{adj[:4, :4]}")

    g = GData(x=h, edge_index=adj.indices(), edge_attr=adj.values())

    _msgp = MSGP(
        input_dim=1,
        output_dims_nd=[8, 32, 128],
        output_dim_g_emb=256,
        n_heads=[2, 3],
        n_hop=2,
        threshold_subgraph_overlap=0.98,
        dropout=0.2,
        negative_slope=0.2,
        use_all_subgraphs=False,
    )

    g_emb = _msgp(g)

from deeptan.graph.model import MSGPMTL
from deeptan.utils.data import random_datamodule
from deeptan.utils.uni import train_model


if __name__ == '__main__':
    init_g_node_dim = 1
    label_dim = 1

    datamodule = random_datamodule(300, init_g_node_dim, 1003, 1)

    model = MSGPMTL(
        input_dim=init_g_node_dim,
        output_dim=label_dim,
        is_regression=True,
        output_dims_nd=[32, 64],
        output_dim_g_emb=256,
        n_hop=1,
        threshold_edge_exist=0.25,
        threshold_subgraph_overlap=0.9,
        n_heads=[4, 4],
        dropout=0.1,
        lr=1e-3,
        negative_slope=0.2,
    )

    train_model(
        model=model,
        datamodule=datamodule,
        es_patience=10,
        max_epochs=100,
        min_epochs=50,
        log_dir='./logs/',
        # accelerator='cpu',
    )

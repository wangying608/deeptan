
# from torch_geometric.data.lightning import LightningDataset as PyG_LightningDataset
from frn.utils.uni import time_string, train_model
from frn.graph.pipeline_general import MyGAT
from frn.graph.dataset import _tmp_GraphDataModule_MUTAG


if __name__ == "__main__":
    
    # Init Lightning datamodule
    datamodule = _tmp_GraphDataModule_MUTAG()
    datamodule.setup()

    _model = MyGAT(
        in_channels=datamodule.dim_input,
        graph_label_dim=datamodule.dim_output,
        regression=False,
    )

    loss_min = train_model(
        model=_model,
        datamodule=datamodule,
        es_patience=15,
        max_epochs=1000,
        min_epochs=10,
        log_dir=f".tmp/runs/{time_string()}",
        in_dev=False,
    )

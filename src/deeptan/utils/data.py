r"""
Graph data module.
"""

import os
from typing import List
from pathlib import Path
import mudata
import numpy as np
import polars as pl
import scanpy as sc
import anndata
import torch
from torch_geometric.data import Data as GData
from torch_geometric.data import Dataset as GDataset
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.utils import erdos_renyi_graph
from lightning import LightningDataModule
from deeptan.utils.uni import get_avail_cpu_count, get_map_location


def generate_random_graph(
    num_nodes: int, num_features: int, num_classes: int | None, is_regression: bool
) -> GData:
    """
    Generate a random graph data object with graph-level labels.

    Args:
        num_nodes (int): Number of nodes in the graph.
        num_features (int): Feature dimension of each node.
        num_classes (int): Number of classes (for classification) or output dimension (for regression).
        is_regression (bool): Whether the task is regression.

    Returns:
        Data: Randomly generated graph data object.
    """
    # Randomly generate node features
    x = torch.randn(
        num_nodes, num_features
    )  # Node feature matrix (num_nodes, num_features)

    # Randomly generate edge indices (using Erdős-Rényi model to generate a random graph)
    edge_index = erdos_renyi_graph(
        num_nodes, edge_prob=0.2
    )  # Edge indices (2, num_edges)

    # Randomly generate edge attributes
    edge_attr = torch.rand(
        edge_index.size(1), 1
    )  # Edge attribute matrix (num_edges, 1)

    # Randomly generate node names (assuming node names are strings)
    node_names = [f"node_{i}" for i in range(num_nodes)]

    # Remove edges with weights less than the threshold 0.2
    mask = edge_attr.squeeze() > 0.2
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[
        mask
    ]  # Filtered edge attribute matrix (num_filtered_edges, 1)

    if num_classes is None:
        # Create the graph data object
        graph_data = GData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_names=node_names,
        )
    else:
        # Randomly generate graph-level labels
        if is_regression:
            y = torch.rand(
                1, num_classes
            )  # Regression task labels (1, output_dim), representing the entire graph
        else:
            y = torch.randint(
                0, num_classes, (1,)
            )  # Classification task labels (1,), representing the entire graph

        # Create the graph data object
        graph_data = GData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            node_names=node_names,
        )

    return graph_data


class RandomGraphDataset(GDataset):
    def __init__(
        self,
        num_graphs: int = 100,
        num_nodes_max: int = 100,
        node_dim: int = 16,
        num_label_classes: int | None = 10,
        is_regression: bool = False,
    ):
        self.num_graphs = num_graphs
        self.num_nodes_max = num_nodes_max
        self.node_dim = node_dim
        self.num_label_classes = num_label_classes
        self.is_regression = is_regression
        super().__init__()

    def len(self):
        return self.num_graphs

    def get(self, idx):
        return generate_random_graph(
            self.num_nodes_max,
            self.node_dim,
            self.num_label_classes,
            self.is_regression,
        )


class GraphDataModule(LightningDataModule):
    def __init__(
        self,
        train_dataset: GDataset,
        val_dataset: GDataset,
        test_dataset: GDataset,
        batch_size: int = 4,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return GDataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return GDataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return GDataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


def random_g_datamodule(
    num_graphs: int,
    node_dim: int,
    num_nodes_max: int = 1000,
    num_label_classes: int | None = 10,
    is_regression: bool = False,
    batch_size: int = 4,
):
    train_dataset = RandomGraphDataset(
        num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression
    )
    val_dataset = RandomGraphDataset(
        num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression
    )
    test_dataset = RandomGraphDataset(
        num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression
    )
    # pred_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression)

    ltn_dm = GraphDataModule(train_dataset, val_dataset, test_dataset, batch_size)
    return ltn_dm


def read_h5ad(h5ad_file: str) -> anndata.AnnData:
    r"""Read h5ad file and return AnnData object.

    Args:
        h5ad_file (str): Path to h5ad file.

    Returns:
        anndata.AnnData: AnnData object.
    """
    adata = sc.read_h5ad(h5ad_file)
    return adata


def read_h5mu(h5mu_file: str):
    r"""Read h5mu file and return AnnData object.
    Args:
        h5mu_file (str): Path to h5mu file.
    Returns:
        anndata.AnnData: AnnData object.
    """
    adata = mudata.read_h5mu(Path(h5mu_file))
    print(adata)
    return adata


def adata_to_parquet(
    adata: anndata.AnnData,
    output_dir: str,
    output_prefix: str,
    randomly_select_features: int | None = None,
) -> None:
    r"""Save AnnData object to Parquet files.
    Args:
        adata (anndata.AnnData): AnnData object.
        output_dir (str): Output directory.
        output_prefix (str): Output prefix.
    """
    X = adata.X.toarray()
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy array.")
    print(f"X shape: {X.shape}")

    os.makedirs(output_dir, exist_ok=True)

    # Create a Polars DataFrame with obs_names and var_names.
    obs_names = adata.obs_names.astype(str).to_list()
    var_names = adata.var_names.astype(str).to_list()
    if randomly_select_features is not None:
        rands = np.random.choice(X.shape[1], randomly_select_features, replace=False)
        var_names = [var_names[i] for i in rands.tolist()]
        X = X[:, rands]
    df = pl.DataFrame({"obs_names": obs_names}).hstack(
        pl.DataFrame(X, schema=var_names)
    )
    print(f"DataFrame shape: {df.shape}")
    print(f"Head of DataFrame:\n{df.head()}\n")

    # Check number of None values
    print(f"Number of None values:\n{df.null_count().sum_horizontal()}\n")

    df.write_parquet(os.path.join(output_dir, f"{output_prefix}.parquet"))


def h5ad_to_parquet(input_dir: str, output_dir: str):
    r"""Read h5ad files and save them to Parquet files.
    Args:
        input_dir (str): Input directory.
        output_dir (str): Output directory.
    """
    h5ad_files = [f for f in os.listdir(input_dir) if f.endswith(".h5ad")]
    for h5ad_file in h5ad_files:
        adata = read_h5ad(os.path.join(input_dir, h5ad_file))
        adata_to_parquet(adata, output_dir, h5ad_file)


def h5_to_parquet(h5_file: str, output_parquet: str):
    r"""Read a 10X Genomics H5 file and save it to a Parquet file.
    Args:
        h5_file (str): Path to the H5 file.
        output_parquet (str): Path to the output Parquet file.
    """
    adata = sc.read_10x_h5(h5_file)
    print(f"Read {h5_file} with {adata.shape[0]} cells and {adata.shape[1]} features.")
    print(f"Saving to {output_parquet}...")

    adata.var_names_make_unique(join="_")
    adata.obs_names_make_unique(join="_")
    adata_to_parquet(
        adata, os.path.dirname(output_parquet), os.path.basename(output_parquet)
    )


def h5mu_to_parquet(h5mu_file: str, output_parquet: str):
    r"""Read single-cell multi-modal data from an H5MU file and save it to a Parquet file.
    Args:
        h5mu_file (str): Path to the H5MU file.
        output_parquet (str): Path to the output Parquet file.
    """
    # Read the H5MU file using mudata
    mdata = mudata.read_h5mu(Path(h5mu_file))

    adata_rna = mdata.mod["rna"]
    adata_atac = mdata.mod["atac"]
    # Concatenate RNA and ATAC data into a single AnnData object
    adata_combined = anndata.concat([adata_rna, adata_atac], axis=1, join="outer")

    adata_combined.obs_names_make_unique(join="_")
    adata_combined.var_names_make_unique(join="_")

    adata_to_parquet(
        adata_combined,
        os.path.dirname(output_parquet),
        os.path.basename(output_parquet),
    )


def split_parquet(
    parquet_file: str, output_dir: str, ratio: List[float], seeds: List[int]
):
    r"""
    Read a dataframe from a parquet file and split it into multiple parts based on the given ratios.
    The splits are saved as separate parquet files in the specified output directory.
    The length of seeds is the repetition of the split process.
    """
    df = pl.read_parquet(parquet_file)
    assert len(seeds) > 0, "Seeds list must not be empty."
    assert sum(ratio) == 1, "Ratios must sum to 1."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for seed in seeds:
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(df))
        split_indices = [int(r * len(df)) for r in np.cumsum(ratio)]
        df_i = df[shuffled_indices]
        for i, (start, end) in enumerate(zip([0] + split_indices, split_indices)):
            split_df = df_i[start:end]
            print(f"Split {i} with seed {seed} has {len(split_df)} rows")
            output_file = os.path.join(output_dir, f"split_seed_{seed}_{i}.parquet")
            split_df.write_parquet(output_file)
            print(f"Saved split {i} with seed {seed} to {output_file}")


def read_nmic_results(npz_path: str):
    data = np.load(npz_path)

    # Print keys
    print("\nKeys in the npz file:", data.files)
    # Keys in the npz file: ['mi_values', 'feat_pairs', 'processed_mat', 'mat_feat_indices', 'mat_simi_feat_pairs', 'thre_cv', 'thre_pcc', 'thre_mi', 'ratio_max_window', 'ratio_min_window', 'ratio_step_window', 'ratio_step_sliding']

    # Access the data
    print(f"\nThreshold of MCV: {data['thre_cv']}")
    print(f"Threshold of NMIC: {data['thre_mi']}")
    print(f"processed_mat shape: {data['processed_mat'].shape}")
    print(f"mat_feat_indices shape: {data['mat_feat_indices'].shape}")
    print(f"mat_feat_indices max: {data['mat_feat_indices'].max()}")
    print(f"mi_values shape: {data['mi_values'].shape}")
    print(f"feat_pairs shape: {data['feat_pairs'].shape}")
    print(f"feat_pairs min: {data['feat_pairs'].min()}")
    print(f"feat_pairs max: {data['feat_pairs'].max()}")

    print("\nProcessed matrix:")
    df = pl.read_parquet(npz_path.replace(".npz", ".parquet"))
    print(df, "\n")
    print(df.columns)
    # The first column is obs_names, other columns are features.


class NMICGraphDataset(GDataset):
    def __init__(self, npz_path: str, device: str):
        """
        Initialize the NMIC graph dataset.

        Args:
            npz_path: Path to the .npz file containing NMIC results.
            device: Device to store the graph data on.
        """
        super().__init__()
        self.device = device
        (
            self.edge_attr,
            self.edge_index,
            self.mat,
            self.mat_feat_indices,
            self.obs_names,
            self.node_names,
        ) = self.read_nmic_results(npz_path)

    def len(self):
        return len(self.obs_names)

    def get(self, idx):
        values: np.ndarray = self.mat[idx]
        avail_col_indices = np.where(values > 0)[0]
        avail_feat_indices = self.mat_feat_indices[avail_col_indices]

        x = torch.tensor(
            values[avail_col_indices], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        # Filter edges based on available nodes
        edge_mask = np.isin(self.edge_index[0], avail_feat_indices) & np.isin(
            self.edge_index[1], avail_feat_indices
        )

        edge_indices = self.edge_index[:, edge_mask]

        # Map edge indices to current feature indices
        # Create a mapping from original feature indices to current indices
        feat_index_to_avail_index = {
            feat_idx: i for i, feat_idx in enumerate(avail_feat_indices)
        }
        # Apply the mapping to edge_indices
        mapped_edge_indices = edge_indices.copy()
        for i in range(edge_indices.shape[1]):
            mapped_edge_indices[0, i] = feat_index_to_avail_index[edge_indices[0, i]]
            mapped_edge_indices[1, i] = feat_index_to_avail_index[edge_indices[1, i]]
        edge_index = torch.tensor(
            mapped_edge_indices, dtype=torch.long, device=self.device
        )

        edge_attrs = torch.tensor(
            self.edge_attr[edge_mask], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        node_names = [self.node_names[i] for i in avail_col_indices]

        # Create the graph data object
        graph_data = GData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            node_names=node_names,
        )

        return graph_data

    def read_nmic_results(self, npz_path: str):
        r"""
        Read NMIC results from a .npz file and convert them into a graph data.
        """
        # Load the NMIC results
        results = np.load(npz_path)
        df = pl.read_parquet(npz_path.replace(".npz", ".parquet"))

        # Extract relevant data
        edge_attr: np.ndarray = results["mi_values"]
        edge_index: np.ndarray = results["feat_pairs"].T
        mat: np.ndarray = results["processed_mat"].T
        mat_feat_indices: np.ndarray = results["mat_feat_indices"]

        # Extract node features (assuming the first column is obs_names)
        obs_names: List[str] = df.select("obs_names").to_series().to_list()
        node_names: List[str] = df.columns[1:]

        return edge_attr, edge_index, mat, mat_feat_indices, obs_names, node_names


class NMICGraphDatasetRely(GDataset):
    def __init__(self, parquet_path: str, depGDataset: NMICGraphDataset, device: str):
        super().__init__()
        self.device = device
        self.depGDataset = depGDataset
        df = pl.read_parquet(parquet_path)
        # Extract relevant data
        self.selected_mat = df.select(depGDataset.node_names).to_numpy()

    def len(self):
        return self.selected_mat.shape[0]

    def get(self, idx):
        values = self.selected_mat[idx]
        avail_col_indices = np.where(values > 0)[0]

        x = torch.tensor(
            values[avail_col_indices], dtype=torch.float32, device=self.device
        ).unsqueeze(1)

        avail_feat_indices = self.depGDataset.mat_feat_indices[avail_col_indices]
        # Filter edges based on available nodes
        edge_mask = np.isin(
            self.depGDataset.edge_index[0], avail_feat_indices
        ) & np.isin(self.depGDataset.edge_index[1], avail_feat_indices)

        edge_indices = self.depGDataset.edge_index[:, edge_mask]

        # Map edge indices to current feature indices
        # Create a mapping from original feature indices to current indices
        feat_index_to_avail_index = {
            feat_idx: i for i, feat_idx in enumerate(avail_feat_indices)
        }
        # Apply the mapping to edge_indices
        mapped_edge_indices = edge_indices.copy()
        for i in range(edge_indices.shape[1]):
            mapped_edge_indices[0, i] = feat_index_to_avail_index[edge_indices[0, i]]
            mapped_edge_indices[1, i] = feat_index_to_avail_index[edge_indices[1, i]]
        edge_index = torch.tensor(
            mapped_edge_indices, dtype=torch.long, device=self.device
        )

        edge_attrs = torch.tensor(
            self.depGDataset.edge_attr[edge_mask],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        node_names = [self.depGDataset.node_names[i] for i in avail_col_indices]

        # Create the graph data object
        graph_data = GData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            node_names=node_names,
        )

        return graph_data


class DeepTANDataModule(LightningDataModule):
    def __init__(
        self,
        files: dict[str, str],
        batch_size: int,
        num_workers: int | None = None,
        device: str | None = None,
    ):
        super().__init__()
        if files.keys() != {"trn", "val", "tst"}:
            raise ValueError("files must contain 'trn', 'val', and 'tst' keys")
        if not files["trn"].endswith(".npz"):
            raise ValueError("files['trn'] must be a .npz file")
        self.files = files
        self.batch_size = batch_size
        self.num_workers = (
            get_avail_cpu_count(num_workers) if num_workers else get_avail_cpu_count(28)
        )
        self.device = get_map_location(device)

    def setup(self, stage=None):
        self.train = NMICGraphDataset(self.files["trn"], self.device)
        self.val = NMICGraphDatasetRely(self.files["val"], self.train, self.device)
        self.test = NMICGraphDatasetRely(self.files["tst"], self.train, self.device)
        dict_node_names_values = [i for i in range(len(self.train.node_names))]
        self.dict_node_names = dict(zip(self.train.node_names, dict_node_names_values))

    def train_dataloader(self):
        return GDataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return GDataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return GDataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
        )


# if __name__ == "__main__":
#     # Read npz file
#     path_npz = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/GSE235510_control_split/nmic_g/split_seed_47_0.parquet.npz"
#     # read_nmic_results(path_npz)
#     g_dataset = NMICGraphDataset(path_npz, "cuda")

#     print(len(g_dataset))
#     print(g_dataset[666])
#     print(g_dataset[4999])

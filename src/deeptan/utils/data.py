r"""
Graph data module.
"""

import os
from typing import List, Optional, Union

import anndata
import numpy as np
import polars as pl
import scanpy as sc
import torch
from lightning import LightningDataModule
from litdata import StreamingDataLoader, StreamingDataset
from torch_geometric.data import Data as GData
from torch_geometric.data import Dataset as GDataset
from torch_geometric.loader import DataLoader as GDataLoader
from torch_geometric.utils import erdos_renyi_graph

from deeptan.utils.uni import collate_fn, get_avail_cpu_count


def read_csv_celltypes2onehot(csv_path):
    df = pl.read_csv(csv_path)
    df.columns = ["bc", "ct"]

    # One-hot encode the 'ct' column
    df_one_hot = df_one_hot = df.to_dummies(columns=["ct"])

    # Add a new column "ct_unknown" which all values are 0 (type: u8)
    col_unk = pl.Series("ct_unknown", [0] * len(df_one_hot), dtype=pl.UInt8)
    df_one_hot = df_one_hot.with_columns(col_unk)

    # SORT
    df_one_hot = df_one_hot.sort("bc")

    # Save the one-hot encoded DataFrame
    df_one_hot.write_parquet(csv_path.replace(".csv", "_celltypes_onehot.parquet"))


def read_h5ad_celltypes2onehot(h5ad_path, celltype_key="Celltype"):
    celltypes = sc.read_h5ad(h5ad_path).obs[celltype_key]
    print(celltypes.value_counts())
    celltypes_pl = pl.DataFrame({"bc": celltypes.index, "ct": celltypes.values.astype(str)})
    celltypes_onehot = celltypes_pl.to_dummies(columns=["ct"])

    # Add a new column "ct_unknown" which all values are 0 (type: u8)
    col_unk = pl.Series("ct_unknown", [0] * len(celltypes_onehot), dtype=pl.UInt8)
    celltypes_onehot = celltypes_onehot.with_columns(col_unk)

    # SORT
    celltypes_onehot = celltypes_onehot.sort("bc")

    # Save the one-hot encoded DataFrame
    celltypes_onehot.write_parquet(h5ad_path.replace(".h5ad", "_celltypes_onehot.parquet"))


def celltypes_class_weights(df_onehot: pl.DataFrame) -> List[float]:
    r"""
    Compute class weights for weighted cross entropy loss.
    """
    class_weights = df_onehot.select(pl.exclude("bc")).sum().to_numpy().flatten()
    class_weights[-1] = class_weights.mean()

    class_weights = 1 / (class_weights / class_weights.sum())

    class_weights = class_weights / class_weights.sum()
    output = class_weights.tolist()

    return output


def read_nmic_npz(npz_path: str):
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


class NMICGraphDataset(GDataset):
    def __init__(
        self,
        npz_path: str,
        labels: str | None,
        edge_attr_threshold: float = 0.1,
        specify_features: Optional[List[str]] = None,
        if_log1p: bool = True,
    ):
        """
        Initialize the NMIC graph dataset.

        Args:
            npz_path: Path to the .npz file containing NMIC results.
            labels: Path to the obs labels.
            edge_attr_threshold: Threshold for edge attributes.
            specify_features: List of features to specify. If None, all features are used.
            if_log1p: Whether to apply log1p transformation to the data.
        """
        super().__init__()
        (
            self.edge_attr,
            self.edge_index,
            self.mat,
            self.mat_feat_indices,
            self.obs_names,
            self.node_names,
        ) = read_nmic_npz(npz_path)

        # Check if obs_names and node_names are unique
        if len(set(self.obs_names)) != len(self.obs_names):
            raise ValueError("obs_names must be unique")
        if len(set(self.node_names)) != len(self.node_names):
            raise ValueError("node_names must be unique")

        if if_log1p:
            self.mat = np.log1p(self.mat)

        if labels is None:
            self.labels = None
            self.label_dim = None
        else:
            self.labels = pl.read_parquet(labels)
            if "obs_names" in self.labels.columns:
                self.labels = self.labels.rename({"obs_names": "bc"})
            self.label_dim = self.labels.shape[1] - 1

        self.edge_attr_threshold = edge_attr_threshold
        self.specify_features = specify_features

        if self.specify_features is not None:
            # Interact with self.node_names using set
            self.node_names_for_dict = list(set(self.node_names).intersection(set(self.specify_features)))
            print(f"\nNumber of node names for dictionary after intersection: {len(self.node_names_for_dict)}")
        else:
            self.node_names_for_dict = self.node_names
            print(f"\nNumber of node names for dictionary: {len(self.node_names_for_dict)}")

    def len(self):
        return len(self.obs_names)

    def get(self, idx):
        values = self.mat[idx]
        avail_col_indices = np.where(np.abs(values) > 1e-6)[0]
        avail_feat_indices = self.mat_feat_indices[avail_col_indices]

        # Apply specify_features filter if provided
        if self.specify_features is not None:
            # Find indices of specified features
            specified_feat_indices = [self.mat_feat_indices[i] for i, _name in enumerate(self.node_names) if _name in self.node_names_for_dict]
            # Filter avail_feat_indices to only include specified features
            mask = np.isin(avail_feat_indices, specified_feat_indices)
            avail_col_indices = avail_col_indices[mask]
            avail_feat_indices = avail_feat_indices[mask]

        # If no features are available?
        if len(avail_col_indices) < 10:
            print("\nNumber of available features is too small.")

        # Filter edges based on available nodes
        edge_mask = np.logical_and(np.isin(self.edge_index[0], avail_feat_indices), np.isin(self.edge_index[1], avail_feat_indices))
        edge_indices = self.edge_index[:, edge_mask]
        edge_attrs = self.edge_attr[edge_mask]

        # Filter edges based on edge_attr threshold
        edge_attr_mask = edge_attrs > self.edge_attr_threshold
        edge_indices = edge_indices[:, edge_attr_mask]
        edge_attrs = edge_attrs[edge_attr_mask]

        # Filter nodes based on used edge indices
        if edge_indices.size > 0:
            used_feat_indices = np.unique(edge_indices.flatten())
            final_node_mask = np.isin(avail_feat_indices, used_feat_indices)
            final_col_indices = avail_col_indices[final_node_mask]
            final_feat_indices = avail_feat_indices[final_node_mask]
        else:
            # Handle no edge case: retain the first 10 features or raise an exception
            final_col_indices = avail_col_indices[:10]
            final_feat_indices = avail_feat_indices[:10]
            if len(final_col_indices) < 1:
                raise ValueError("No valid features after filtering")

        # Re-generate the feature matrix
        x = torch.tensor(values[final_col_indices], dtype=torch.float16).unsqueeze(1)

        # Map edge indices to current feature indices
        # Create a mapping from original feature indices to current indices
        feat_mapping = {feat: idx for idx, feat in enumerate(final_feat_indices)}
        if edge_indices.size > 0:
            mapped_edges = np.vectorize(lambda x: feat_mapping.get(x, -1))(edge_indices)
            valid_mask = (mapped_edges[0] != -1) & (mapped_edges[1] != -1)
            edge_index = torch.tensor(mapped_edges[:, valid_mask], dtype=torch.long)
            edge_attrs = torch.tensor(edge_attrs[valid_mask], dtype=torch.float16).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attrs = torch.tensor([], dtype=torch.float16).unsqueeze(1)

        # Reorder node names based on final column indices
        node_names = [self.node_names[i] for i in final_col_indices]

        # Create the graph data object
        _y = None
        if self.labels is not None:
            _y = torch.tensor(self.pick_label(self.obs_names[idx]), dtype=torch.float16)

        return GData(
            x=x,
            y=_y,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            node_names=node_names,
        )

    def pick_label(self, obs_name: str):
        if self.labels is None:
            return None
        else:
            _label = self.labels.filter(pl.col("bc") == obs_name).drop("bc").to_numpy().astype(np.float16)
            return _label


class NMICGraphDatasetRely(GDataset):
    def __init__(
        self,
        parquet_path: str,
        depGDataset: NMICGraphDataset,
        if_log1p: bool = True,
    ):
        super().__init__()
        self.depGDataset = depGDataset
        df = pl.read_parquet(parquet_path)
        self.obs_names = df[df.columns[0]].to_list()
        # Extract relevant data
        self.selected_mat = df.select(depGDataset.node_names).to_numpy()
        if if_log1p:
            self.selected_mat = np.log1p(self.selected_mat)

    def len(self):
        return self.selected_mat.shape[0]

    def get(self, idx):
        values = self.selected_mat[idx]
        avail_col_indices = np.where(np.abs(values) > 1e-6)[0]
        avail_feat_indices = self.depGDataset.mat_feat_indices[avail_col_indices]

        # Apply specify_features filter if provided
        if self.depGDataset.specify_features is not None:
            # Find indices of specified features
            specified_feat_indices = [self.depGDataset.mat_feat_indices[i] for i, _name in enumerate(self.depGDataset.node_names) if _name in self.depGDataset.node_names_for_dict]
            # Filter avail_feat_indices to only include specified features
            mask = np.isin(avail_feat_indices, specified_feat_indices)
            avail_col_indices = avail_col_indices[mask]
            avail_feat_indices = avail_feat_indices[mask]

        # Filter edges based on available nodes
        edge_mask = np.isin(self.depGDataset.edge_index[0], avail_feat_indices) & np.isin(self.depGDataset.edge_index[1], avail_feat_indices)
        edge_indices = self.depGDataset.edge_index[:, edge_mask]
        edge_attrs = self.depGDataset.edge_attr[edge_mask]

        # Filter edges based on edge_attr threshold
        edge_attr_mask = edge_attrs > self.depGDataset.edge_attr_threshold
        edge_indices = edge_indices[:, edge_attr_mask]
        edge_attrs = edge_attrs[edge_attr_mask]

        # Filter nodes based on used edge indices
        if edge_indices.size > 0:
            used_feat_indices = np.unique(edge_indices.flatten())
            final_node_mask = np.isin(avail_feat_indices, used_feat_indices)
            final_col_indices = avail_col_indices[final_node_mask]
            final_feat_indices = avail_feat_indices[final_node_mask]
        else:
            # Handle no edge case: retain the first 10 features or raise an exception
            final_col_indices = avail_col_indices[:10]
            final_feat_indices = avail_feat_indices[:10]
            if len(final_col_indices) < 1:
                raise ValueError("No valid features after filtering")

        # Re-generate the feature matrix
        x = torch.tensor(values[final_col_indices], dtype=torch.float16).unsqueeze(1)

        # Map edge indices to current feature indices
        # Create a mapping from original feature indices to current indices
        feat_mapping = {feat: idx for idx, feat in enumerate(final_feat_indices)}
        if edge_indices.size > 0:
            mapped_edges = np.vectorize(lambda x: feat_mapping.get(x, -1))(edge_indices)
            valid_mask = (mapped_edges[0] != -1) & (mapped_edges[1] != -1)
            edge_index = torch.tensor(mapped_edges[:, valid_mask], dtype=torch.long)
            edge_attrs = torch.tensor(edge_attrs[valid_mask], dtype=torch.float16).unsqueeze(1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attrs = torch.tensor([], dtype=torch.float16).unsqueeze(1)

        # Reorder node names based on final column indices
        node_names = [self.depGDataset.node_names[i] for i in final_col_indices]

        # Create the graph data object
        _y = None
        if self.depGDataset.labels is not None:
            _y = torch.tensor(self.depGDataset.pick_label(self.obs_names[idx]), dtype=torch.float16)

        return GData(
            x=x,
            y=_y,
            edge_index=edge_index,
            edge_attr=edge_attrs,
            node_names=node_names,
        )


class DeepTANDataModule(LightningDataModule):
    def __init__(
        self,
        files: dict[str, str],
        labels: str | None,
        batch_size: int = 1,
        edge_attr_threshold: float = 0.1,
        specify_features: Union[None, str, List[str]] = None,
        if_log1p: bool = True,
    ):
        super().__init__()
        if files.keys() != {"trn", "val", "tst"}:
            raise ValueError("files must contain 'trn', 'val', and 'tst' keys")
        if not files["trn"].endswith(".npz"):
            raise ValueError("files['trn'] must be a .npz file")
        self.files = files
        self.labels = labels
        self.batch_size = batch_size
        self.edge_attr_threshold = edge_attr_threshold
        self.if_log1p = if_log1p

        if isinstance(specify_features, str):
            if not specify_features.endswith(".csv"):
                raise ValueError("specify_features must be a .csv file")
            self.specify_features = pl.read_csv(specify_features, has_header=True, infer_schema=False).to_series().to_list()
        else:
            self.specify_features = specify_features

    def setup(self, stage=None):
        self.train = NMICGraphDataset(self.files["trn"], self.labels, self.edge_attr_threshold, self.specify_features, self.if_log1p)
        self.val = NMICGraphDatasetRely(self.files["val"], self.train, self.if_log1p)
        self.test = NMICGraphDatasetRely(self.files["tst"], self.train, self.if_log1p)
        dict_node_names_values = [i for i in range(len(self.train.node_names_for_dict))]
        self.dict_node_names = dict(zip(self.train.node_names_for_dict, dict_node_names_values))
        self.label_dim = self.train.label_dim

    def train_dataloader(self):
        return GDataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return GDataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return GDataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
        )


class DeepTANDataModuleLit(LightningDataModule):
    def __init__(
        self,
        litdata_dir: str,
        batch_size: int,
        n_workers: int | None = 1,
    ):
        r"""LightningDataModule for training.

        Args:
            litdata_dir: Directory containing the LitData for "trn", "val", and "tst".

            batch_size: Batch size for dataloader.

            n_workers: Number of workers for dataloader.
        """
        super().__init__()
        self.litdata_dir = litdata_dir
        self.batch_size = batch_size
        self.n_workers = get_avail_cpu_count(n_workers) if n_workers else get_avail_cpu_count(28)

    def setup(self, stage=None):
        self.dataloder_trn = StreamingDataLoader(
            StreamingDataset(os.path.join(self.litdata_dir, "trn"), max_cache_size="10GB"),
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
            pin_memory_device="cpu",
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.dataloader_val = StreamingDataLoader(
            StreamingDataset(os.path.join(self.litdata_dir, "val"), max_cache_size="10GB"),
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
            pin_memory=False,
            pin_memory_device="cpu",
            collate_fn=collate_fn,
        )
        self.dataloader_test = StreamingDataLoader(
            StreamingDataset(os.path.join(self.litdata_dir, "tst"), max_cache_size="10GB"),
            batch_size=self.batch_size,
            num_workers=self.n_workers,
            persistent_workers=True,
            pin_memory=False,
            pin_memory_device="cpu",
            collate_fn=collate_fn,
        )

    def train_dataloader(self):
        return self.dataloder_trn

    def val_dataloader(self):
        return self.dataloader_val

    def test_dataloader(self):
        return self.dataloader_test


def generate_random_graph(num_nodes: int, num_features: int, num_classes: int | None, is_regression: bool) -> GData:
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
    x = torch.randn(num_nodes, num_features)  # Node feature matrix (num_nodes, num_features)

    # Randomly generate edge indices (using Erdős-Rényi model to generate a random graph)
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.2)  # Edge indices (2, num_edges)

    # Randomly generate edge attributes
    edge_attr = torch.rand(edge_index.size(1), 1)  # Edge attribute matrix (num_edges, 1)

    # Randomly generate node names (assuming node names are strings)
    node_names = [f"node_{i}" for i in range(num_nodes)]

    # Remove edges with weights less than the threshold 0.2
    mask = edge_attr.squeeze() > 0.2
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]  # Filtered edge attribute matrix (num_filtered_edges, 1)

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
            y = torch.rand(1, num_classes)  # Regression task labels (1, output_dim), representing the entire graph
        else:
            y = torch.randint(0, num_classes, (1,))  # Classification task labels (1,), representing the entire graph

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
    train_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression)
    val_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression)
    test_dataset = RandomGraphDataset(num_graphs, num_nodes_max, node_dim, num_label_classes, is_regression)
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


# def read_h5mu(h5mu_file: str):
#     r"""Read h5mu file and return AnnData object.
#     Args:
#         h5mu_file (str): Path to h5mu file.
#     Returns:
#         anndata.AnnData: AnnData object.
#     """
#     adata = mudata.read_h5mu(Path(h5mu_file))
#     print(adata)
#     return adata


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
    df = pl.DataFrame({"obs_names": obs_names}).hstack(pl.DataFrame(X, schema=var_names))
    print(f"DataFrame shape: {df.shape}")
    print(f"Head of DataFrame:\n{df.head()}\n")

    # Check number of None values
    print(f"Number of None values:\n{df.null_count().sum_horizontal()}\n")
    if output_prefix.endswith(".parquet"):
        df.write_parquet(os.path.join(output_dir, output_prefix))
    else:
        df.write_parquet(os.path.join(output_dir, f"{output_prefix}.parquet"))


def h5ad_to_parquet_dir(input_dir: str, output_dir: str):
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
    adata_to_parquet(adata, os.path.dirname(output_parquet), os.path.basename(output_parquet))


def h5ad_to_parquet(h5ad_file: str, output_parquet: str, uniq_names: bool = True):
    r"""Read a h5ad file and save data to a Parquet file.
    Args:
        h5ad_file (str): Path to the h5ad file.
        output_parquet (str): Path to the output Parquet file.
    """
    adata = read_h5ad(h5ad_file)
    print(f"Read {h5ad_file} with {adata.shape[0]} cells and {adata.shape[1]} features.")
    print(f"Saving to {output_parquet}...")

    if uniq_names:
        adata.var_names_make_unique(join="_")
        adata.obs_names_make_unique(join="_")

    adata_to_parquet(adata, os.path.dirname(output_parquet), os.path.basename(output_parquet))


# def h5mu_to_parquet(h5mu_file: str, output_parquet: str):
#     r"""Read single-cell multi-modal data from an H5MU file and save it to a Parquet file.
#     Args:
#         h5mu_file (str): Path to the H5MU file.
#         output_parquet (str): Path to the output Parquet file.
#     """
#     # Read the H5MU file using mudata
#     mdata = mudata.read_h5mu(Path(h5mu_file))

#     adata_rna = mdata.mod["rna"]
#     adata_atac = mdata.mod["atac"]
#     # Concatenate RNA and ATAC data into a single AnnData object
#     adata_combined = anndata.concat([adata_rna, adata_atac], axis=1, join="outer")

#     adata_combined.obs_names_make_unique(join="_")
#     adata_combined.var_names_make_unique(join="_")

#     adata_to_parquet(
#         adata_combined,
#         os.path.dirname(output_parquet),
#         os.path.basename(output_parquet),
#     )


def split_parquet(parquet_file: str, output_dir: str, ratio: List[float], seeds: List[int]):
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
            output_file = os.path.join(output_dir, f"split_{seed}_{i}.parquet")
            split_df.write_parquet(output_file)
            print(f"Saved split {i} with seed {seed} to {output_file}")


def split_parquet_with_celltypes(
    cell_types: List[str],
    parquet_file: str,
    output_dir: str,
    ratio: List[float],
    seeds: List[int],
):
    df = pl.read_parquet(parquet_file)
    assert len(seeds) > 0, "Seeds list must not be empty."
    assert abs(sum(ratio) - 1.0) < 1e-9, "Ratios must sum to 1"
    assert len(cell_types) == len(df), "Cell types list must match the number of rows in the dataframe."
    os.makedirs(output_dir, exist_ok=True)

    # Create a list of unique cell types
    unique_cell_types = np.unique(cell_types)

    # Stratified Sampling based on cell types
    for seed in seeds:
        np.random.seed(seed)

        # Initialize an empty list to store indices for each split
        split_indices = [[] for _ in range(len(ratio))]

        # For each unique cell type, shuffle and split the indices
        for cell_type in unique_cell_types:
            # Get indices of rows corresponding to the current cell type
            cell_type_indices = np.where(np.array(cell_types) == cell_type)[0]
            # Shuffle the indices
            shuffled_indices = np.random.permutation(cell_type_indices)
            # Calculate the split points based on the ratio
            split_points = np.cumsum(np.array(ratio[:-1]) * len(shuffled_indices)).astype(int)
            cell_type_split_indices = np.split(shuffled_indices, split_points)

            # Append the split indices for each cell type to the overall split indices
            for i, indices in enumerate(cell_type_split_indices):
                split_indices[i].extend(indices)

        # Shuffle the indices within each split to avoid any ordering bias
        for i in range(len(split_indices)):
            np.random.shuffle(split_indices[i])

        # Create the splits based on the final split indices
        for i, indices in enumerate(split_indices):
            split_df = df[indices]
            print(f"Split {i} with seed {seed} has {len(split_df)} rows")
            output_file = os.path.join(output_dir, f"split_{seed}_{i}.parquet")
            split_df.write_parquet(output_file)
            print(f"Saved split {i} with seed {seed} to {output_file}")


class JointStratifiedSplitter:
    def __init__(
        self,
        cell_types: List[str],
        orig_idents: List[str],
        parquet_file: str,
        output_dir: str,
        ratio: List[float],
        seeds: List[int],
        balance_strategy: str = "none",
        retain_ratio: float = 1.0,
    ):
        """
        Split parquet data with balanced sampling across both cell types and orig.ident attributes.
        Ensures each split maintains similar distributions for both attributes using joint stratification.
        Supports oversampling, undersampling, or a combined strategy for balancing.

        Args:
            cell_types: List of cell types to split by.
            orig_idents: List of orig.ident values to split by.
            parquet_file: Path to the input Parquet file.
            output_dir: Directory to save the split Parquet files.
            ratio: List of ratios for each split.
            seeds: List of seeds for reproducibility.
            balance_strategy: Strategy for balancing the splits. Options are ["none", "oversample", "undersample", "combined"].
            retain_ratio: Ratio of data to retain after balancing. 0 < retain_ratio ≤ 1.0, Defaults to 1.0.
        """
        self.cell_types = cell_types
        self.orig_idents = orig_idents
        self.parquet_file = parquet_file
        self.output_dir = output_dir
        self.ratio = ratio
        self.seeds = seeds
        self.balance_strategy = balance_strategy
        self.retain_ratio = retain_ratio
        assert 0 < self.retain_ratio <= 1.0, "retain_ratio must be between 0 and 1"
        assert len(self.cell_types) == len(self.orig_idents), "Feature lengths mismatch"
        self.df = pl.read_parquet(self.parquet_file)
        assert len(self.cell_types) == len(self.df), "Cell types length mismatch"
        assert len(self.orig_idents) == len(self.df), "Orig.idents length mismatch"
        assert len(self.seeds) > 0, "Seeds list must not be empty"
        assert abs(sum(self.ratio) - 1.0) < 1e-9, "Ratios must sum to 1"
        assert self.balance_strategy in [
            "none",
            "oversample",
            "undersample",
            "combined",
        ], f"Invalid balance strategy: {self.balance_strategy}"

        # Precompute strata labels.
        strata = np.array(list(zip(self.cell_types, self.orig_idents)))
        self.unique_strata, self.stratum_labels = np.unique(strata, axis=0, return_inverse=True)

        os.makedirs(self.output_dir, exist_ok=True)

    def execute(self):
        """Excute the data splitting process."""

        for seed in self.seeds:
            if self.balance_strategy == "none":
                splits = self._none_strategy_split(seed)
            else:
                balanced_indices = self._apply_balance_strategy(seed)
                splits = self._split_balanced_data(balanced_indices)

            self._save_splits(splits, seed)

    def _apply_balance_strategy(self, seed: int) -> np.ndarray:
        """Balance the dataset using the specified strategy."""
        np.random.seed(seed)
        balanced_indices = []

        if self.balance_strategy == "oversample":
            self._oversample_strategy(balanced_indices)
        elif self.balance_strategy == "undersample":
            self._undersample_strategy(balanced_indices)
        elif self.balance_strategy == "combined":
            self._combined_strategy(balanced_indices)

        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices)
        return balanced_indices

    def _oversample_strategy(self, balanced_indices: list, seed: Optional[int] = None):
        """Oversample minority strata to match majority size."""
        stratum_counts = np.bincount(self.stratum_labels)
        max_stratum_size = np.max(stratum_counts)

        for stratum_idx in range(len(self.unique_strata)):
            stratum_seed = seed + stratum_idx if seed is not None else None
            stratum_indices = self._get_stratum_indices(stratum_idx, stratum_seed)
            repeat_factor = int(np.ceil(max_stratum_size / len(stratum_indices)))
            balanced_indices.extend(np.tile(stratum_indices, repeat_factor)[:max_stratum_size])

    def _undersample_strategy(self, balanced_indices: list, seed: Optional[int] = None):
        """Undersample majority strata to match minority size."""
        stratum_counts = np.bincount(self.stratum_labels)
        min_stratum_size = np.min(stratum_counts)

        for stratum_idx in range(len(self.unique_strata)):
            stratum_seed = seed + stratum_idx if seed is not None else None
            stratum_indices = self._get_stratum_indices(stratum_idx, stratum_seed)
            sampled = np.random.choice(stratum_indices, min_stratum_size, replace=False)
            balanced_indices.extend(sampled)

    def _combined_strategy(self, balanced_indices: list, seed: Optional[int] = None):
        """Hybrid: oversample small strata and undersample large ones."""
        stratum_counts = np.bincount(self.stratum_labels)
        target_stratum_size = int(np.mean(stratum_counts))

        for stratum_idx in range(len(self.unique_strata)):
            stratum_seed = seed + stratum_idx if seed is not None else None
            stratum_indices = self._get_stratum_indices(stratum_idx, stratum_seed)

            if len(stratum_indices) < target_stratum_size:
                repeat_factor = int(np.ceil(target_stratum_size / len(stratum_indices)))
                sampled = np.tile(stratum_indices, repeat_factor)[:target_stratum_size]
            else:
                sampled = np.random.choice(stratum_indices, target_stratum_size, replace=False)
            balanced_indices.extend(sampled)

    def _none_strategy_split(self, seed: int) -> List[np.ndarray]:
        """None strategy split."""
        split_indices = [[] for _ in range(len(self.ratio))]

        for stratum_idx in range(len(self.unique_strata)):
            stratum_seed = seed + stratum_idx
            stratum_indices = self._get_stratum_indices(stratum_idx, stratum_seed)
            np.random.seed(stratum_seed)
            shuffled = np.random.permutation(stratum_indices)

            split_sizes = self._calc_stratum_split_sizes(len(shuffled))
            self._distribute_indices(shuffled, split_sizes, split_indices)

        return [np.random.permutation(indices) for indices in split_indices]

    def _get_stratum_indices(self, stratum_idx: int, seed: Optional[int] = None) -> np.ndarray:
        """Get indices for a specific stratum."""
        stratum_mask = self.stratum_labels == stratum_idx
        indices = np.where(stratum_mask)[0]

        if self.retain_ratio < 1.0:
            retain_num = max(1, int(len(indices) * self.retain_ratio))
            if seed is not None:
                np.random.seed(seed)

            indices = np.random.choice(indices, retain_num, replace=False)

        return indices

    def _calc_stratum_split_sizes(self, total: int) -> List[int]:
        """Calculate split sizes based on the ratio."""
        split_sizes = []
        cumulative = 0
        for i, r in enumerate(self.ratio):
            if i == len(self.ratio) - 1:
                split_size = total - cumulative
            else:
                split_size = int(round(r * total))
            split_sizes.append(split_size)
            cumulative += split_size
        return split_sizes

    def _distribute_indices(self, indices: np.ndarray, split_sizes: List[int], split_indices: List[list]):
        """Distribute indices based on split sizes and store them in split_indices."""
        current = 0
        for i, size in enumerate(split_sizes):
            if size <= 0:
                continue
            end = current + size
            split_indices[i].extend(indices[current:end].tolist())
            current = end

    def _split_balanced_data(self, balanced_indices: np.ndarray) -> List[np.ndarray]:
        """Split balanced indices into specified ratios and store them in splits."""
        num_samples = len(balanced_indices)
        split_targets = [max(1, int(r * num_samples)) for r in self.ratio]
        split_targets[-1] = num_samples - sum(split_targets[:-1])

        splits = []
        current_pos = 0
        for target in split_targets:
            splits.append(balanced_indices[current_pos : current_pos + target])
            current_pos += target
        return splits

    def _save_splits(self, splits: List[np.ndarray], seed: int):
        """Write the splits to parquet files."""
        for split_idx, indices in enumerate(splits):
            output_path = os.path.join(self.output_dir, f"split_{seed}_{split_idx}.parquet")
            try:
                self.df[indices].write_parquet(output_path)
                print(f"Created split {split_idx} (seed {seed}) with {len(indices)} samples")
            except Exception as e:
                print(f"Failed to write split {split_idx} (seed {seed}): {e}")


def read_nmic_results(npz_path: str):
    data = np.load(npz_path)

    # Print keys
    print("\nKeys in the npz file:", data.files)
    # Keys in the npz file:
    # ['mi_values', 'feat_pairs', 'processed_mat', 'mat_feat_indices', 'mat_simi_feat_pairs', 'thre_cv', 'thre_pcc', 'thre_mi', 'ratio_max_window', 'ratio_min_window', 'ratio_step_window', 'ratio_step_sliding']

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

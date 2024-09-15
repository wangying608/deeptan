r"""
Read multi-omics data (and labels) and split into nested train/val/test sets.

The processed data will be stored in litdata's format.

!!! The input labels (phenotypes) are expected as follows:
    - For REGRESSION task
        + Please keep original values that are not standardized.
        + If you have MULTIPLE traits, please set different columns names in CSV file.
        + _In our pipeline, we standardize/normalize the data after splitting to avoid data leakage._
        + _The method and parameters of standardization/normalization are kept the same as those in training data._
    - For CLASSIFICATION task
        + Please transform labels by one-hot encoder MANUALLY BEFORE input.
        + The length of one-hot vectors is recommended to be **n_categories + 1** for **UNPRECEDENTED labels**.
        + If you have MULTIPLE traits, please concatenate one-hot encoded matrix along the horizontal axis before input.

"""
import os
from typing import Optional, Union, List
import numpy as np
import polars as pl
from litdata import optimize, StreamingDataLoader, StreamingDataset
from lightning import LightningDataModule
from torch import Tensor
from frn.utils.uni import intersect_lists, read_labels, zscore_labels, get_indices_ncv


class MyDataset:
    """
    Read data for litdata optimization.
    """
    def __init__(
            self,
            paths_omics: list[str],
            path_label: Optional[str] = None,
            col2use: Optional[Union[List[str], List[int]]] = None,
            sample_ind_for_zsc_label: Optional[List] = None,
            target_n_samples: Optional[int] = None,
            seed_resample: int = 42,
            n_fragments: int = 1,
        ):
        super().__init__()
        omics_dfs = [pl.read_csv(pathx, schema_overrides={"ID": pl.Utf8}) for pathx in paths_omics]
        intersect_ids_in_omics, _indices = intersect_lists([df.select("ID").to_series().to_list() for df in omics_dfs])
        for i_df in range(len(_indices)):
            omics_dfs[i_df] = omics_dfs[i_df][_indices[i_df],:].sort("ID")
        
        sample_ids = []
        labels_df = None
        dim_model_output = None
        if path_label is not None:
            labels_df, dim_model_output, sample_ids_in_labels = read_labels(path_label, col2use)
            intersect_ids, _indices = intersect_lists([intersect_ids_in_omics, sample_ids_in_labels])
            sample_ids = intersect_ids
            omics_dfs = [df[_indices[0],:].sort("ID") for df in omics_dfs]
            labels_df = labels_df[_indices[1],:].sort("ID")
        else:
            sample_ids = intersect_ids_in_omics
        n_samples = len(sample_ids)
        
        n_samples_to_add = 0
        match target_n_samples:
            case None:
                if n_fragments > 1:
                    if n_samples % n_fragments != 0:
                        n_samples_to_add = n_fragments - (n_samples % n_fragments)
            case x if x > n_samples:
                n_samples_to_add = x - n_samples
            case _:
                raise Warning("target_n_samples must be larger than the number of samples")
        
        n_samples_target = n_samples + n_samples_to_add

        if n_samples_to_add > 0:
            # Generate random indices to add
            np.random.seed(seed_resample)
            new_indices: list[int] = np.random.choice(n_samples, n_samples_to_add, replace=True).tolist()
            n_samples = n_samples_target
            sample_ids = sample_ids + [sample_ids[i] for i in new_indices]
            
            if labels_df is not None:
                labels_df = labels_df.vstack(labels_df[new_indices,:])

            omics_dfs = [df.vstack(df[new_indices,:]) for df in omics_dfs]
        
        self.sample_ids = sample_ids
        self.n_samples = n_samples
        self.omics_data = [df.drop("ID").to_numpy().astype(np.float32) for df in omics_dfs]
        self.omics_features = [df.columns for df in omics_dfs]
        if labels_df is not None:
            labels_df, z_mean, z_sd = zscore_labels(labels_df, sample_ind_for_zsc_label)
            self.z_mean = z_mean
            self.z_sd = z_sd
            self.label_data = labels_df.drop("ID").to_numpy().astype(np.float32)
            self.label_features = labels_df.columns
            self.model_output_dim = dim_model_output
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        omics_data_i = [omics_x[index, :] for omics_x in self.omics_data]
        sample_id_i = self.sample_ids[index]
        if hasattr(self, "label_data"):
            label_data_i = self.label_data[index, :]
            data_o = {"index": index, "label": label_data_i, "omics": omics_data_i, "id": sample_id_i}
        else:
            data_o = {"index": index, "omics": omics_data_i, "id": sample_id_i}
        return data_o


def optimize_data_ncv(
        output_dir: str,
        k_outer: int,
        k_inner: int,
        paths_omics: List[str],
        path_label: Optional[str] = None,
        col2use: Optional[Union[List[str], List[int]]] = None,
        std_labels: bool = True,
        fragment_elem_ids: Optional[List[List[int]]] = None,
        seed_permut: int = 42,
        seed_resample: int = 42,
        compression: Optional[str] = "zstd",
        n_workers: int = 2,
    ):
    """
    Args:
        `output_dir`: Directory to save the optimized data.
        `k_outer`: Number of outer folds.
        `k_inner`: Number of inner folds.
        `paths_omics`: List of paths to omics data.
        `path_label`: Path to label data.
        `col2use`: List of columns (of label data) to use.
        `std_labels`: Whether to standardize labels.
        `seed_permut`: Seed for permutation.
        `seed_resample`: Seed for resampling.
    """
    if fragment_elem_ids is None:
        n_fragments = int(k_outer * k_inner)
        tmp_data_init = MyDataset(paths_omics, path_label, col2use, None, None, seed_resample, n_fragments)

        # Permutate samples
        np.random.seed(seed_permut)
        _indices = np.random.permutation(len(tmp_data_init))
        fragments = np.array_split(_indices, n_fragments)
        fragments = [i.tolist() for i in fragments]
    else:
        fragments = fragment_elem_ids
        n_fragments = len(fragment_elem_ids)
        assert k_outer * k_inner == n_fragments

    for xo in range(k_outer):
        for xi in range(k_inner):
            fr_indices_trn, fr_indices_val, fr_indices_test = get_indices_ncv(k_outer, k_inner, xo, xi)
            indices_trn_samples = np.concatenate([fragments[i] for i in fr_indices_trn]).tolist()
            indices_val_samples = np.concatenate([fragments[i] for i in fr_indices_val]).tolist()
            indices_tst_samples = np.concatenate([fragments[i] for i in fr_indices_test]).tolist()
            dir_xoxi = os.path.join(output_dir, f"ncv_test_{xo}_val_{xi}")
            os.makedirs(dir_xoxi, exist_ok=True)
            
            # IF STANDARDIZE labels, calc mean & std of training set and apply to validation & test data.
            if std_labels:
                dataset_xoxi = MyDataset(paths_omics, path_label, col2use, indices_trn_samples, None, seed_resample, n_fragments)
                if dataset_xoxi.z_mean is not None:
                    dataset_xoxi.z_mean.write_csv(os.path.join(dir_xoxi, "z_mean_labels_train.csv"))
                if dataset_xoxi.z_sd is not None:
                    dataset_xoxi.z_sd.write_csv(os.path.join(dir_xoxi, "z_sd_labels_train.csv"))
            else:
                dataset_xoxi = MyDataset(paths_omics, path_label, col2use, None, None, seed_resample, n_fragments)
            
            # Start optimizing
            optimize(
                fn = dataset_xoxi.__getitem__,
                inputs = indices_trn_samples,
                output_dir = os.path.join(dir_xoxi, "train"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
            optimize(
                fn = dataset_xoxi.__getitem__,
                inputs = indices_val_samples,
                output_dir = os.path.join(dir_xoxi, "valid"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
            optimize(
                fn = dataset_xoxi.__getitem__,
                inputs = indices_tst_samples,
                output_dir = os.path.join(dir_xoxi, "test"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
    
    if path_label is not None:
        _, dim_model_output, _ = read_labels(path_label, col2use)
        df_output_dim = pl.DataFrame(data={"model_output_dim": [dim_model_output]})
        df_output_dim.write_csv(os.path.join(output_dir, "model_output_dim.csv"))
    
    return None


def optimize_data_external(
        output_dir: str,
        paths_omics: list[str],
        path_label: Optional[str] = None,
        col2use: Optional[Union[List[str], List[int]]] = None,
        compression: Optional[str] = "zstd",
        n_workers: int = 2,
    ):
    """
    Optimize data for external use.
    """
    dataset_ext = MyDataset(paths_omics, path_label, col2use)
    optimize(
        fn = dataset_ext.__getitem__,
        inputs = range(len(dataset_ext)),
        output_dir = output_dir,
        chunk_bytes = "256MB",
        compression = compression,
        num_workers = n_workers,
    )


class MyDataModule4Train(LightningDataModule):
    """
    LightningDataModule for training models with LitData.
    
    Args:
    - `litdata_dir` (str): Directory containing the LitData fragments.
    - `which_outer_testset` (int): Index of the outer test set fold.
    - `which_inner_valset` (int): Index of the inner validation set fold.
    - `batch_size` (int): Batch size for training and evaluation.
    """
    def __init__(
            self,
            litdata_dir: str,
            which_outer_testset: int,
            which_inner_valset: int,
            batch_size: int,
            n_workers: int = 2,
        ):
        super().__init__()
        self.litdata_dir = litdata_dir
        self.which_outer_testset = which_outer_testset
        self.which_inner_valset = which_inner_valset
        self.batch_size = batch_size
        self.n_workers = n_workers

    def setup(self, stage=None):
        self.dataloder_trn, self.dataloader_val, self.dataloader_test = self.read_litdata_ncv()
    
    def train_dataloader(self):
        return self.dataloder_trn
    def val_dataloader(self):
        return self.dataloader_val
    def test_dataloader(self):
        return self.dataloader_test
    
    def read_litdata_ncv(self):
        """
        Read litdata from directories and return dataloaders for NCV.
        """
        dir_train, dir_valid, dir_test = self.get_dir_ncv_litdata()
        dataloader_train = StreamingDataLoader(StreamingDataset(dir_train), batch_size=self.batch_size, num_workers=self.n_workers)
        dataloader_valid = StreamingDataLoader(StreamingDataset(dir_valid), batch_size=self.batch_size, num_workers=self.n_workers)
        dataloader_test = StreamingDataLoader(StreamingDataset(dir_test), batch_size=self.batch_size, num_workers=self.n_workers)
        return dataloader_train, dataloader_valid, dataloader_test
    
    def get_dir_ncv_litdata(self):
        dir_xoi = os.path.join(self.litdata_dir, f"ncv_test_{self.which_outer_testset}_val_{self.which_inner_valset}")
        dir_trn = os.path.join(dir_xoi, "train")
        dir_val = os.path.join(dir_xoi, "valid")
        dir_tst = os.path.join(dir_xoi, "test")
        return dir_trn, dir_val, dir_tst


class MyDataModule4Uni(LightningDataModule):
    """
    LightningDataModule for predicting/testing.

    Args:
    - `litdata_dir` (str): Directory containing the LitData for prediction.
    - `batch_size` (int): Batch size for prediction.
    """
    def __init__(
            self,
            litdata_dir: str,
            batch_size: int,
            n_workers: int = 2,
        ):
        super().__init__()
        self.litdata_dir = litdata_dir
        self.batch_size = batch_size
        self.n_workers = n_workers
    
    def setup(self, stage=None):
        self.dataloader_xxx = StreamingDataLoader(StreamingDataset(self.litdata_dir), batch_size=self.batch_size, num_workers=self.n_workers)
    
    def predict_dataloader(self):
        return self.dataloader_xxx

    # def test_dataloader(self):
    #     return self.dataloader_x


def auto_proc_feat4trn(in_array: np.ndarray, threshold_ptp: float=100.0):
    """
    Args:
    - `in_array` (np.ndarray): Input array with shape (n_features, n_samples).
    - `threshold_ptp` (float): Threshold for the range.

    Steps:
    1. Remove feature if its range is similar to zero.
    2. Apply scale and log2 transformation to each feature in the training set. (Apply log2 transformation if the range of the array is larger than the threshold.)
    
    """
    values_min = np.min(in_array, axis=1)
    values_max = np.max(in_array, axis=1)
    values_ptp = values_max - values_min

    out_array = np.copy(in_array) + 1e-4
    feat2rm = np.where(values_ptp < 1e-3)[0]
    
    for i in range(in_array.shape[0]):
        if i in feat2rm:
            continue
        if values_ptp[i] > threshold_ptp:
            # Apply log2 transformation
            out_array[i, :] = np.log2(out_array[i, :])
    
    out_array = np.delete(out_array, feat2rm, axis=0)

    values_min = np.min(out_array, axis=1)
    values_max = np.max(out_array, axis=1)
    # out_array = (out_array - values_min) / (values_max - values_min)
    out_array = (out_array - values_min[:, None]) / (values_max[:, None] - values_min[:, None])
    return out_array, values_min, values_max, feat2rm

def omics_tensor_list_to_np(batch: List[Tensor]):
    concatenated = np.concatenate([ts.numpy() for ts in batch], axis=None)
    return concatenated


def read_litdata_ncv_for_mi(
        litdata_dir: str,
        output_dir: str,
        which_outer_test: int,
        which_inner_val: int,
        threshold_ptp: float=100.0,
        path_excutable: Optional[str] = None,
        thre_sd: float = 0.05,
        thre_pcc: float = 0.9,
        thre_mi: float = 0.2,
    ) -> None:
    """
    Read specific NCV litdata from directories and calculate MI for each inner training set.
    """
    # Check if path_excutable is None
    if path_excutable is None:
        path_excutable = os.path.join(os.path.dirname(__file__), "mi2graph")
    if not os.path.exists(path_excutable):
        raise FileNotFoundError(f"Executable file not found: {path_excutable}")

    # Read litdata
    dir_xoi = os.path.join(litdata_dir, f"ncv_test_{which_outer_test}_val_{which_inner_val}")
    dataloader_trn = StreamingDataLoader(StreamingDataset(os.path.join(dir_xoi, "train")))

    # Run the compiled MI-based proccessing procedure on training data
    trnset_npy_dir = os.path.join(output_dir, "tmp_trnset")
    os.makedirs(trnset_npy_dir, exist_ok=True)
    mi_net_dir = os.path.join(output_dir, "mi_net_for_traindataset")
    os.makedirs(mi_net_dir, exist_ok=True)
    
    # Read (multiple) omics' data from dataloader_trn and save it to a matrix, then save the matrix to a NPY file.
    trn_data_matrix = [omics_tensor_list_to_np(batch['omics']) for batch in dataloader_trn]
    trn_data_matrix = np.array(trn_data_matrix).astype(np.float64).transpose()
    
    # Scale and log2 transformation for each feature
    trn_data_matrix, values_min, values_max, feat2rm = auto_proc_feat4trn(trn_data_matrix, threshold_ptp)

    # Check if the matrix contains 'None' value
    if np.any(np.isnan(trn_data_matrix)):
        raise ValueError("The matrix contains 'None' value.")
    
    # Save the matrix to a NPY file
    path_npy = os.path.join(trnset_npy_dir, f"trn_{which_outer_test}_{which_inner_val}.npy")
    np.save(path_npy, trn_data_matrix)
    
    # Save the min and max values for scaling back & scaling validation and testing dataset
    path_range = os.path.join(mi_net_dir, f"trn_{which_outer_test}_{which_inner_val}_range.npz")
    np.savez(path_range, values_min=values_min, values_max=values_max, feat2rm=feat2rm)
    
    # Run
    path_npz = os.path.join(mi_net_dir, f"trn_{which_outer_test}_{which_inner_val}.npz")
    cmd_mi = f"{path_excutable} -i {path_npy} -o {path_npz} --thresd {thre_sd} --threpcc {thre_pcc} --thremi {thre_mi}"
    print("\nRUNNING: ", cmd_mi, "\n")
    os.system(cmd_mi)

    os.remove(path_npy)
    return None

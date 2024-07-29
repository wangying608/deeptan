import os
import numpy as np
import pandas as pd
from litdata import optimize, StreamingDataLoader, StreamingDataset
from lightning import LightningDataModule
from torch import Tensor
from multiprocessing import cpu_count
n_threads = np.ceil(cpu_count() * 0.8).astype(int)


class MyDataset4Trn:
    """
    Read data from CSV files and return data for litdata.

    """
    def __init__(
            self,
            paths_omics: list[str],
            path_label: str,
            output_dim: int,
            traits_name: str | list[str],
            round_before_onehot: bool = False,
            transpose_omics: bool = False,
            resample: int | None = None,
            seed: int = 42,
        ):
        self.transpose_omics = transpose_omics
        if output_dim < 1:
            raise ValueError("output_dim must be greater than or equal to 1")
        if isinstance(traits_name, str):
            traits_name = [traits_name]
        if len(traits_name) > 1 and output_dim > 1:
            raise ValueError("output_dim must be 1 when traits_name has more than one element")
        
        self.omics_dfs = [pd.read_csv(pathx, index_col=0) for pathx in paths_omics]
        self.label_df = pd.read_csv(path_label, index_col=0)
        
        # Pick traits if corresponding column names are given
        self.label_df = self.label_df[traits_name]

        # Resample if resample is given
        if resample is not None:
            if resample > len(self.label_df.index):
                n_samples_to_add = resample - len(self.label_df.index)
                # Generate random indices to add
                np.random.seed(seed)
                new_indices = np.random.choice(len(self.label_df.index), n_samples_to_add, replace=True)
                # Add rows to the label_df
                self.label_df = pd.concat([self.label_df, self.label_df.iloc[new_indices]])
                # Add rows to the omics_dfs
                if transpose_omics:
                    self.omics_dfs = [pd.concat([df, df.iloc[:, new_indices]], axis=1) for df in self.omics_dfs]
                else:
                    self.omics_dfs = [pd.concat([df, df.iloc[new_indices, :]], axis=0) for df in self.omics_dfs]
            else:
                raise ValueError("resample must be larger than the number of samples")
        
        # If output_dim > 1, apply one-hot encoding to the label
        if output_dim > 1:
            if round_before_onehot:
                self.label_df = self.label_df.round().astype(int)
            self.label_df = pd.get_dummies(self.label_df)
            
    def __len__(self):
        return len(self.label_df.index)
    
    def __getitem__(self, index):
        if self.transpose_omics:
            omics = [df.iloc[:, index].to_numpy(np.float32) for df in self.omics_dfs]
        else:
            omics = [df.iloc[index, :].to_numpy(np.float32) for df in self.omics_dfs]
        label = self.label_df.iloc[index].to_numpy(np.float32)
        id_index = self.label_df.index[index]
        data_o = {"index": index, "label": label, "omics": omics, "id": id_index}
        return data_o


def get_indices_ncv(
        k_outer: int,
        k_inner: int,
        which_outer_test: int,
        which_inner_val: int,
    ):
    """
    Get indices of fragments for NCV.
    """
    # Init fragment indices for test dataset
    n_fragments = int(k_outer * k_inner)
    n_f_test = int(n_fragments / k_outer)
    indices_test_dataset = [i for i in range(int(which_outer_test * n_f_test), int((which_outer_test + 1) * n_f_test))]
    # Indices excluding test dataset
    indices_train_dataset = [i for i in range(n_fragments) if i not in indices_test_dataset]
    # Indices for validation dataset
    parts = np.array_split(indices_train_dataset, k_inner)
    indices_val_dataset = parts[which_inner_val].tolist()
    indices_trn_dataset = [i for i in indices_train_dataset if i not in indices_val_dataset]
    
    return indices_trn_dataset, indices_val_dataset, indices_test_dataset


def read_litdata_ncv(
        litdata_dir: str,
        which_outer_test: int,
        which_inner_val: int,
        batch_size: int = 16,
    ):
    """
    Read litdata from directories and return dataloaders for NCV.
    """
    dir_xoi = os.path.join(litdata_dir, f"ncv_test_{which_outer_test}_val_{which_inner_val}")
    dataloader_train = StreamingDataLoader(StreamingDataset(os.path.join(dir_xoi, "train")), batch_size=batch_size, num_workers=n_threads)
    dataloader_valid = StreamingDataLoader(StreamingDataset(os.path.join(dir_xoi, "valid")), batch_size=batch_size, num_workers=n_threads)
    dataloader_test = StreamingDataLoader(StreamingDataset(os.path.join(dir_xoi, "test")), batch_size=batch_size, num_workers=n_threads)
    return dataloader_train, dataloader_valid, dataloader_test


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
        ):
        super().__init__()
        self.litdata_dir = litdata_dir
        self.which_outer_testset = which_outer_testset
        self.which_inner_valset = which_inner_valset
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataloder_trn, self.dataloader_val, self.dataloader_test = read_litdata_ncv(
            self.litdata_dir,
            self.which_outer_testset,
            self.which_inner_valset,
            self.batch_size,
        )
    
    def train_dataloader(self):
        return self.dataloder_trn
    def val_dataloader(self):
        return self.dataloader_val
    def test_dataloader(self):
        return self.dataloader_test


def auto_proc_feat4trn(in_array: np.ndarray, threshold_ptp: float=100.0):
    """
    Steps:
    1. Remove feature if its range is similar to zero.
    2. Apply scale and log2 transformation to each feature in the training set. (Apply log2 transformation if the range of the array is larger than the threshold.)
    
    Args:
    - `array` (np.ndarray): Input array with shape (n_features, n_samples).
    - `threshold_ptp` (float): Threshold for the range.
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

def omics_tensor_list_to_np(batch: list[Tensor]):
    cated = np.concatenate([ts.numpy() for ts in batch], axis=None)
    return cated
    # omics = [ts.numpy() for ts in batch]
    # return omics


def read_litdata_ncv_for_mi(
        litdata_dir: str,
        output_dir: str,
        which_outer_test: int,
        which_inner_val: int,
        threshold_ptp: float=100.0,
        path_excutable: str | None = None,
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

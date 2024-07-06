import os
from multiprocessing import cpu_count
import numpy as np
import pandas as pd
from litdata import optimize, StreamingDataLoader, StreamingDataset, CombinedStreamingDataset
from lightning import LightningDataModule


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
                self.label_df = self.label_df.round()
            self.label_df = pd.get_dummies(self.label_df, columns=traits_name)
            
    def __len__(self):
        return len(self.label_df.index)
    
    def __getitem__(self, index):
        if self.transpose_omics:
            omics = [df.iloc[:, index].astype(np.float32) for df in self.omics_dfs]
        else:
            omics = [df.iloc[index, :].astype(np.float32) for df in self.omics_dfs]
        label = self.label_df.iloc[index].astype(np.float32)
        id_index = self.label_df.index[index]
        data_o = {"index": index, "label": label, "omics": omics, "id": id_index}
        return data_o


def data_opt_trn(
        output_dir: str,
        paths_omics: list[str],
        path_label: str,
        output_dim: int,
        traits_name: str | list[str],
        k_outer: int,
        k_inner: int,
        seed: int = 42,
        round_before_onehot: bool = False,
        transpose_omics: bool = False,
    ):
    """
    """
    # Find the sample number that is divisible by k_outer and k_inner
    n_samples = len(pd.read_csv(path_label, index_col=0).index)
    
    if n_samples % (k_outer * k_inner) == 0:
        num2add = 0
        goal_num = n_samples
    else:
        num2add = k_outer * k_inner - (n_samples % (k_outer * k_inner))
        goal_num = n_samples + num2add

    if num2add > 0:
        data_init = MyDataset4Trn(paths_omics, path_label, output_dim, traits_name, round_before_onehot, transpose_omics, goal_num, seed)
    else:
        data_init = MyDataset4Trn(paths_omics, path_label, output_dim, traits_name, round_before_onehot, transpose_omics, None, seed)
    
    n_fragments = int(k_outer * k_inner)

    # Set the random seed
    np.random.seed(seed)
    # Generate a random permutation of the indices
    indices = np.random.permutation(goal_num)
    # Split the indices into fragments
    fragments = np.array_split(indices, n_fragments)

    # # Optimize the data for each fragment
    n_threads = round(cpu_count() * 0.9)
    for i in range(n_fragments):
        optimize(
            fn = data_init.__getitem__,
            inputs = fragments[i].tolist(),
            output_dir = os.path.join(output_dir, f"fragment_{i}"),
            chunk_bytes = "64MB",
            num_workers = n_threads,
            # compression = "zstd",
        )


def read_litdata_to_ncv(
        litdata_dir: str,
        k_outer: int,
        k_inner: int,
        which_outer_test: int,
        which_inner_val: int,
        batch_size: int = 16,
    ):
    """
    Read litdata from directories and return dataloader for NCV.
    """
    # Init fragment indices for test dataset
    n_fragments = int(k_outer * k_inner)
    n_f_test = int(n_fragments / k_outer)
    indices_test_dataset = [i for i in range(int(which_outer_test * n_f_test), int((which_outer_test + 1) * n_f_test))]
    # Indices excluding test dataset
    indices_train_dataset = [i for i in range(n_fragments) if i not in indices_test_dataset]
    # Indices for validation dataset
    parts = np.array_split(indices_train_dataset, k_inner)
    indices_val_dataset = parts[which_inner_val]
    indices_trn_dataset = [i for i in indices_train_dataset if i not in indices_val_dataset]
    # print(f"Indices for test dataset: {indices_test_dataset}")
    # print(f"Indices for validation dataset: {indices_val_dataset}")
    # print(f"Indices for training dataset: {indices_trn_dataset}")

    # Read litdata from directories
    dataset_train = [StreamingDataset(os.path.join(litdata_dir, f"fragment_{i}")) for i in indices_trn_dataset]
    dataset_val = [StreamingDataset(os.path.join(litdata_dir, f"fragment_{i}")) for i in indices_val_dataset]
    dataset_test = [StreamingDataset(os.path.join(litdata_dir, f"fragment_{i}")) for i in indices_test_dataset]
    combined_dataset_trn = CombinedStreamingDataset(dataset_train)
    combined_dataset_val = CombinedStreamingDataset(dataset_val)
    combined_dataset_test = CombinedStreamingDataset(dataset_test)

    n_threads = round(cpu_count() * 0.9)
    dataloader_trn = StreamingDataLoader(combined_dataset_trn, batch_size=batch_size, pin_memory=True, num_workers=n_threads)
    dataloader_val = StreamingDataLoader(combined_dataset_val, batch_size=batch_size, pin_memory=True, num_workers=n_threads)
    dataloader_test = StreamingDataLoader(combined_dataset_test, batch_size=batch_size, pin_memory=True, num_workers=n_threads)
    return dataloader_trn, dataloader_val, dataloader_test


class MyDataModule4Train(LightningDataModule):
    """
    LightningDataModule for training models with LitData.

    Args:
    - `litdata_dir` (str): Directory containing the LitData fragments.
    - `k_outer` (int): Number of outer folds.
    - `k_inner` (int): Number of inner folds.
    - `which_outer_testset` (int): Index of the outer test set fold.
    - `which_inner_valset` (int): Index of the inner validation set fold.
    - `batch_size` (int): Batch size for training and evaluation.
    """
    def __init__(
            self,
            litdata_dir: str,
            k_outer: int,
            k_inner: int,
            which_outer_testset: int,
            which_inner_valset: int,
            batch_size: int,
        ):
        super().__init__()
        self.litdata_dir = litdata_dir
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.which_outer_testset = which_outer_testset
        self.which_inner_valset = which_inner_valset
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataloder_trn, self.dataloader_val, self.dataloader_test = read_litdata_to_ncv(
            self.litdata_dir,
            self.k_outer,
            self.k_inner,
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


def read_litdata_ncv_for_mi(
        litdata_dir: str,
        output_dir: str,
        k_outer: int,
        k_inner: int,
        which_outer_test: int,
        which_inner_val: int,
    ):
    """
    Read specific NCV litdata from directories and calculate MI for each inner training set.
    """
    # Init fragment indices for test dataset
    n_fragments = int(k_outer * k_inner)
    n_f_test = int(n_fragments / k_outer)
    indices_test_dataset = [i for i in range(int(which_outer_test * n_f_test), int((which_outer_test + 1) * n_f_test))]
    # Indices excluding test dataset
    indices_train_dataset = [i for i in range(n_fragments) if i not in indices_test_dataset]
    # Indices for validation dataset
    parts = np.array_split(indices_train_dataset, k_inner)
    indices_val_dataset = parts[which_inner_val]
    indices_trn_dataset = [i for i in indices_train_dataset if i not in indices_val_dataset]

    # Read litdata from directories
    dataset_train = [StreamingDataset(os.path.join(litdata_dir, f"fragment_{i}")) for i in indices_trn_dataset]
    combined_dataset_trn = CombinedStreamingDataset(dataset_train)

    # Run the compiled MI-based proccessing procedure on training data
    

    return None

r"""
Read multi-omics data (and labels) and split into nested train/val/test sets.

The processed data will be stored in litdata's format.

"""

import os
import shutil
from typing import Optional, Union, List, Dict
import itertools
import numpy as np
import polars as pl
from litdata import optimize, StreamingDataLoader, StreamingDataset
from lightning import LightningDataModule
from torch import Tensor
import frn.constants as MC
from frn.utils.uni import intersect_lists, read_labels, get_indices_ncv, ProcOnTrainSet, read_pkl_gv, onehot_encode_snp_mat, random_string


class MyDataset:
    """
    Read data for litdata optimization.

    If you have labels (phenotypes) and want to use them,
    the input labels (phenotypes) are expected as follows:
    - For **REGRESSION** task
        + Please keep original values that are not preprocessed.
        + If you have MULTIPLE traits, please set different columns names in CSV file.
        + ***In our pipeline, we standardize/normalize the data after splitting to avoid data leakage.***
        + *The method and parameters of standardization/normalization are kept the same as those in training data.*
    - For **CLASSIFICATION** task
        + Please transform labels by one-hot encoder ***MANUALLY BEFORE input***.
        + The length of one-hot vectors is recommended to be **n_categories + 1** for **UNPRECEDENTED labels**.
        + If you have MULTIPLE traits, please **concatenate** one-hot encoded matrix along the horizontal axis before input.

    *Parameters*:
    - `reproduction_mode`: whether to use the processors fitted before. Please provide `dir_preprocessors` if `reproduction_mode` is `True`.
    - `paths_omics`: the paths to omics data. Commonly, it is a `Dict` of paths to multiple `CSV` files. ***For SNP data, it is a path to a `PKL` file.***
    - `path_label`: The path to label data (a `CSV` file).
    - `col2use_in_label`: The columns to use in label data.
    - `sample_ind_for_preproc`: The indices for selecting samples for preprocessors fitting. ***If it is `None`, all samples are used.***
    - `dir_preprocessors`: The directory used to save data processors that fitted on training data. ***If it is `None`, preprocessing is not performed.***
    - `target_n_samples`: The target sample size for expanding the dataset through random sampling.
    - `seed_resample`: The random seed for sampling new samples.
    - `n_fragments`: The number of fragments (= k_outer * k_inner).
    - `prepr_labels`: Whether to preprocess labels or not.
    - `prepr_omics`: Whether to preprocess omics data or not.
    """
    def __init__(
            self,
            reproduction_mode: bool,
            paths_omics: Dict[str, str],
            path_label: Optional[str] = None,
            col2use_in_label: Optional[Union[List[str], List[int]]] = None,
            sample_ind_for_preproc: Optional[List] = None,
            dir_preprocessors: Optional[str] = None,
            target_n_samples: Optional[int] = None,
            seed_resample: int = MC.default.seed_1,
            n_fragments: int = 1,
            prepr_labels: bool = True,
            prepr_omics: bool = True,
            snp_onehot_bits: int = MC.default.snp_onehot_bits,
        ):
        super().__init__()

        self.omics_name = sorted(list(paths_omics.keys()))
        self.key_gv = None
        self.n_omics = len(self.omics_name)
        self.omics_dfs: Dict[str, pl.DataFrame] = {}
        
        self._pick_shared_samples_in_omics(paths_omics)
        
        self.sample_ids = []
        self.labels_df = None
        self.dim_model_output = None

        if path_label is not None:
            self._pick_shared_samples_in_omics_and_labels(path_label, col2use_in_label)
        else:
            self.sample_ids = self.intersect_ids_in_omics
        
        self.n_samples = len(self.sample_ids)
        self._calc_n_samples2sample(target_n_samples, n_fragments)
        

        if self.n_samples_to_add > 0:
            self._sample_new2add(seed_resample)
        
        if self.key_gv is not None:
            gv_np = self.omics_dfs[self.key_gv].drop(MC.dkey.id).to_numpy()
            gv_np_onehot = onehot_encode_snp_mat(gv_np, snp_onehot_bits)
            self.omics_dfs[self.key_gv] = pl.DataFrame({MC.dkey.id: self.sample_ids}).hstack(pl.DataFrame(data=gv_np_onehot))

        self._proc_omics(prepr_omics, sample_ind_for_preproc, dir_preprocessors, reproduction_mode)
        self._proc_labels(prepr_labels, sample_ind_for_preproc, dir_preprocessors, reproduction_mode)

        self.omics_data = []
        self.omics_features = []
        self.omics_dims = []
        for i in range(self.n_omics):
            self.omics_data.append(self.omics_dfs[self.omics_name[i]].drop(MC.dkey.id).to_numpy().astype(np.float32))
            self.omics_features.append(self.omics_dfs[self.omics_name[i]].drop(MC.dkey.id).columns)
            self.omics_dims.append(self.omics_data[i].shape[1])
        
        if dir_preprocessors is not None:
            # Write omics' dimensions for the initialization of the model
            pl.DataFrame(data={MC.dkey.omics_dim: self.omics_dims}).write_csv(os.path.join(dir_preprocessors, MC.fname.predata_omics_dims))
            # Write omics' features
            for i in range(self.n_omics):
                pl.DataFrame(data={MC.dkey.omics_feature: self.omics_features[i]}).write_csv(os.path.join(dir_preprocessors, f"omics_features_{self.omics_name[i]}.csv"))
            # Write labels' names
            if self.labels_df is not None:
                pl.DataFrame(data={MC.dkey.label: self.labels_df.columns[1:]}).write_csv(os.path.join(dir_preprocessors, MC.fname.predata_label_names))
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        omics_data_i = [omics_x[index, :] for omics_x in self.omics_data]
        sample_id_i = self.sample_ids[index]
        if hasattr(self, "label_data"):
            label_data_i = self.label_data[index, :]
            data_o = {MC.dkey.litdata_index: index, MC.dkey.litdata_label: label_data_i, MC.dkey.litdata_omics: omics_data_i, MC.dkey.litdata_id: sample_id_i}
        else:
            data_o = {MC.dkey.litdata_index: index, MC.dkey.litdata_omics: omics_data_i, MC.dkey.litdata_id: sample_id_i}
        return data_o
    
    def _pick_shared_samples_in_omics(self, paths_omics: Dict[str, str]):
        original_omics_IDs = []
        for i in range(self.n_omics):
            _tmp_path = paths_omics[self.omics_name[i]]

            if _tmp_path.upper().endswith(".CSV"):
                self.omics_dfs[self.omics_name[i]] = pl.read_csv(_tmp_path, schema_overrides={MC.dkey.id: pl.Utf8})
            
            elif _tmp_path.endswith(".pkl.gz"):
                #
                self.key_gv = self.omics_name[i]
                #
                snp_data_dict = read_pkl_gv(_tmp_path)
                snp_matrix = snp_data_dict[MC.dkey.genotype_matrix]
                snp_sample_ids = snp_data_dict[MC.dkey.sample_ids]
                snp_ids = snp_data_dict[MC.dkey.snp_ids]
                # snp_block_ids = snp_data_dict['block_ids']
                _tmp_snp_df = pl.DataFrame(data=snp_matrix, schema=snp_ids)
                # print(type(snp_sample_ids))
                # print(snp_sample_ids)
                _tmp_id = pl.DataFrame({MC.dkey.id: snp_sample_ids})
                # print(_tmp_id.shape, _tmp_snp_df.shape)
                _tmp_snp_df = _tmp_id.hstack(_tmp_snp_df)
                self.omics_dfs[self.omics_name[i]] = _tmp_snp_df

            else:
                raise ValueError("The path to the omics data is not valid.")
            
            original_omics_IDs.append(self.omics_dfs[self.omics_name[i]].select(MC.dkey.id).to_series().to_list())
        
        intersect_ids_in_omics, _indices = intersect_lists(original_omics_IDs)
        for i in range(self.n_omics):
            self.omics_dfs[self.omics_name[i]] = self.omics_dfs[self.omics_name[i]][_indices[i],:].sort(MC.dkey.id)
        
        self.intersect_ids_in_omics = intersect_ids_in_omics
    
    def _pick_shared_samples_in_omics_and_labels(self, path_label: str, col2use_in_label: Optional[Union[List[str], List[int]]]):
        labels_df, dim_model_output, sample_ids_in_labels = read_labels(path_label, col2use_in_label)
        intersect_ids, _indices = intersect_lists([self.intersect_ids_in_omics, sample_ids_in_labels])
        
        for i in range(self.n_omics):
            self.omics_dfs[self.omics_name[i]] = self.omics_dfs[self.omics_name[i]][_indices[0],:].sort(MC.dkey.id)
        labels_df = labels_df[_indices[1],:].sort(MC.dkey.id)

        self.sample_ids = intersect_ids
        self.labels_df = labels_df
        self.model_output_dim = dim_model_output
    
    def _calc_n_samples2sample(self, target_n_samples: Optional[int], n_fragments: int):
        n_samples_to_add = 0
        match target_n_samples:
            case None:
                if n_fragments > 1:
                    if self.n_samples % n_fragments != 0:
                        n_samples_to_add = n_fragments - (self.n_samples % n_fragments)
            case x if x > self.n_samples:
                n_samples_to_add = x - self.n_samples
            case _:
                raise Warning("target_n_samples must be larger than the number of samples")
        self.n_samples_to_add = n_samples_to_add
        self.n_samples_target = self.n_samples + n_samples_to_add
    
    def _sample_new2add(self, seed_resample: int):
        # Generate random indices to add
        np.random.seed(seed_resample)
        new_indices: list[int] = np.random.choice(self.n_samples, self.n_samples_to_add, replace=True).tolist()
        self.n_samples = self.n_samples_target
        self.sample_ids = self.sample_ids + [self.sample_ids[i] for i in new_indices]
        
        if self.labels_df is not None:
            self.labels_df = self.labels_df.vstack(self.labels_df[new_indices,:])

        for i in range(self.n_omics):
            self.omics_dfs[self.omics_name[i]] = self.omics_dfs[self.omics_name[i]].vstack(self.omics_dfs[self.omics_name[i]][new_indices,:])
    
    def _proc_omics(self, prepr_omics: bool, sample_ind_for_proc: Optional[List[int]], dir_preprocessors: Optional[str], reproduction_mode: bool):
        if prepr_omics:
            if reproduction_mode and dir_preprocessors is not None:
                if os.path.exists(dir_preprocessors):
                    for i in range(self.n_omics):
                        if self.key_gv is not None and self.key_gv == self.omics_name[i]:
                            continue
                        _loaded_proc = ProcOnTrainSet(self.omics_dfs[self.omics_name[i]], None)
                        _loaded_proc.load_run_preprocessors(dir_preprocessors, f'preprocessors_for_omics_{self.omics_name[i]}.pkl')
                        self.omics_dfs[self.omics_name[i]] = _loaded_proc._df
                else:
                    raise FileNotFoundError(f"Processor files for omics data are not found in {dir_preprocessors}")
            else:
                for i in range(self.n_omics):
                    if self.key_gv is not None and self.key_gv == self.omics_name[i]:
                        continue
                    _tmp_proc = ProcOnTrainSet(self.omics_dfs[self.omics_name[i]], sample_ind_for_proc)
                    _tmp_proc.pr_impute(strategy="mean")
                    _tmp_proc.pr_minmax()
                    if dir_preprocessors is not None:
                        _tmp_proc.save_preprocessors(dir_preprocessors, f'preprocessors_for_omics_{self.omics_name[i]}.pkl')
                    self.omics_dfs[self.omics_name[i]] = _tmp_proc._df
    
    def _proc_labels(self, prepr_labels: bool, sample_ind_for_proc: Optional[List[int]], dir_preprocessors: Optional[str], reproduction_mode: bool):
        if self.labels_df is not None and prepr_labels:
            if reproduction_mode and dir_preprocessors is not None:
                if os.path.exists(dir_preprocessors):
                    labels_processor = ProcOnTrainSet(self.labels_df, None)
                    labels_processor.load_run_preprocessors(dir_preprocessors, 'preprocessors_for_labels.pkl')
                    self.labels_df = labels_processor._df
                else:
                    raise FileNotFoundError(f"Processor files for labels are not found in {dir_preprocessors}")
            else:
                labels_processor = ProcOnTrainSet(self.labels_df, sample_ind_for_proc)
                labels_processor.pr_impute(strategy="mean")
                # labels_processor.pr_minmax()
                labels_processor.pr_zscore()
                if dir_preprocessors is not None:
                    labels_processor.save_preprocessors(dir_preprocessors, 'preprocessors_for_labels.pkl')
                self.labels_df = labels_processor._df
                self.label_data = self.labels_df.drop(MC.dkey.id).to_numpy().astype(np.float32)


class OptimizeLitdataNCV:
    def __init__(
            self,
            paths_omics: Dict[str, str],
            path_label: Optional[str],
            output_dir: str,
            k_outer: int,
            k_inner: int,
            fragment_elem_ids: Optional[List[List[int]]] = None,
            which_outer_inner: Optional[List[int]] = None,
            col2use_in_labels: Optional[Union[List[str], List[int]]] = None,
            prepr_labels: bool = True,
            prepr_omics: bool = True,
            seed_permut: int = MC.default.seed_1,
            seed_resample: int = MC.default.seed_2,
            compression: Optional[str] = MC.default.compression_alg,
            n_workers: int = MC.default.n_workers_litdata,
        ):
        """
        Args:
            `paths_omics`: Paths to omics data. Dict of {name: path}.
            `path_label`: Path to label data.
            `output_dir`: Directory to save the optimized data.
            `k_outer`: Number of outer folds.
            `k_inner`: Number of inner folds.
            `fragment_elem_ids`: List of list of indices of elements in each fragment. For nested cross validation with 10 outer folds and 5 inner folds, this should be a list of 50 lists of indices.
            `which_outer_inner`: If specified, only the specified outer-inner fold will be optimized.
            `col2use_in_labels`: Columns to use in labels.
            `prepr_labels`: Whether to preprocess labels.
            `prepr_omics`: Whether to preprocess omics.
            `seed_permut`: Seed for permutation.
            `seed_resample`: Seed for resampling for the target number of samples.
            `compression`: Compression method.
            `n_workers`: Number of workers.
        """
        self.paths_omics = paths_omics
        self.path_label = path_label
        self.output_dir = output_dir
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.col2use_in_labels = col2use_in_labels
        self.prepr_labels = prepr_labels
        self.prepr_omics = prepr_omics
        self.seed_resample = seed_resample
        self.compression = compression
        self.n_workers = n_workers
        self.which_outer_inner = which_outer_inner
        if which_outer_inner is not None:
            if len(which_outer_inner) != 2:
                raise ValueError("which_outer_inner must be a list of two elements")
        self.fragments = self._check(seed_permut, fragment_elem_ids)
        self.n_fragments = len(self.fragments)

        self.litdata_cache_dir = os.path.join(output_dir, f".cache_{random_string(9)}")
        os.environ["DATA_OPTIMIZER_CACHE_FOLDER"] = self.litdata_cache_dir
        # self.run_optimization()

    def run_optimization(self):
        if self.which_outer_inner is None:
            combn_outer_inner = list(itertools.product(range(self.k_outer), range(self.k_inner)))
            for xo, xi in combn_outer_inner:
                self.optimize_xoxi(xo, xi)
        else:
            self.optimize_xoxi(*self.which_outer_inner)
        
        if self.path_label is not None:
            _, dim_model_output, _ = read_labels(self.path_label, self.col2use_in_labels)
            df_output_dim = pl.DataFrame(data={MC.dkey.model_output_dim: [dim_model_output]})
            path_output_dim = os.path.join(self.output_dir, MC.fname.output_dim)
            if not os.path.exists(path_output_dim):
                df_output_dim.write_csv(path_output_dim)
        
        # Check if Genomic Variants are available in paths_omics
        for px in self.paths_omics.values():
            if px.endswith(".pkl.gz"):
                path_cp_pklgz = os.path.join(self.output_dir, MC.fname.genotypes)
                if not os.path.exists(path_cp_pklgz):
                    shutil.copy(px, path_cp_pklgz)

        # Remove cache dir
        os.remove(self.litdata_cache_dir)
        return None

    def _check(self, seed_permut: int, fragment_elem_ids: Optional[List[List[int]]]):
        if fragment_elem_ids is None:
            n_fragments = int(self.k_outer * self.k_inner)
            tmp_init = MyDataset(False, self.paths_omics, self.path_label, self.col2use_in_labels, None, None, None, MC.default.seed_1, n_fragments, False, False)

            np.random.seed(seed_permut)
            _indices = np.random.permutation(len(tmp_init))
            fragments = np.array_split(_indices, n_fragments)
            fragments = [i.tolist() for i in fragments]
        else:
            fragments = fragment_elem_ids
            n_fragments = len(fragments)
            assert n_fragments == self.k_outer * self.k_inner
        
        return fragments
    
    def optimize_xoxi(self, which_outer_test: int, which_inner_valid: int):
        fr_indices_trn, fr_indices_val, fr_indices_test = get_indices_ncv(self.k_outer, self.k_inner, which_outer_test, which_inner_valid)
        ind_trn = np.concatenate([self.fragments[i] for i in fr_indices_trn]).tolist()
        ind_val = np.concatenate([self.fragments[i] for i in fr_indices_val]).tolist()
        ind_tst = np.concatenate([self.fragments[i] for i in fr_indices_test]).tolist()
        dir_xoxi = os.path.join(self.output_dir, f"ncv_test_{which_outer_test}_val_{which_inner_valid}")
        os.makedirs(dir_xoxi, exist_ok=True)

        dataset_xoxi = MyDataset(False, self.paths_omics, self.path_label, self.col2use_in_labels, ind_trn, dir_xoxi, None, self.seed_resample, self.n_fragments, self.prepr_labels, self.prepr_omics)
        sample_ids = dataset_xoxi.sample_ids

        # Write sample IDs
        _df_ids = pl.DataFrame(sample_ids, schema=[MC.dkey.id])
        _df_ids.write_csv(os.path.join(dir_xoxi, MC.fname.predata_ids))
        _df_ids_trn = pl.DataFrame([sample_ids[i] for i in ind_trn], schema=[MC.dkey.id])
        _df_ids_trn.write_csv(os.path.join(dir_xoxi, MC.fname.predata_ids_trn))
        _df_ids_val = pl.DataFrame([sample_ids[i] for i in ind_val], schema=[MC.dkey.id])
        _df_ids_val.write_csv(os.path.join(dir_xoxi, MC.fname.predata_ids_val))
        _df_ids_tst = pl.DataFrame([sample_ids[i] for i in ind_tst], schema=[MC.dkey.id])
        _df_ids_tst.write_csv(os.path.join(dir_xoxi, MC.fname.predata_ids_tst))
        
        # Start optimizing
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_trn,
            output_dir = os.path.join(dir_xoxi, MC.title_train),
            chunk_bytes = MC.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_val,
            output_dir = os.path.join(dir_xoxi, MC.title_val),
            chunk_bytes = MC.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )
        optimize(
            fn = dataset_xoxi.__getitem__,
            inputs = ind_tst,
            output_dir = os.path.join(dir_xoxi, MC.title_test),
            chunk_bytes = MC.default.chunk_bytes,
            compression = self.compression,
            num_workers = self.n_workers,
        )


def optimize_data_external(
        output_dir: str,
        paths_omics: Dict[str, str],
        path_label: Optional[str] = None,
        col2use_in_labels: Optional[Union[List[str], List[int]]] = None,
        prepr_labels: bool = True,
        prepr_omics: bool = True,
        reproduction_mode: bool = False,
        dir_preprocessors: Optional[str] = None,
        compression: Optional[str] = MC.default.compression_alg,
        n_workers: int = MC.default.n_workers_litdata,
        chunk_bytes: str = MC.default.chunk_bytes,
    ):
    """
    Optimize data for external use.

    For details, see the documentation of the class `OptimizeLitdataNCV`.
    
    """
    dataset_ext = MyDataset(
        reproduction_mode=reproduction_mode,
        paths_omics=paths_omics,
        path_label=path_label,
        col2use_in_label=col2use_in_labels,
        dir_preprocessors=dir_preprocessors,
        prepr_labels=prepr_labels,
        prepr_omics=prepr_omics,
    )
    optimize(
        fn = dataset_ext.__getitem__,
        inputs = range(len(dataset_ext)),
        output_dir = output_dir,
        chunk_bytes = chunk_bytes,
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
            n_workers: int = MC.default.n_workers,
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
        dataloader_train = StreamingDataLoader(StreamingDataset(dir_train), batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True)
        dataloader_valid = StreamingDataLoader(StreamingDataset(dir_valid), batch_size=self.batch_size, num_workers=self.n_workers)
        dataloader_test = StreamingDataLoader(StreamingDataset(dir_test), batch_size=self.batch_size, num_workers=self.n_workers)
        return dataloader_train, dataloader_valid, dataloader_test
    
    def get_dir_ncv_litdata(self):
        self.dir_xoi = os.path.join(self.litdata_dir, f"ncv_test_{self.which_outer_testset}_val_{self.which_inner_valset}")
        dir_trn = os.path.join(self.dir_xoi, MC.title_train)
        dir_val = os.path.join(self.dir_xoi, MC.title_val)
        dir_tst = os.path.join(self.dir_xoi, MC.title_test)
        return dir_trn, dir_val, dir_tst
    
    def read_omics_dims(self):
        return pl.read_csv(os.path.join(self.dir_xoi, MC.fname.predata_omics_dims)).select(MC.dkey.omics_dim).to_series().to_list()


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
            n_workers: int = MC.default.n_workers,
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

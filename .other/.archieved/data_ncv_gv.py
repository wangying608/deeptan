import os
import shutil
from typing import Optional, Union, List
import numpy as np
import polars as pl
from litdata import optimize
from frn.utils.uni import one_hot_encode_snp_matrix, read_pkl_gv, intersect_lists, read_labels, get_indices_ncv, ProcOnTrainSet


class SNPDataset:
    """
    Return data for litdata optimization.

    The input labels (phenotypes) are expected as follows:
    - For REGRESSION task
        + Please keep original values that are not standardized.
        + If you have MULTIPLE traits, please set different columns names in CSV file.
        + _In our pipeline, we standardize/normalize the data after splitting to avoid data leakage._
        + _The method and parameters of standardization/normalization are kept the same as those in training data._
    - For CLASSIFICATION task
        + Please transform labels by one-hot encoder MANUALLY BEFORE input.
        + The length of one-hot vectors is recommended to be **n_categories + 1** for **UNPRECEDENTED labels**.
        + If you have MULTIPLE traits, please concatenate one-hot encoded matrix along the horizontal axis before input.

    Input:
    - `col2use`: column names or indices of label dataframe to be used.
    - `sample_ind_for_zsc_label`: indices of samples to be used for standardization of all labels.
    - `len_one_hot_vec`: the length of the one-hot vector for each SNP.
        - Default is 10, which means 10 genotypes.
        - If some SNPs are missing, all elements of their vectors are 0.
    - `resample`: the goal number of samples.
    - `seed_resample`: the seed for random sampling.
    """
    def __init__(
            self,
            reproduction_mode: bool,
            path_gtype_pkl: str,
            path_label: Optional[str] = None,
            col2use: Optional[Union[List[str], List[int]]] = None,
            sample_ind_for_zsc_label: Optional[List] = None,
            dir_save_processors: Optional[str] = None,
            len_one_hot_vec: int = 10,
            resample: Optional[int] = None,
            seed_resample: int = 42,
            process_labels: bool = True,
        ):
        super().__init__()

        snp_data_dict = read_pkl_gv(path_gtype_pkl)
        snp_matrix = snp_data_dict['gt_mat']
        sample_ids_in_mat = snp_data_dict['sample_ids']

        if path_label is not None:
            self.labels_df, self.dim_model_output, sample_ids_in_labels = read_labels(path_label, col2use)
            
            intersect_ids, _indices = intersect_lists([sample_ids_in_labels, sample_ids_in_mat])
            indices_in_labels, indices_in_snp = _indices
            self.snp_mat = snp_matrix[indices_in_snp]
            self.labels_df = self.labels_df[indices_in_labels,:]
            self.sample_ids = intersect_ids
        else:
            self.snp_mat = snp_matrix
            self.sample_ids = sample_ids_in_mat

        self.snp_data = one_hot_encode_snp_matrix(self.snp_mat, len_one_hot_vec)
        self.num_samples = len(self.snp_data)

        if resample is not None:
            if resample > self.num_samples:
                n_samples_to_add = resample - self.num_samples
                # Generate random indices to add
                np.random.seed(seed_resample)
                new_indices: list[int] = np.random.choice(self.num_samples, n_samples_to_add, replace=True).tolist()
                print(f"Resampled: {new_indices}")
                self.num_samples = resample

                if hasattr(self, 'labels_df'):
                    self.labels_df = self.labels_df.vstack(self.labels_df[new_indices,:])
                
                tmp_snp2add = [self.snp_data[i] for i in new_indices]
                self.snp_data = self.snp_data + tmp_snp2add

                tmp_ids2add = [self.sample_ids[i] for i in new_indices]
                self.sample_ids = self.sample_ids + tmp_ids2add

            else:
                raise ValueError("resample must be larger than the number of samples")
        
        self._proc_labels(process_labels, sample_ind_for_zsc_label, dir_save_processors, reproduction_mode)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.snp_data[idx]
        idxx = self.sample_ids[idx]
        if hasattr(self, 'label_data'):
            y = self.label_data[idx]
            data_o = {"index": idx, "label": y, "snp": x, "id": idxx}
        else:
            data_o = {"index": idx, "snp": x, "id": idxx}
        return data_o

    def _proc_labels(self, process_labels: bool, sample_ind_for_proc: Optional[List[int]], dir_save_processors: Optional[str], reproduction_mode: bool):
        if self.labels_df is not None and process_labels:
            if reproduction_mode and dir_save_processors is not None:
                if os.path.exists(dir_save_processors):
                    labels_processor = ProcOnTrainSet(self.labels_df, None)
                    labels_processor.load_run_processors(dir_save_processors, 'data_processors_for_labels.pkl')
                    self.labels_df = labels_processor._df
                else:
                    raise FileNotFoundError(f"Processor files for labels are not found in {dir_save_processors}")
            else:
                labels_processor = ProcOnTrainSet(self.labels_df, sample_ind_for_proc)
                labels_processor.pr_impute(strategy="mean")
                labels_processor.pr_zscore()
                if dir_save_processors is not None:
                    labels_processor.save_processors(dir_save_processors, 'data_processors_for_labels.pkl')
                self.labels_df = labels_processor._df
                self.label_data = self.labels_df.drop("ID").to_numpy().astype(np.float32)


def init_snp_dataset(
        n_fragments: int,
        path_gtype_pkl: str,
        path_label: Optional[str] = None,
        col2use: Optional[Union[List[str], List[int]]] = None,
        sample_ind_for_zsc_label: Optional[List] = None,
        len_one_hot_vec: int = 10,
        seed_resample: int = 42,
        dir_save_processors: Optional[str] = None,
    ):
    n_samples = 0
    if path_label is not None:
        n_samples = len(pl.read_csv(path_label, columns=[0]))
    else:
        n_samples = len(read_pkl_gv(path_gtype_pkl)['sample_ids'])
    
    print(f"\nTotal number of samples: {n_samples}\n")

    if n_samples % n_fragments == 0:
        num2add = 0
        goal_num = n_samples
    else:
        num2add = n_fragments - (n_samples % n_fragments)
        goal_num = n_samples + num2add
    
    if num2add > 0:
        data_init = SNPDataset(False, path_gtype_pkl, path_label, col2use, sample_ind_for_zsc_label, dir_save_processors, len_one_hot_vec, goal_num, seed_resample)
    else:
        data_init = SNPDataset(False, path_gtype_pkl, path_label, col2use, sample_ind_for_zsc_label, dir_save_processors, len_one_hot_vec, None)
    
    return data_init


def snp_data_opt_ncv(
        output_dir: str,
        k_outer: int,
        k_inner: int,
        path_gtype_pkl: str,
        path_label: Optional[str] = None,
        col2use: Optional[Union[List[str], List[int]]] = None,
        std_labels: bool = True,
        fragment_elem_ids: Optional[List[List[int]]] = None,
        len_one_hot_vec: int = 10,
        seed_permut: int = 42,
        seed_resample: int = 42,
        compression: Optional[str] = "zstd",
        n_workers: int = 2,
    ):
    """
    Generate SNP (and label) data for litdata optimization.
    For nested cross-validation (NCV).
    """
    if fragment_elem_ids is None:
        n_fragments = int(k_outer * k_inner)
        tmp_data_init = init_snp_dataset(n_fragments, path_gtype_pkl, path_label, col2use, None, len_one_hot_vec, seed_resample)

        # Set the random seed
        np.random.seed(seed_permut)
        # Generate a random permutation of the indices
        _indices = np.random.permutation(len(tmp_data_init))
        # Split the indices into fragments
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
                snp_dataset_xoxi = init_snp_dataset(n_fragments, path_gtype_pkl, path_label, col2use, indices_trn_samples, len_one_hot_vec, seed_resample, dir_xoxi)
            else:
                snp_dataset_xoxi = init_snp_dataset(n_fragments, path_gtype_pkl, path_label, col2use, None, len_one_hot_vec, seed_resample, dir_xoxi)
            #
            optimize(
                fn = snp_dataset_xoxi.__getitem__,
                inputs = indices_trn_samples,
                output_dir = os.path.join(dir_xoxi, "train"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
            optimize(
                fn = snp_dataset_xoxi.__getitem__,
                inputs = indices_val_samples,
                output_dir = os.path.join(dir_xoxi, "valid"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
            optimize(
                fn = snp_dataset_xoxi.__getitem__,
                inputs = indices_tst_samples,
                output_dir = os.path.join(dir_xoxi, "test"),
                chunk_bytes = "256MB",
                compression = compression,
                num_workers = n_workers,
            )
    
    shutil.copy(path_gtype_pkl, os.path.join(output_dir, "genotypes.pkl.gz"))
    if path_label is not None:
        _, dim_model_output, _ = read_labels(path_label, col2use)
        df_output_dim = pl.DataFrame(data={"model_output_dim": [dim_model_output]})
        df_output_dim.write_csv(os.path.join(output_dir, "model_output_dim.csv"))
    return None


def snp_data_opt_external(
        output_dir: str,
        path_gtype_pkl: str,
        path_label: Optional[str] = None,
        col2use: Optional[Union[list[int], list[str]]] = None,
        reproduction_mode: bool = False,
        len_one_hot_vec: int = 10,
        compression: Optional[str] = "zstd",
        n_workers: int = 2,
    ):
    """
    Generate SNP (and label) data for litdata optimization.
    - For **External validation** purpose.
    - For **Prediction** purpose, use this function without input `path_label` and `traits_name`.
    """
    data_init = SNPDataset(
        reproduction_mode=reproduction_mode,
        path_gtype_pkl=path_gtype_pkl,
        path_label=path_label,
        col2use=col2use,
        sample_ind_for_zsc_label=None,
        dir_save_processors=output_dir,
        len_one_hot_vec=len_one_hot_vec,
    )
    optimize(
        fn = data_init.__getitem__,
        inputs = list(range(data_init.__len__())),
        output_dir = output_dir,
        chunk_bytes = "256MB",
        compression = compression,
        num_workers = n_workers,
    )

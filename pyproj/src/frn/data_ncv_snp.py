import os
import shutil
import numpy as np
import pandas as pd
from litdata import optimize
from utils import one_hot_encode_snp_matrix, read_pkl_gv
from data_ncv import get_indices_ncv
from multiprocessing import cpu_count
# n_threads = np.ceil(cpu_count() * 0.33).astype(int)
n_threads = 2
if cpu_count() < n_threads:
    n_threads = cpu_count()


class SNPDataset:
    """
    Return data for litdata optimization.

    Labels (Phenotypes) are expected as follows:
    - For REGRESSION task, please keep original values that are not standardized.
      If you have MULTIPLE traits, please set different columns names in CSV file.
    - For CLASSIFICATION task, please transform labels by one-hot encoder.
      The length of one-hot vectors should be **n_categories + 1** for **UNPRECEDENTED labels**.
      If you have MULTIPLE traits, please concatenate one-hot encoded matrix along the horizontal axis before input.

    Input:
    - `col2use`: column names or indices to be used.
    - `sample_inds_for_zsc_label`: indices of samples to be used for standardization of labels.
    - `len_one_hot_vec`: the length of the one-hot vector for each SNP.
        - Default is 10, which means 10 genotypes.
        - If all elements of the vector are 0, the SNP is missing.
    - `resample`: the number of samples to be picked for augmentation.
    - `seed`: random seed for reproducibility.
    """
    def __init__(
            self,
            path_gtype_pkl: str,
            path_label: str | None = None,
            col2use: list[str] | list[int] | None = None,
            sample_ind_for_zsc_label: list | None = None,
            len_one_hot_vec: int = 10,
            resample: int | None = None,
            seed_resample: int = 42,
        ):
        super().__init__()

        snp_data_dict = read_pkl_gv(path_gtype_pkl)
        snp_matrix = snp_data_dict['gt_mat']
        sample_ids_in_mat = snp_data_dict['sample_ids']

        if path_label is not None:
            self.labels_df, self.dim_model_output, sample_ids_in_labels = self.read_labels(path_label, col2use)
            
            intersect_ids, indices_in_snp, indices_in_labels = self.intersect_label_snp(sample_ids_in_labels, sample_ids_in_mat)
            assert len(intersect_ids) > 0, "No intersection between SNPs and labels"
            self.snp_mat = snp_matrix[indices_in_snp]
            self.labels_df = self.labels_df.iloc[indices_in_labels]
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
                    self.labels_df = pd.concat([self.labels_df, self.labels_df.iloc[new_indices]], axis=0)
                
                tmp_snp2add = [self.snp_data[i] for i in new_indices]
                self.snp_data = self.snp_data + tmp_snp2add

                tmp_ids2add = [self.sample_ids[i] for i in new_indices]
                self.sample_ids = self.sample_ids + tmp_ids2add

            else:
                raise ValueError("resample must be larger than the number of samples")
        
        if hasattr(self, 'labels_df'):
            self.labels_df, self.z_mean, self.z_sd = self.zscore_labels(self.labels_df, sample_ind_for_zsc_label)
            self.label_data = self.labels_df.to_numpy(np.float32)

    def __len__(self):
        return self.num_samples

    def get_item(self, idx):
        x = self.snp_data[idx]
        idxx = self.sample_ids[idx]
        if hasattr(self, 'label_data'):
            y = self.label_data[idx]
            data_o = {"index": idx, "label": y, "snp": x, "id": idxx}
        else:
            data_o = {"index": idx, "snp": x, "id": idxx}
        return data_o

    def intersect_label_snp(self, sample_ids_in_labels: list[str], sample_ids_in_mat: list[str]):
        intersect_ids = np.sort(np.intersect1d(sample_ids_in_labels, sample_ids_in_mat)).tolist()
        assert len(intersect_ids) == len(sample_ids_in_labels), "Some sample ids in snp matrix are not found in label file"
        
        # Get indices of samples in mat that are also in intersect_ids
        # The result should keep the order of intersect_ids
        indices_in_snp = []
        indices_in_labels = []
        for id in intersect_ids:
            indices_in_snp.append(np.where(np.isin(sample_ids_in_mat, id))[0].tolist()[0])
            indices_in_labels.append(np.where(np.isin(sample_ids_in_labels, id))[0].tolist()[0])

        return intersect_ids, indices_in_snp, indices_in_labels
    
    def read_labels(
            self,
            path_label: str,
            col2use: list[str] | list[int] | None = None,
        ):
        label_df = pd.read_csv(path_label, index_col=0)
        if col2use is not None:
            label_df = label_df.loc[:, col2use]
        dim_model_output = len(label_df.columns)
        sample_ids = label_df.index.astype(str).tolist()
        return label_df, dim_model_output, sample_ids
    
    def zscore_labels(self, labels: pd.DataFrame, sample_ind_for_zsc_label: list | None = None):
        z_mean = None
        z_sd = None
        df_o = labels
        if sample_ind_for_zsc_label is not None:
            # Calculate mean and std for each column for the samples in sample_ind_for_zsc_label
            z_mean = labels.iloc[sample_ind_for_zsc_label].mean()
            z_sd = labels.iloc[sample_ind_for_zsc_label].std()
            # And then z-score the whole labels
            df_o = (labels - z_mean) / z_sd
        print(f"z-score mean: {z_mean}, z-score std: {z_sd}")
        return df_o, z_mean, z_sd


def init_snp_dataset(
        n_fragments: int,
        path_gtype_pkl: str,
        path_label: str | None = None,
        col2use: list[str] | list[int] | None = None,
        sample_ind_for_zsc_label: list | None = None,
        len_one_hot_vec: int = 10,
        seed_resample: int = 42,
    ):
    n_samples = 0
    if path_label is not None:
        n_samples = len(pd.read_csv(path_label, index_col=0).index)
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
        data_init = SNPDataset(path_gtype_pkl, path_label, col2use, sample_ind_for_zsc_label, len_one_hot_vec, goal_num, seed_resample)
    else:
        data_init = SNPDataset(path_gtype_pkl, path_label, col2use, sample_ind_for_zsc_label, len_one_hot_vec, None)
    
    return data_init


def snp_data_opt_ncv(
        output_dir: str,
        k_outer: int,
        k_inner: int,
        path_gtype_pkl: str,
        path_label: str | None = None,
        col2use: list[int] | list[str] | None = None,
        standardize_label: bool = True,
        len_one_hot_vec: int = 10,
        seed_permut: int = 42,
        seed_resample: int = 42,
    ):
    """
    Generate SNP (and label) data for litdata optimization.
    For nested cross-validation (NCV).
    """
    n_fragments = int(k_outer * k_inner)
    tmp_data_init = init_snp_dataset(n_fragments, path_gtype_pkl, path_label, col2use, None, len_one_hot_vec, seed_resample)

    # Set the random seed
    np.random.seed(seed_permut)
    # Generate a random permutation of the indices
    indices = np.random.permutation(len(tmp_data_init))
    # Split the indices into fragments
    fragments = np.array_split(indices, n_fragments)

    for xo in range(k_outer):
        for xi in range(k_inner):
            fr_indices_trn_dataset, fr_indices_val_dataset, fr_indices_test_dataset = get_indices_ncv(k_outer, k_inner, xo, xi)
            indices_trn_samples = np.concatenate([fragments[i].tolist() for i in fr_indices_trn_dataset]).tolist()
            indices_val_samples = np.concatenate([fragments[i].tolist() for i in fr_indices_val_dataset]).tolist()
            indices_tst_samples = np.concatenate([fragments[i].tolist() for i in fr_indices_test_dataset]).tolist()
            dir_xoxi = os.path.join(output_dir, f"ncv_test_{xo}_val_{xi}")
            os.makedirs(dir_xoxi, exist_ok=True)
            # IF STANDARDIZE labels, calc mean & std of training set and apply to validation & test sets
            if standardize_label:
                snp_dataset_xoxi = init_snp_dataset(n_fragments, path_gtype_pkl, path_label, col2use, indices_trn_samples, len_one_hot_vec, seed_resample)
                # Save mean & std of training set
                snp_dataset_xoxi.z_mean.to_csv(os.path.join(dir_xoxi, "z_mean_labels_train.csv"))
                snp_dataset_xoxi.z_sd.to_csv(os.path.join(dir_xoxi, "z_sd_labels_train.csv"))
            else:
                snp_dataset_xoxi = tmp_data_init
            #
            optimize(
                fn = snp_dataset_xoxi.get_item,
                inputs = indices_trn_samples,
                output_dir = os.path.join(dir_xoxi, "train"),
                chunk_bytes = "256MB",
                compression = "zstd",
                num_workers = n_threads,
            )
            optimize(
                fn = snp_dataset_xoxi.get_item,
                inputs = indices_val_samples,
                output_dir = os.path.join(dir_xoxi, "valid"),
                chunk_bytes = "256MB",
                compression = "zstd",
                num_workers = n_threads,
            )
            optimize(
                fn = snp_dataset_xoxi.get_item,
                inputs = indices_tst_samples,
                output_dir = os.path.join(dir_xoxi, "test"),
                chunk_bytes = "256MB",
                compression = "zstd",
                num_workers = n_threads,
            )
    
    shutil.copy(path_gtype_pkl, os.path.join(output_dir, "genotypes.pkl.gz"))
    if path_label is not None:
        df_output_dim = pd.DataFrame(data={"model_output_dim": [tmp_data_init.dim_model_output]})
        df_output_dim.to_csv(os.path.join(output_dir, "model_output_dim.csv"), index=False)
    return None


def snp_data_opt_external(
        output_dir: str,
        path_gtype_pkl: str,
        path_label: str | None = None,
        col2use: list[int] | list[str] | None = None,
        len_one_hot_vec: int = 10,
    ):
    """
    Generate SNP (and label) data for litdata optimization.
    - For **External validation** purpose.
    - For **Prediction** purpose, use this function without input `path_label` and `traits_name`.
    """
    data_init = SNPDataset(
        path_gtype_pkl,
        path_label,
        col2use,
        None,
        len_one_hot_vec,
    )
    optimize(
        fn = data_init.get_item,
        inputs = list(range(data_init.__len__())),
        output_dir = output_dir,
        chunk_bytes = "256MB",
        compression = "zstd",
        num_workers = n_threads,
    )

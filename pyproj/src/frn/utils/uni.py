r"""
Some universal functions.
"""
import os
import time
import random
import string
import numpy as np
import pandas as pd
import pickle
import gzip
import optuna
from typing import Any, List, Dict, Optional, Union
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from multiprocessing import cpu_count


def get_avail_cpu_count(target_n: int) -> int:
    total_n = cpu_count()
    n_cpu = target_n
    if target_n <= 0:
        n_cpu = total_n
    else:
        n_cpu = min(target_n, total_n)
    return n_cpu


def time_string() -> str:
    _time_str = time.strftime('%Y%m%d%H%M%S', time.localtime())
    return _time_str

def random_string(length: int = 7) -> str:
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def idx_convert(indices: List[int], len_one_hot_vec: int = 10) -> List[int]:
    """
    Convert the indices to the corresponding indices in the one-hot vector.
    """
    converted_indices = [(i * len_one_hot_vec + nx) for nx in range(len_one_hot_vec) for i in indices]
    return sorted(converted_indices)


# def intersect_two_str_list(ids_seq1: List[str], ids_seq2: List[str]):
#     intersect_ids = np.sort(np.intersect1d(ids_seq1, ids_seq2)).tolist()
#     # assert len(intersect_ids) == len(ids_seq1), "Some ids in seq1 are not found in seq2."
#     assert len(intersect_ids) > 9, "Less than 10 intersecting IDs!"
#     # Get indices of IDs in seq2 that are also in intersect_ids
#     # The result should keep the order of intersect_ids
#     indices_in_seq2 = []
#     indices_in_seq1 = []
#     for id in intersect_ids:
#         indices_in_seq2.append(np.where(np.isin(ids_seq2, id))[0].tolist()[0])
#         indices_in_seq1.append(np.where(np.isin(ids_seq1, id))[0].tolist()[0])
#     return intersect_ids, indices_in_seq1, indices_in_seq2

def intersect_lists(lists: List[List[Any]], get_indices: bool = True, to_sorted: bool = True):
    shared = list(set.intersection(*map(set, lists)))
    assert len(shared) > 0, "No intersecting elements!"
    if to_sorted:
        shared = sorted(shared)
    if get_indices:
        indices = []
        # for xl in range(len(lists)):
        #     # Accelerate the search by NumPy
        #     indices.append(np.where(np.isin(lists[xl], shared))[0].tolist())
        # !!!!!!!!!!! The following can keep the order of `shared` !!!!!!!!!!!!!!
        for xl in range(len(lists)):
            indices.append([])
        for i in shared:
            for xl in range(len(lists)):
                indices[xl].append(np.where(np.isin(lists[xl], i))[0].tolist()[0])
        return shared, indices
    else:
        return shared

def read_labels(path_label: str, col2use: Optional[Union[List[str], List[int]]] = None):
    label_df = pd.read_csv(path_label, index_col=0)
    if col2use is not None:
        label_df = label_df.loc[:, col2use]
    dim_model_output = len(label_df.columns)
    sample_ids = label_df.index.astype(str).tolist()
    return label_df, dim_model_output, sample_ids

def zscore_labels(labels: pd.DataFrame, sample_ind_for_zsc_label: Optional[List] = None):
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


def one_hot_encode_snp_matrix(
        snp_matrix: np.ndarray,
        len_one_hot_vec: int = 10,
        genes_snps: Optional[List[List[int]]] = None,
    ):
    """
    One-hot encode the SNP matrix.
    """
    if genes_snps is not None:
        num_genes = len(genes_snps)
        indices_snp = []
        for i_gene in range(num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_one_hot_vec))
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_one_hot_vec + 1)[snp_vec][:, 1:].reshape(-1)
            snp_vec_genes = [snp_vec[indices_snp[i_gene]].astype(np.float32) for i_gene in range(num_genes)]
            snp_data.append(snp_vec_genes)
    else:
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_one_hot_vec + 1)[snp_vec][:, 1:].reshape(-1).astype(np.float32)
            snp_data.append(snp_vec)

    return snp_data


def read_pkl_gv(path_pkl: str) -> Dict[str, Any]:
    """
    Read processed VCF data from a pickle file.
    """
    with gzip.open(path_pkl, 'rb') as file:
        # Initialize an empty list to hold all the deserialized vectors
        _vectors = []

        # While there is data in the file, load it
        while True:
            try:
                # Load the next pickled object from the file
                _data = pickle.load(file)
                # Append the loaded data to the list
                _vectors.append(_data)
            except EOFError:
                # An EOFError is raised when there is no more data to read
                break

    _sample_ids = _vectors[0]
    _snp_ids = _vectors[1]
    _block_ids = _vectors[2]
    _block2gtype = _vectors[3]
    _mat_vec = _vectors[4]
    _mat_shape = (len(_snp_ids), len(_sample_ids))

    # Reshape the matrix to the correct shape
    vcf_mat = np.reshape(_mat_vec, _mat_shape).transpose()

    return {
        'gt_mat': vcf_mat,
        'block2gtype': _block2gtype,
        'sample_ids': _sample_ids,
        'snp_ids': _snp_ids,
        'block_ids': _block_ids,
    }


def train_model(
        model,
        dataloader_train,
        dataloader_val,
        es_patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        devices: Union[list[int], str, int] = 'auto',
        accelerator: str = 'auto',
        in_dev: bool = False,
    ):
    """
    Fit the model.
    """
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices

    callback_es = EarlyStopping(
        monitor='val_loss',
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename='best-model-{epoch:04d}-{val_loss:.4f}',
        monitor='val_loss',
    )

    logger_tr = TensorBoardLogger(
        save_dir=log_dir,
        name='',
    )

    trainer = Trainer(
        fast_dev_run=in_dev,
        logger=logger_tr,
        log_every_n_steps=1,
        # precision='16-mixed',
        devices=avail_dev,
        accelerator=accelerator,
        max_epochs=max_epochs,
        min_epochs=min_epochs,
        callbacks=[callback_es, callback_ckpt],
        num_sanity_val_steps=0,
        default_root_dir=log_dir,
    )
    
    trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    return callback_ckpt.best_model_score.item()


class CollectFitLog:
    def __init__(self, dir_log: str):
        """
        Collect training logs from optuna db files and ckpt files.
        """
        self.dir_log = dir_log
        if os.path.exists(self.dir_log) == False:
            raise ValueError(f'Directory {self.dir_log} does not exist.')
    
    def collect(self) -> Dict[str, pd.DataFrame]:
        """
        Collect training logs from optuna db files and ckpt files.
        """
        best_trials_df, all_ckpt = self.collect_ckpt()
        optuna_best_inners_df = self.collect_optuna_db()

        # Merge the two dataframes on the 'x_outer' and 'x_inner' columns
        logs_df = pd.merge(optuna_best_inners_df, best_trials_df, on=['x_outer', 'x_inner'], how='left')
        # Remove the 'val_loss' column from the merged dataframe
        logs_df = logs_df.drop(columns=['val_loss'])
        # Rename 'min_loss' column to 'val_loss'
        logs_df = logs_df.rename(columns={'min_loss': 'val_loss'})

        best_inners_df = logs_df.loc[logs_df.groupby(['x_outer'])['val_loss'].idxmin()]

        print("\nFound model logs:")
        print(logs_df)
        print("\nBest inner folds:")
        print(best_inners_df)

        return {'logs': logs_df, 'best_inners': best_inners_df}
    
    def collect_ckpt(self):
        """
        Collect info of ckpt files.
        """
        paths_ckpt = self.search_ckpt()

        # Pick ids of outer and inner folds, val_loss and version from ckpt file paths
        val_loss_values = [float(os.path.basename(path_x).split('-')[3].split('=')[1].split('.ckpt')[0]) for path_x in paths_ckpt]
        trial_tags = [path_x.split('/')[-2] for path_x in paths_ckpt]
        study_tags = [path_x.split('/')[-4].split('_')[-1] for path_x in paths_ckpt]
        ncv_inner_x = [int(path_x.split('/')[-3].split('_')[-1]) for path_x in paths_ckpt]
        ncv_outer_x = [int(path_x.split('/')[-3].split('_')[-2]) for path_x in paths_ckpt]

        # Create a dataframe with the above values
        ckpt_df = pd.DataFrame({'x_outer': ncv_outer_x, 'x_inner': ncv_inner_x, 'val_loss': val_loss_values, 'trial_tag': trial_tags, 'study_tag': study_tags, 'path_ckpt': paths_ckpt})

        # Pick the best model based on val_loss between the trials of the same outer and inner fold
        best_trials_df = ckpt_df.loc[ckpt_df.groupby(['x_outer', 'x_inner'])['val_loss'].idxmin()]

        return best_trials_df, ckpt_df

    def collect_optuna_db(self):
        """
        Collect info of optuna db files.
        """
        # Find all optuna db files in the directory `dir_log` and its subdirectories
        paths_optuna_db = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(self.dir_log)
                    for f in files if f.endswith('.db')
        ]
        paths_optuna_db.sort()
        print(f'Found {len(paths_optuna_db)} optuna db files\n')
        
        # Read optuna db files and store the results in a dataframe
        studies_dicts = [self.read_optuna_db(path_optuna_db) for path_optuna_db in paths_optuna_db]
        studies_df = pd.DataFrame(studies_dicts)
        
        return studies_df

    def read_optuna_db(self, path_optuna_db: str) -> Dict[str, Any]:
        loaded_study = optuna.load_study(study_name=None, storage=f"sqlite:///{path_optuna_db}")
        study_name = loaded_study.study_name
        min_loss = loaded_study.best_value
        # trials_df = loaded_study.trials_dataframe()

        frag_name = study_name.split('_')
        assert len(frag_name) > 4, 'Study name is not in the expected format.'
        x_outer = int(frag_name[2])
        x_inner = int(frag_name[3])
        x_time = frag_name[4]
        return {'study_name': study_name, 'x_outer': x_outer, 'x_inner': x_inner, 'min_loss': min_loss, 'x_time': x_time}
    
    def search_ckpt(self):
        """
        Search checkpoints in the directory and its subdirectories.
        """
        paths_ckpt = [os.path.join(dirpath, f)
                    for dirpath, dirnames, files in os.walk(self.dir_log)
                    for f in files if f.endswith('.ckpt')]
        paths_ckpt.sort()
        print(f'Found {len(paths_ckpt)} checkpoints.\n')
        return paths_ckpt
    
    def remove_inferior_models(self):
        """
        Remove inferior models based on the collected result table.
        """
        best_trials, all_trials = self.collect_ckpt()
        n_all_ckpt = len(all_trials)
        n_removed_models = 0
        for _x in range(n_all_ckpt):
            # Check if all_trials['trial_tag'][_x] is in best_trials['trial_tag']
            if best_trials['trial_tag'].str.contains(all_trials['trial_tag'][_x]).any():
                continue
            else:
                os.remove(all_trials['path_ckpt'][_x])
                print(f"Removed {all_trials['path_ckpt'][_x]}")
                n_removed_models += 1
        print(f"Removed {n_removed_models} inferior models.")
        return None


def rm_old_ckpt(ckpt_dir: str, rmALL: bool = False):
    """
    Remove checkpoints from a versions directory.
    """
    if not os.path.isdir(ckpt_dir):
        print("Error: {} is not a directory".format(ckpt_dir))
        exit(1)
    
    version_dirs = os.listdir(ckpt_dir)
    version_ids = [int(v.split("_")[1]) for v in version_dirs]
    # Get sortperm of version_ids
    sortperm = sorted(range(len(version_ids)), key=lambda k: version_ids[k])

    if rmALL:
        version_dirs = [version_dirs[i] for i in sortperm]
        print("Remove all versions")
    else:
        version_dirs = [version_dirs[i] for i in sortperm[:-2]]
        print("Remove versions from {} to {}".format(version_dirs[0], version_dirs[-1]))
    
    for dir_version in version_dirs:
        if os.path.isdir(os.path.join(ckpt_dir, dir_version)):
            for file_name in os.listdir(os.path.join(ckpt_dir, dir_version, "checkpoints")):
                if file_name.endswith(".ckpt"):

                    os.remove(os.path.join(ckpt_dir, dir_version, "checkpoints", file_name))
                    print("Remove {} in {}".format(file_name, os.path.join(ckpt_dir, dir_version, "checkpoints")))

    print("Done!")

r"""
Some universal functions.
"""
import os
import time
import random
import string
import numpy as np
import polars as pl
import pickle
import gzip
import optuna
from typing import Any, List, Dict, Optional, Sequence, Union
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
# from sklearn.feature_selection import VarianceThreshold, f_classif
# from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from lightning import Trainer, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.accelerators.cuda import find_usable_cuda_devices
from torch.cuda import device_count
from multiprocessing import cpu_count
import frn.constants as MC


def get_avail_cpu_count(target_n: int) -> int:
    total_n = cpu_count()
    n_cpu = target_n
    if target_n <= 0:
        n_cpu = total_n
    else:
        n_cpu = min(target_n, total_n)
    return n_cpu

def get_avail_nvgpu(devices: Union[list[int], str, int] = MC.default.devices):
    if type(devices) == int and device_count() > 0:
        avail_dev = find_usable_cuda_devices(devices)
    elif devices == 'auto' and device_count() > 0:
        avail_dev = find_usable_cuda_devices()
    else:
        avail_dev = devices
    return avail_dev

def get_map_location(map_loc: Optional[str] = None):
    if map_loc is None:
        if device_count() > 0:
            which_dev = find_usable_cuda_devices(1)
            if len(which_dev) == 0:
                return 'cpu'
            else:
                return f'cuda:{which_dev[0]}'
        else:
            return 'cpu'
    else:
        return map_loc


def time_string() -> str:
    _time_str = time.strftime(MC.default.time_format, time.localtime())
    return _time_str

def random_string(length: int = 7) -> str:
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def idx_convert(indices: List[int], onehot_bits: int = MC.default.snp_onehot_bits) -> List[int]:
    """
    Convert the indices to the corresponding indices in the one-hot vector.
    """
    converted_indices = np.array(indices)[:, np.newaxis] * onehot_bits + np.arange(onehot_bits)
    converted_indices = np.sort(converted_indices.flatten()).tolist()
    return converted_indices


def intersect_lists(lists: List[List[Any]], get_indices: bool = True, to_sorted: bool = True):
    """
    Find the shared elements between multiple lists.
    """
    if len(lists) == 0:
        raise ValueError("The list of lists is empty.")
    elif len(lists) == 1:
        shared = lists[0]
    else:
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

def read_labels(path_label: str, col2use: Optional[List[Any]] = None):
    """
    Read labels from a csv file.
    
    If `col2use` is `List[int]`, its numbers are the indices **(1-based)** of the columns to be used.
    """
    label_df = read_omics(path_label)
    sample_ids = label_df.select(MC.dkey.id).to_series().to_list()

    if col2use is not None:
        if type(col2use[0]) == str:
            label_df = label_df.select([MC.dkey.id] + col2use)
        elif type(col2use[0]) == int:
            label_df = label_df[[0]+col2use,:]
        else:
            raise ValueError("col2use must be either a list of strings or a list of integers.")
    dim_model_output = len(label_df.columns) - 1
    return label_df, dim_model_output, sample_ids


def read_omics(data_path: str):
    """
    Read omics data from various formats.
    """
    # Check if the path is a file or a folder
    if os.path.isdir(data_path):
        raise ValueError(f"The path {data_path} is a folder. Please provide a path to a single file.")
    
    # Check the file extension
    file_ext = os.path.splitext(data_path)[-1]
    if len(file_ext) < 2:
        raise ValueError("The file extension is empty.")
    else:
        file_ext = file_ext.lower()
    
    match file_ext:
        case '.csv':
            _data = pl.read_csv(data_path, schema_overrides={MC.dkey.id: pl.Utf8})
        case '.parquet':
            _data = pl.read_parquet(data_path)
        case '.gz':
            if data_path.endswith('.pkl.gz'):
                snp_data_dict = read_pkl_gv(data_path)
                snp_matrix = snp_data_dict[MC.dkey.genotype_matrix]
                snp_sample_ids = snp_data_dict[MC.dkey.sample_ids]
                snp_ids = snp_data_dict[MC.dkey.snp_ids]
                # snp_block_ids = snp_data_dict['block_ids']
                _tmp_snp_df = pl.DataFrame(data=snp_matrix, schema=snp_ids)
                # print(type(snp_sample_ids))
                # print(snp_sample_ids)
                _tmp_id = pl.DataFrame({MC.dkey.id: snp_sample_ids})
                # print(_tmp_id.shape, _tmp_snp_df.shape)
                _data = _tmp_id.hstack(_tmp_snp_df)
            else:
                raise ValueError("The file extension is not supported.")
        case _:
            raise ValueError("The file extension is not supported.")
    
    return _data


def read_omics_xoxi(
        data_path: str,
        which_outer_test: int,
        which_inner_val: int,
        trnvaltst: str = MC.abbr_train,
        file_ext: Optional[str] = None,
        prefix: Optional[str] = None,
    ):
    """
    Read processed data from a directory.
    """
    if not os.path.isdir(data_path):
        raise ValueError(f"The path {data_path} is not a directory.")
    if file_ext is None:
        fname_ext = MC.fname.data_ext
    else:
        fname_ext = file_ext
    
    # Walk through the directory and find all files with the specified pattern
    if prefix is None:
        files_found = [os.path.join(dir_path, f) for dir_path, _, files in os.walk(data_path) for f in files if f.endswith(fname_ext)]
    else:
        files_found = [os.path.join(dir_path, f) for dir_path, _, files in os.walk(data_path) for f in files if f.startswith(prefix) and f.endswith(fname_ext)]
    
    if len(files_found) == 0:
        raise ValueError(f"No files found with the specified pattern in {data_path}.")

    # Search for the specific file name
    for file_path in files_found:
        _tmp_name = os.path.basename(file_path)
        _tmp_name = os.path.splitext(_tmp_name)[0]
        _tmp_name_parts = _tmp_name.split('_')
        if _tmp_name_parts[-1] == trnvaltst:
            if _tmp_name_parts[-2] == str(which_inner_val) and _tmp_name_parts[-3] == str(which_outer_test):
                return read_omics(file_path)
    
    raise ValueError(f"No file found with the specified pattern in {data_path}.")


class ProcOnTrainSet:
    """
    Process all data points based on the training set.

    How to use:
    - Initialize the class.
    - Call the method `pr_xxxxx` to process the data.
    - Call the method `save_processors` to save the processors (as a dict) to a pickle file.
    """
    def __init__(self, df_in: pl.DataFrame, ind_for_fit: Optional[List[Any]], n_feat2save: Optional[int] = None, df_labels: Optional[pl.DataFrame] = None):
        self.n_feat2save = n_feat2save
        self._df = df_in
        if df_labels is not None:
            self._labels = df_labels

        if ind_for_fit is not None:
            self._df_part = self._df[ind_for_fit,:]
            if df_labels is not None:
                self._labels_part = self._labels[ind_for_fit,:]
                
        self.preprocessors = {}
    
    def keep_preprocessors(self, x_value):
        """
        The key (int, ***0-based***) is automatically generated by the order of the data processor,
        for the reproduction of data processing steps.
        
        - `x_value`: the processor to be kept.
        """
        x_order = len(self.preprocessors)
        self.preprocessors[x_order] = x_value

    def save_preprocessors(self, dir_save_processors: str, file_name: Optional[str] = None):
        if file_name is None:
            fname_preprocessors = MC.fname.preprocessors
        else:
            fname_preprocessors = file_name
        os.makedirs(dir_save_processors, exist_ok=True)
        path_save_processors = os.path.join(dir_save_processors, fname_preprocessors)
        if os.path.exists(path_save_processors):
            raise FileExistsError(f"The file {path_save_processors} already exists.")
        
        if len(self.preprocessors) < 1:
            raise Warning("No data processor is saved.")
        else:
            with open(path_save_processors, "wb") as f:
                pickle.dump(self.preprocessors, f)
            print(f"The data processors have been saved to: {path_save_processors}")
    
    def load_run_preprocessors(self, dir_save_processors: str, file_name: str):
        path_processors = os.path.join(dir_save_processors, file_name)
        with open(path_processors, "rb") as f:
            processors_dict = pickle.load(f)
        # Run the processors
        _tmp_df = self._df.drop(MC.dkey.id).to_numpy()
        for i in range(len(processors_dict.keys())):
            _tmp_df = processors_dict[i].transform(_tmp_df)
        _tmp_df = pl.DataFrame(_tmp_df, schema=self._df.columns[1:])
        _tmp_df = pl.DataFrame({MC.dkey.id: self._df[MC.dkey.id]}).hstack(_tmp_df)
        self._df = _tmp_df
    
    def general_preprocessor(self, _processor):
        try:
            _processor.fit(self._df_part.drop(MC.dkey.id).to_numpy())
            df_o = _processor.transform(self._df.drop(MC.dkey.id).to_numpy())
            df_o = pl.DataFrame(df_o, schema=self._df.columns[1:])
            df_o = pl.DataFrame({MC.dkey.id: self._df[MC.dkey.id]}).hstack(df_o)
        except:
            _processor.fit(self._df_part, self._labels_part)
            df_o = _processor.transform(self._df)
        
        self._df = df_o
        self.keep_preprocessors(_processor)
    
    def pr_minmax(self):
        _processor = MinMaxScaler()
        self.general_preprocessor(_processor)
    
    def pr_zscore(self):
        _processor = StandardScaler()
        self.general_preprocessor(_processor)
        
    def pr_impute(self, strategy: str = "mean"):
        _imputer = SimpleImputer(strategy=strategy)
        self.general_preprocessor(_imputer)
    
    def pr_rf(self, random_states: List[int], n_estimators: int = MC.default.n_estimators):
        if not hasattr(self, "_labels") or self.n_feat2save is None:
            raise ValueError("The labels are not provided.")
        
        _selector = RFSelector(self.n_feat2save, random_states, n_estimators, n_jobs=MC.default.n_jobs_rf)
        self.general_preprocessor(_selector)


class RFSelector:
    def __init__(self, n_feat2save: int, random_states: List[int], n_estimators: int, n_jobs: int = MC.default.n_jobs_rf):
        self.n_feat2save = n_feat2save
        self.random_states = random_states
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.processors = {}
    
    def fit(self, omics_df: pl.DataFrame, labels_df: pl.DataFrame):
        _omics_np = omics_df.drop(MC.dkey.id).to_numpy()
        _labels_np = labels_df.drop(MC.dkey.id).to_numpy()
        _feat_imp = np.zeros(shape=(len(self.random_states), _omics_np.shape[1]))
        print(f"Starting to fit RF for {len(self.random_states)} random states...")
        for i in range(len(self.random_states)):
            _feat_imp[i,:] = self.fit_1(_omics_np, _labels_np, self.random_states[i])
            print(f"Finished {i+1}/{len(self.random_states)}")
        _feat_imp_mean = np.mean(_feat_imp, axis=0)
        _feat_imp_mean_sorted = np.argsort(_feat_imp_mean)[::-1]
        if self.n_feat2save <= _omics_np.shape[1]:
            _feat_to_save = _feat_imp_mean_sorted[:self.n_feat2save]
        self.colname_to_save = omics_df.drop(MC.dkey.id).columns[_feat_to_save]
    
    def transform(self, X_df: pl.DataFrame):
        _selected = X_df.select(self.colname_to_save)
        df_o = pl.DataFrame({MC.dkey.id: X_df[MC.dkey.id]}).hstack(_selected)
        return df_o

    def keep_preprocessor(self, x_processor):
        """
        The key (int, ***0-based***) is automatically generated by the order of the data processor,
        for the reproduction of data processing steps.
        
        - `x_processor`: the processor to be kept.
        """
        x_order = len(self.processors)
        self.processors[x_order] = x_processor

    def fit_1(self, X: np.ndarray, y: np.ndarray, random_state: int):
        _processor = RandomForestRegressor(n_estimators=self.n_estimators, n_jobs=self.n_jobs, random_state=random_state)
        _processor.fit(X, y)
        self.keep_preprocessor(_processor)
        return _processor.feature_importances_


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


def onehot_encode_snp_mat(
        snp_matrix: np.ndarray,
        onehot_bits: Optional[int] = None,
        genes_snps: Optional[List[List[int]]] = None,
    ):
    """
    One-hot encode the SNP matrix.
    """
    if onehot_bits is None:
        len_onehot = MC.default.snp_onehot_bits
    else:
        len_onehot = onehot_bits
    
    if genes_snps is not None:
        num_genes = len(genes_snps)
        indices_snp = []
        for i_gene in range(num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_onehot))
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_onehot + 1)[snp_vec][:, 1:].reshape(-1)
            snp_vec_genes = [snp_vec[indices_snp[i_gene]].astype(np.float32) for i_gene in range(num_genes)]
            snp_data.append(snp_vec_genes)
    else:
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_onehot + 1)[snp_vec][:, 1:].reshape(-1).astype(np.float32)
            snp_data.append(snp_vec)
    snp_data_np = np.array(snp_data)
    return snp_data_np


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
        MC.dkey.genotype_matrix: vcf_mat,
        MC.dkey.gblock2gtype: _block2gtype,
        MC.dkey.sample_ids: _sample_ids,
        MC.dkey.snp_ids: _snp_ids,
        MC.dkey.gblock_ids: _block_ids,
    }


def train_model(
        model,
        datamodule: LightningDataModule,
        es_patience: int,
        max_epochs: int,
        min_epochs: int,
        log_dir: str,
        devices: Union[list[int], str, int] = MC.default.devices,
        accelerator: str = MC.default.accelerator,
        in_dev: bool = False,
    ):
    """
    Fit the model.
    """
    avail_dev = get_avail_nvgpu(devices)

    callback_es = EarlyStopping(
        monitor=MC.title_val_loss,
        patience=es_patience,
        mode='min',
        verbose=True,
    )
    callback_ckpt = ModelCheckpoint(
        dirpath=log_dir,
        filename=MC.default.ckpt_fname_format,
        monitor=MC.title_val_loss,
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
    
    trainer.fit(model=model, datamodule=datamodule)

    if callback_ckpt.best_model_score is not None:
        best_score = callback_ckpt.best_model_score.item()
    else:
        best_score = None

    # trainer.test(model=model, dataloaders=datamodule)

    return best_score


class CollectFitLog:
    def __init__(self, dir_log: str):
        """
        Collect training logs from optuna db files and ckpt files.
        """
        self.dir_log = dir_log
        if not os.path.exists(self.dir_log):
            raise ValueError(f'Directory {self.dir_log} does not exist.')
    
    def get_df_csv(self, dir_output: str, overwrite_collected_log: bool = False):
        """
        Collect trained models for each fold in nested cross-validation.
        """
        collected_logs = self.collect()

        models_bv = collected_logs[MC.dkey.best_trials]
        path_log_best_trials = os.path.join(dir_output, '_log_best_trials' + '.csv')
        if os.path.exists(path_log_best_trials) and not overwrite_collected_log:
            models_bv = pl.read_csv(path_log_best_trials)
        else:
            models_bv.write_csv(path_log_best_trials)

        models_bi = collected_logs[MC.dkey.best_inner_folds]
        path_log_best_inners = os.path.join(dir_output, '_log_best_inners' + '.csv')
        if os.path.exists(path_log_best_inners) and not overwrite_collected_log:
            models_bi = pl.read_csv(path_log_best_inners)
        else:
            models_bi.write_csv(path_log_best_inners)
        
        return models_bv, models_bi
    
    def collect(self) -> Dict[str, pl.DataFrame]:
        """
        Collect training logs from optuna db files and ckpt files.
        """
        best_trials_df, all_ckpt = self.collect_ckpt()
        optuna_best_inners_df = self.collect_optuna_db()

        # Merge the two dataframes on the MC.dkey.which_outer and MC.dkey.which_inner columns
        logs_df = optuna_best_inners_df.join(best_trials_df, on=[MC.dkey.which_outer, MC.dkey.which_inner], how='left')
        # Remove the MC.title_val_loss column from the merged dataframe
        logs_df = logs_df.drop(MC.title_val_loss)
        # Rename 'min_loss' column to MC.title_val_loss
        logs_df = logs_df.rename({'min_loss': MC.title_val_loss})

        best_inners_df = logs_df.group_by(MC.dkey.which_outer).agg(pl.col(MC.title_val_loss).min()).join(logs_df, on=[MC.dkey.which_outer, MC.title_val_loss], how='left')

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
        ckpt_df = pl.DataFrame({MC.dkey.which_outer: ncv_outer_x, MC.dkey.which_inner: ncv_inner_x, MC.title_val_loss: val_loss_values, MC.dkey.trial_tag: trial_tags, MC.dkey.study_tag: study_tags, MC.dkey.ckpt_path: paths_ckpt})

        # Pick the best model based on val_loss between the trials of the same outer and inner fold
        best_trials_df = ckpt_df.group_by([MC.dkey.which_outer, MC.dkey.which_inner]).agg([pl.col(MC.title_val_loss).min()]).join(ckpt_df, on=[MC.dkey.which_outer, MC.dkey.which_inner, MC.title_val_loss], how='left')

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
        studies_df = pl.DataFrame(studies_dicts)
        
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
        return {MC.dkey.study_name: study_name, MC.dkey.which_outer: x_outer, MC.dkey.which_inner: x_inner, MC.dkey.min_loss: min_loss, MC.dkey.time_str: x_time}
    
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
            # Check if all_trials[MC.dkey.trial_tag][_x] is in best_trials[MC.dkey.trial_tag]
            if best_trials[MC.dkey.trial_tag].str.contains(all_trials[MC.dkey.trial_tag][_x]).any():
                continue
            else:
                os.remove(all_trials[MC.dkey.ckpt_path][_x])
                print(f"Removed {all_trials[MC.dkey.ckpt_path][_x]}")
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

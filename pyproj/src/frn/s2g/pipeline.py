r"""
The pipeline for converting SNPs to genome blocks representations.
"""
import os
from typing import Optional, Tuple, Union, List, Dict, Any
import time
import numpy as np
import polars as pl
import optuna
from lightning import Trainer
from frn.s2g.model import SNPReductionNet, SNP2GB
from frn.utils.uni import get_avail_nvgpu, train_model, CollectFitLog, random_string, read_pkl_gv, time_string
from frn.utils.data_ncv import MyDataModule4Train, MyDataModule4Uni


class SNP2GBFit:
    """
    SNP-to-genome-block model training with hyperparameter optimization.
    """
    def __init__(
            self,
            log_dir: str,
            log_name: str,
            litdata_dir: str,
            which_outer_testset: int,
            which_inner_valset: int,
            regression: bool,
            dense_layers_hidden_dims: List[int],
            len_one_hot_vec: int = 10,
            devices: Union[List[int], str, int] = 'auto',
            accelerator: str = 'auto',
            n_jobs: int = 1,
            learning_rate: float = 1e-3,
            patience: int = 10,
            max_epochs: int = 1000,
            min_epochs: int = 20,
            batch_size: int = 16,
        ):
        """
        Initialize SNP2GBTrain.
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.regression = regression
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.len_one_hot_vec = len_one_hot_vec
        self.blocks_gt = read_pkl_gv(os.path.join(litdata_dir, 'genotypes.pkl.gz'))['block2gtype']
        self.model_out_dim = pl.read_csv(os.path.join(litdata_dir, "model_output_dim.csv"), has_header=True)[0,0]

        self.datamodule = MyDataModule4Train(litdata_dir, which_outer_testset, which_inner_valset, batch_size, n_jobs)
        self.datamodule.setup()

        self.hparams = self.hparams_fit(
            learning_rate=learning_rate,
            patience=patience,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            batch_size=batch_size,
            dense_layers_hidden_dims=dense_layers_hidden_dims,
        )
    
    def hparams_fit(
            self,
            learning_rate: float,
            patience: int,
            max_epochs: int,
            min_epochs: int,
            batch_size: int,
            dense_layers_hidden_dims: List[int],
        ) -> Dict[str, Any]:
        """
        Generate a dictionary of hyperparameters for SNP2GB model training.
        """
        hparams = {
            'learning_rate': learning_rate,
            'patience': patience,
            'max_epochs': max_epochs,
            'min_epochs': min_epochs,
            'batch_size': batch_size,
            'dense_layers_hidden_dims': dense_layers_hidden_dims,
        }
        return hparams

    def snp2gb_fit(
            self,
            hparams: Dict[str, Any],
            devices: Union[List[int], str, int],
            accelerator: str,
        ):
        """
        Train SNP2GB model with given hyperparameters.
        """
        _model = SNPReductionNet(
            output_dim=self.model_out_dim,
            blocks_gt=self.blocks_gt,
            len_one_hot_vec=self.len_one_hot_vec,
            dense_layers_hidden_dims=hparams['dense_layers_hidden_dims'],
            learning_rate=hparams['learning_rate'],
            regression=self.regression,
        )
        # Unique tag for the experiment
        log_dir_uniq_model = os.path.join(self.log_dir, self.log_name, random_string())

        val_loss_min = train_model(
            model=_model,
            dataloader_train=self.datamodule.train_dataloader(),
            dataloader_val=self.datamodule.val_dataloader(),
            es_patience=hparams['patience'],
            max_epochs=hparams['max_epochs'],
            min_epochs=hparams['min_epochs'],
            log_dir=log_dir_uniq_model,
            devices=devices,
            accelerator=accelerator,
        )
        if val_loss_min is None:
            raise ValueError('\nTraining failed.\n')
        return val_loss_min
    
    def manual_fit(self):
        """
        Train SNP2GB model with manually set hyperparameters.
        """
        val_loss_min = self.snp2gb_fit(
            hparams=self.hparams,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        return val_loss_min
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for SNP2GB model hyperparameter optimization.
        """
        print('Trial number:', trial.number)
        if self.n_jobs > 1:
            time_delay = (trial.number + self.n_jobs) % self.n_jobs * 11.7
            time.sleep(time_delay)
        
        lr = trial.suggest_categorical('lr', [1e-3, 1e-4, 1e-5, 1e-6])
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        hparams_trial = self.hparams.copy()
        hparams_trial['learning_rate'] = lr
        hparams_trial['batch_size'] = batch_size
        
        val_loss_min = self.snp2gb_fit(
            hparams = hparams_trial,
            devices = self.devices,
            accelerator=self.accelerator,
        )
        
        return val_loss_min
    
    def optimize(
            self,
            n_trials: Optional[int] = None,
            storage: str = 'sqlite:///optuna_s2g.db',
        ):
        """
        Hyperparameters optimization for SNP2GB model.
        """
        time_str = time_string()

        study = optuna.create_study(
            storage = storage,
            study_name = self.log_name + '_' + time_str,
            load_if_exists = True,
            direction = 'minimize',
        )
        study.optimize(self.objective, n_jobs=self.n_jobs, n_trials=n_trials, gc_after_trial=True)


class SNP2GBFitPipe:
    r"""
    SNP2GB model pipeline.
    Hyperparameters are optimized for each fold in nested cross-validation.
    The best model for each fold is used to convert SNPs to genome blocks.
    """
    def __init__(
            self,
            litdata_dir: str,
            list_ncv: List[List[int]],
            log_dir: str,
            regression: bool,
            devices: Union[List[int], str, int] = 'auto',
            accelerator: str = 'auto',
            n_jobs: int = 1,
            n_trials: Optional[int] = 10,
            dense_layers_hidden_dims: Optional[List[int]] = None,
            len_onehot_snp: int = 10,
        ):
        """
        Initialize SNP2GB pipeline.

        Parameters:
        - `list_ncv`: List of nested cross-validation folds. e.g., `[[0,0], [0,1], [9,4]]`.
        - `len_onehot_snp`: Length of the one-hot vector for each SNP.
        - `dense_layers_hidden_dims`: List of hidden dimensions for the dense layers.
        """
        # Unique tag for the training log directory
        rand_str = random_string()
        time_str = time_string()
        tag_str = time_str + '_' + rand_str
        self.uniq_logdir = os.path.join(log_dir, 'train_' + tag_str)
        if not os.path.exists(self.uniq_logdir):
            os.makedirs(self.uniq_logdir)

        self.litdata_dir = litdata_dir
        self.list_ncv = list_ncv
        self.n_slice = len(list_ncv)
        
        if dense_layers_hidden_dims is None:
            self.dense_layers_hidden_dims = [1024, 512, 128]
        else:
            self.dense_layers_hidden_dims = dense_layers_hidden_dims
        
        self.regression = regression
        self.len_onehot_snp = len_onehot_snp
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.n_trials = n_trials
    
    def train_pipeline(self):
        """
        Train SNP2GB model for each fold in nested cross-validation.
        """
        # Storage for optuna trials in self.log_dir
        path_storage = 'sqlite:///' + self.uniq_logdir + '/optuna_s2g' + '.db'
        
        print(f"\nNumber of data slices to train: {self.n_slice}\n")
        log_names = []
        for i in range(self.n_slice):
            log_names.append(f'run_ncv_{self.list_ncv[i][0]}_{self.list_ncv[i][1]}')
        
        # Train SNP2GB model for each fold in nested cross-validation
        if self.n_slice == 1:
            snp2gb_train_x = SNP2GBFit(
                log_dir = self.uniq_logdir,
                log_name = log_names[0],
                litdata_dir = self.litdata_dir,
                which_outer_testset = self.list_ncv[0][0],
                which_inner_valset = self.list_ncv[0][1],
                regression = self.regression,
                dense_layers_hidden_dims = self.dense_layers_hidden_dims,
                len_one_hot_vec = self.len_onehot_snp,
                devices = self.devices,
                accelerator=self.accelerator,
                n_jobs=self.n_jobs,
            )
            snp2gb_train_x.optimize(n_trials=self.n_trials, storage=path_storage)
        else:
            for xfold in range(self.n_slice):
                snp2gb_train_x = SNP2GBFit(
                    log_dir = self.uniq_logdir,
                    log_name = log_names[xfold],
                    litdata_dir = self.litdata_dir,
                    which_outer_testset = self.list_ncv[xfold][0],
                    which_inner_valset = self.list_ncv[xfold][1],
                    regression = self.regression,
                    dense_layers_hidden_dims = self.dense_layers_hidden_dims,
                    len_one_hot_vec = self.len_onehot_snp,
                    devices = self.devices,
                    accelerator=self.accelerator,
                    n_jobs=self.n_jobs,
                )
                snp2gb_train_x.optimize(n_trials=self.n_trials, storage=path_storage)


def execute_s2g(
        dir_litdata: str,
        path_gtype_pkl: str,
        path_pretrained_model: str,
        dir_log_predict: str = os.getcwd(),
        len_one_hot_vec: int = 10,
        batch_size: int = 32,
        accelerator: str = 'auto',
    ):
    """
    Run the SNP2GB model for independent test / prediction.
    """
    g_data_dict = read_pkl_gv(path_gtype_pkl)
    datamodule_s2g = MyDataModule4Uni(dir_litdata, batch_size)
    datamodule_s2g.setup()

    model4gene = SNP2GB(
        path_pretrained_model=path_pretrained_model,
        blocks_gt=g_data_dict['block2gtype'],
        len_one_hot_vec=len_one_hot_vec,
    )

    avail_dev = get_avail_nvgpu()

    trainer = Trainer(accelerator=accelerator, devices=avail_dev, default_root_dir=dir_log_predict, logger=False)
    
    predictions = trainer.predict(model=model4gene, datamodule=datamodule_s2g)

    pred_array = np.concatenate(np.array(predictions), axis=0)

    # Rename index to sample_ids
    # - Prepare sample ids
    sample_ids = []
    for batch in datamodule_s2g.dataloader_xxx:
        sample_ids.extend(batch['id'])
    print(f'Number of samples: {len(sample_ids)}')
    print(sample_ids)
    assert len(sample_ids) == len(pred_array)
    assert len(g_data_dict['block_ids']) == pred_array.shape[1]
    
    # Prepare prediction dataframe
    pred_df = pl.DataFrame(pred_array, schema=g_data_dict['block_ids'])
    # Add a column of sample ids
    df_ids = pl.DataFrame(sample_ids, schema=['ID'])
    pred_df = df_ids.hstack(pred_df)

    return pred_df


class SNP2GBTransPipe:
    """
    1. Collect trained models for each fold in nested cross-validation.
    2. Transform SNP features to genome block features.
    """
    def __init__(
            self,
            dir_log: str,
            dir_output: str,
            overwrite_collected_log: bool = False,
    ):
        self.dir_log = dir_log
        self.dir_output = dir_output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.overwrite_collected_log = overwrite_collected_log

    def collect_models(self):
        """
        Collect trained models for each fold in nested cross-validation.
        """
        collector = CollectFitLog(self.dir_log)
        models_bv, models_bi = collector.get_df_csv(self.dir_output, self.overwrite_collected_log)
        self.models_bv = models_bv
        self.models_bi = models_bi

    def convert_snp(
            self,
            dir_litdata: str,
            list_ncv: Optional[List[List[int]]] = None,
            len_one_hot_vec: int = 10,
            accelerator: str = 'auto',
            batch_size: int = 32,
            n_workers: int = 0,
        ):
        """
        Convert SNPs to genome blocks features using the best model for each fold in nested cross-validation.

        If `list_ncv` is `None`, the best model overall is used.
        Otherwise, the best model for each fold in `list_ncv` is used.
        """
        if not hasattr(self,'models_bv'):
            self.collect_models()
        
        path_gtype_pkl = os.path.join(dir_litdata, 'genotypes.pkl.gz')

        if list_ncv is None:
            # Take the best model's path overall by searching the line min `val_loss` in models_bi.
            path_best_model = self.models_bi.filter(pl.col('val_loss') == self.models_bi.select('val_loss').min()).select('path_ckpt')[0,0]
            output = execute_s2g(dir_litdata, path_gtype_pkl, path_best_model, self.dir_output, len_one_hot_vec, batch_size, accelerator)
            output.write_parquet(os.path.join(self.dir_output, 'snp2gb.parquet'))
            
            return None
        
        # For each inner fold
        for data_xx in list_ncv:
            x_outer, x_inner = data_xx
            path_o_pred_trn = os.path.join(self.dir_output, f'snp2gb_{x_outer}_{x_inner}_trn.parquet')
            path_o_pred_val = os.path.join(self.dir_output, f'snp2gb_{x_outer}_{x_inner}_val.parquet')
            path_o_pred_tst = os.path.join(self.dir_output, f'snp2gb_{x_outer}_{x_inner}_tst.parquet')
            # path_o_pred_test_bestinner = os.path.join(self.dir_output, f'snp2gb_{x_outer}_{x_inner}_tst_bestinner.parquet')

            path_mdl = self.models_bv.filter((pl.col('x_outer') == x_outer) & (pl.col('x_inner') == x_inner)).select('path_ckpt')[0,0]
            print(f'\nUsing model {path_mdl}\n')

            ncv_data = MyDataModule4Train(dir_litdata, x_outer, x_inner, batch_size, n_workers)
            dir_train, dir_valid, dir_test = ncv_data.get_dir_ncv_litdata()

            pred_trn = execute_s2g(dir_train, path_gtype_pkl, path_mdl, self.dir_output, len_one_hot_vec, batch_size, accelerator)
            pred_trn.write_parquet(path_o_pred_trn)
            pred_val = execute_s2g(dir_valid, path_gtype_pkl, path_mdl, self.dir_output, len_one_hot_vec, batch_size, accelerator)
            pred_val.write_parquet(path_o_pred_val)
            pred_tst = execute_s2g(dir_test, path_gtype_pkl, path_mdl, self.dir_output, len_one_hot_vec, batch_size, accelerator)
            pred_tst.write_parquet(path_o_pred_tst)

            # # Check if the model is the best for the current outer fold
            # if self.models_bi.loc[self.models_bi['x_outer'] == x_outer]['x_inner'].values[0] == x_inner:
            #     # Copy the predictions of test set
            #     shutil.copy(path_o_pred_tst, path_o_pred_test_bestinner)
        
        return None

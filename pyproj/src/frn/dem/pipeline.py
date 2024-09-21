r"""
DEM model training and hyperparameter optimization.
"""
import os
from typing import Optional, Union, List, Dict, Any
import time
import optuna
from lightning import Trainer
import numpy as np
import polars as pl
from frn.dem.model import DEMLTN
from frn.utils.uni import get_avail_nvgpu, get_map_location, train_model, CollectFitLog, random_string, time_string
from frn.utils.data_ncv import MyDataModule4Train, MyDataModule4Uni


class DEMFit:
    r"""
    DEM model training with hyperparameter optimization using Optuna.
    """
    def __init__(
            self,
            log_dir: str,
            log_name: str,
            litdata_dir: str,
            which_outer_testset: int,
            which_inner_valset: int,
            regression: bool,
            devices: Union[List[int], str, int] = 'auto',
            accelerator: str = 'auto',
            n_jobs: int = 1,
            n_heads: int = 4,
            n_encoders: int = 2,
            hidden_dim: int = 1024,
            learning_rate: float = 1e-5,
            dropout: float = 0.4,
            patience: int = 20,
            max_epochs: int = 1000,
            min_epochs: int = 20,
            batch_size: int = 16,
        ):
        """
        Initialize DEM model training with given hyperparameters and input/output paths.
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.model_out_dim = pl.read_csv(os.path.join(litdata_dir, "model_output_dim.csv"), has_header=True)[0,0]
        self.is_regression = regression
        self.datamodule = MyDataModule4Train(litdata_dir, which_outer_testset, which_inner_valset, batch_size, n_jobs)
        self.datamodule.setup()
        self.omics_dims = self.datamodule.read_omics_dims()

        self.hparams = self.hparams_fit(
            n_heads=n_heads,
            n_encoders=n_encoders,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            dropout=dropout,
            patience=patience,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            batch_size=batch_size,
        )
        
    def hparams_fit(
            self,
            n_heads: int,
            n_encoders: int,
            hidden_dim: int,
            learning_rate: float,
            dropout: float,
            patience: int,
            max_epochs: int,
            min_epochs: int,
            batch_size: int,
        ):
        """
        Generate a dictionary of hyperparameters for DEM model training.
        """
        hparams = {
            'n_heads': n_heads,
            'n_encoders': n_encoders,
            'hidden_dim': hidden_dim,
            'learning_rate': learning_rate,
            'dropout': dropout,
            'patience': patience,
            'max_epochs': max_epochs,
            'min_epochs': min_epochs,
            'batch_size': batch_size,
        }
        return hparams

    def dem_fit(
            self,
            hparams:Dict[str, Any],
            devices: Union[List[int], str, int],
            accelerator: str,
        ):
        """
        Train DEM model with given hyperparameters and input/output paths.
        """
        _model = DEMLTN(
            omics_dim=self.omics_dims,
            n_heads=hparams['n_heads'],
            n_encoders=hparams['n_encoders'],
            hidden_dim=hparams['hidden_dim'],
            output_dim=self.model_out_dim,
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate'],
            is_regression=self.is_regression,
        )
        
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
            raise ValueError("Training failed.")
        return val_loss_min

    def manual_train(self):
        """
        Train DEM model with manually specified hyperparameters.
        """
        val_loss_min = self.dem_fit(
            hparams = self.hparams,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        return val_loss_min

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for DEM model training with Optuna.
        """
        print("Trial number:", trial.number)
        if self.n_jobs > 1:
            time_delay = (trial.number + self.n_jobs) % self.n_jobs * 11.7
            time.sleep(time_delay)

        # Generate hyperparameters
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        n_heads = trial.suggest_categorical("n_heads", [1, 2, 4])
        n_encoders = trial.suggest_categorical("n_encoders", [1, 2, 4])
        hidden_dim = trial.suggest_categorical("hidden_dim", [512, 1024])
        dropout = trial.suggest_float("dropout", 0.0, 0.8, step=0.2)
        lr = trial.suggest_categorical("lr", [1e-7, 1e-6, 1e-5, 1e-4, 1e-3])

        # Update hyperparameters in DEMTrain object based on manual parameters in initialization
        hparams_tmp = self.hparams.copy()
        hparams_tmp['batch_size'] = batch_size
        hparams_tmp['n_heads'] = n_heads
        hparams_tmp['n_encoders'] = n_encoders
        hparams_tmp['hidden_dim'] = hidden_dim
        hparams_tmp['learning_rate'] = lr
        hparams_tmp['dropout'] = dropout
        
        val_loss_min = self.dem_fit(
            hparams = hparams_tmp,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        
        return val_loss_min

    def optimize(
            self,
            n_trials: Optional[int] = None,
            storage: str = "sqlite:///optuna_dem.db",
        ):
        """
        Optimize hyperparameters of DEM model with Optuna.
        """
        time_str = time_string()

        study = optuna.create_study(
            storage = storage,
            study_name = self.log_name + "_" + time_str,
            direction = "minimize",
            load_if_exists = True,
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs, gc_after_trial=True)


class DEMFitPipe:
    r"""
    DEM model training pipeline with hyperparameter optimization using Optuna.
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
        ):
        """
        Initialize DEM model training pipeline.
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
        
        self.regression = regression
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.n_trials = n_trials
            
    def train_pipeline(self):
        """
        Train DEM model for each fold in nested cross-validation.
        """
        # Storage for optuna trials in self.log_dir
        path_storage = 'sqlite:///' + self.uniq_logdir + '/optuna_dem' + '.db'
        
        print(f"\nNumber of data slices to train: {self.n_slice}\n")
        log_names = []
        for i in range(self.n_slice):
            log_names.append(f'run_ncv_{self.list_ncv[i][0]}_{self.list_ncv[i][1]}')
        
        # Train DEM model for each fold in nested cross-validation
        if self.n_slice == 1:
            dem_fit_ = DEMFit(
                log_dir=self.uniq_logdir,
                log_name=log_names[0],
                litdata_dir=self.litdata_dir,
                which_outer_testset=self.list_ncv[0][0],
                which_inner_valset=self.list_ncv[0][1],
                regression=self.regression,
                devices=self.devices,
                accelerator=self.accelerator,
                n_jobs=self.n_jobs,
            )
            dem_fit_.optimize(n_trials=self.n_trials, storage=path_storage)
        else:
            for xfold in range(self.n_slice):
                dem_fit_ = DEMFit(
                    log_dir=self.uniq_logdir,
                    log_name=log_names[xfold],
                    litdata_dir=self.litdata_dir,
                    which_outer_testset=self.list_ncv[xfold][0],
                    which_inner_valset=self.list_ncv[xfold][1],
                    regression=self.regression,
                    devices=self.devices,
                    accelerator=self.accelerator,
                    n_jobs=self.n_jobs,
                )
                dem_fit_.optimize(n_trials=self.n_trials, storage=path_storage)


class DEMPredict:
    r"""
    DEM model prediction for NCV or not.
    """
    def __init__(
            self,
            dir_fit_logs: str,
            dir_output: str,
            overwrite_collected_log: bool = False,
        ):
        """
        Initialize DEM model prediction.
        - map_location: `cuda:0` for GPU, `cpu` for CPU.
        """
        self.dir_logs = dir_fit_logs
        self.dir_output = dir_output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.overwrite_collected_log = overwrite_collected_log

    def runs(
            self,
            dir_litdata: str,
            list_ncv: Optional[List[List[int]]] = None,
            accelerator: str = 'auto',
            batch_size: int = 32,
            n_workers: int = 0,
        ):
        """
        Predict for NCV or not.
        """
        if not hasattr(self, 'models_bv'):
            self.collect_models()
        
        if list_ncv is None:
            # Take the best model's path overall by searching the line min `val_loss` in models_bi.
            path_best_model = self.models_bi.filter(pl.col('val_loss') == self.models_bi.select('val_loss').min()).select('path_ckpt')[0,0]
            output = self.predict(dir_litdata, path_best_model, self.dir_output, batch_size, accelerator, n_workers)
            output.write_parquet(os.path.join(self.dir_output, 'dem_predicted_labels.parquet'))

            return None
        
        # Else for each inner fold
        for data_xx in list_ncv:
            self.run_xo_xi(data_xx[0], data_xx[1], dir_litdata, batch_size, accelerator, n_workers)
        
        return None
    
    def run_xo_xi(self, x_outer: int, x_inner: int, dir_litdata: str, batch_size: int, accelerator: str = 'auto', n_workers: int = 0):
        path_o_pred_trn = os.path.join(self.dir_output, f'dem_predicted_labels_{x_outer}_{x_inner}_trn.parquet')
        path_o_pred_val = os.path.join(self.dir_output, f'dem_predicted_labels_{x_outer}_{x_inner}_val.parquet')
        path_o_pred_tst = os.path.join(self.dir_output, f'dem_predicted_labels_{x_outer}_{x_inner}_tst.parquet')

        path_mdl = self.models_bv.filter((pl.col('x_outer') == x_outer) & (pl.col('x_inner') == x_inner)).select('path_ckpt')[0,0]
        print(f'\nUsing model {path_mdl}\n')

        ncv_data = MyDataModule4Train(dir_litdata, x_outer, x_inner, batch_size, n_workers)
        dir_train, dir_valid, dir_test = ncv_data.get_dir_ncv_litdata()

        pred_trn = self.predict(dir_train, path_mdl, self.dir_output, batch_size, accelerator, n_workers)
        pred_trn.write_parquet(path_o_pred_trn)
        pred_val = self.predict(dir_valid, path_mdl, self.dir_output, batch_size, accelerator, n_workers)
        pred_val.write_parquet(path_o_pred_val)
        pred_tst = self.predict(dir_test, path_mdl, self.dir_output, batch_size, accelerator, n_workers)
        pred_tst.write_parquet(path_o_pred_tst)
        print(f'\nPredicted labels saved to {path_o_pred_trn}, {path_o_pred_val}, {path_o_pred_tst}\n')

    def load_model(self, model_path: str, map_location: Optional[str] = None):
        self._model = DEMLTN.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=get_map_location(map_location),
        )
        self._model.eval()
        self._model.freeze()

    def collect_models(self):
        """
        Collect trained models for each fold in nested cross-validation.
        """
        collector = CollectFitLog(self.dir_logs)
        models_bv, models_bi = collector.get_df_csv(self.dir_output, self.overwrite_collected_log)
        self.models_bv = models_bv
        self.models_bi = models_bi

    def predict(
            self,
            dir_litdata: str,
            path_model_ckpt: str,
            dir_log_predict: str,
            batch_size: int = 32,
            accelerator: str = 'auto',
            n_workers: int = 0,
        ):
        """
        Predict phenotypes from omics data using a trained DEM model.
        """
        datamodule_ = MyDataModule4Uni(dir_litdata, batch_size, n_workers)
        datamodule_.setup()

        self.load_model(path_model_ckpt)

        available_devices = get_avail_nvgpu()

        trainer = Trainer(accelerator=accelerator, devices=available_devices, default_root_dir=dir_log_predict, logger=False)

        predictions = trainer.predict(model=self._model, datamodule=datamodule_)
        pred_array = np.concatenate(np.array(predictions), axis=0)

        # Output

        path_sample_ids = os.path.join(dir_litdata, 'sample_ids.csv')
        path_label_names = os.path.join(dir_litdata, 'label_names.csv')
        if not os.path.exists(path_sample_ids):
            match os.path.basename(dir_litdata):
                case 'train':
                    path_sample_ids = os.path.join(os.path.dirname(dir_litdata), 'sample_ids_trn.csv')
                    path_label_names = os.path.join(os.path.dirname(dir_litdata), 'label_names.csv')
                case 'valid':
                    path_sample_ids = os.path.join(os.path.dirname(dir_litdata), 'sample_ids_val.csv')
                    path_label_names = os.path.join(os.path.dirname(dir_litdata), 'label_names.csv')
                case 'test':
                    path_sample_ids = os.path.join(os.path.dirname(dir_litdata), 'sample_ids_tst.csv')
                    path_label_names = os.path.join(os.path.dirname(dir_litdata), 'label_names.csv')
                case _:
                    raise ValueError(f'Unknown directory name: {dir_litdata}')
        df_sample_ids = pl.read_csv(path_sample_ids)
        print(f'Number of samples in predict_dataloader: {len(df_sample_ids)}')
        assert len(df_sample_ids) == len(pred_array)

        label_names = pl.read_csv(path_label_names)['label'].to_list()
        
        pred_df = pl.DataFrame(pred_array, schema=label_names)
        pred_df = df_sample_ids.hstack(pred_df)
        return pred_df

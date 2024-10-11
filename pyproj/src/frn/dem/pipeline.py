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
import frn.constants as MC


class DEMFit:
    def __init__(
            self,
            log_dir: str,
            log_name: str,
            litdata_dir: str,
            which_outer_testset: int,
            which_inner_valset: int,
            regression: bool,
            devices: Union[List[int], str, int] = MC.default.devices,
            accelerator: str = MC.default.accelerator,
            n_jobs: int = MC.default.n_jobs,
            n_heads: int = MC.default.n_heads,
            n_encoders: int = MC.default.n_encoders,
            hidden_dim: int = MC.default.hidden_dim,
            learning_rate: float = MC.default.lr,
            dropout: float = MC.default.dropout,
            patience: int = MC.default.patience,
            max_epochs: int = MC.default.max_epochs,
            min_epochs: int = MC.default.min_epochs,
            batch_size: int = MC.default.batch_size,
        ):
        r"""DEM model training with hyperparameter optimization
        
        Args:
            log_dir: Directory for saving the training logs and models' checkpoints.

            log_name: Name of the training log.

            litdata_dir: Directory for loading the nested cross-validation data.

            which_outer_testset: Index of the outer test set.

            which_inner_valset: Index of the inner validation set.

            regression: Whether the task is regression or classification.

            devices: Devices to use.
                Default: "auto".
            
            accelerator: Accelerator to use.
                Default: "auto".
            
            n_jobs: Number of jobs to use for parallel hyperparameter optimization.
                Default: 1.
            
            n_heads: Number of heads in the attention mechanism.

            n_encoders: Number of Transformer Encoders.

            hidden_dim: Hidden dimension in the Transformer Encoder.

            learning_rate: Learning rate.

            dropout: Dropout rate.

            patience: Patience for early stopping.

            max_epochs: Maximum number of epochs.

            min_epochs: Minimum number of epochs.

            batch_size: Batch size.
        
        """
        self.log_dir = log_dir
        self.log_name = log_name
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.model_out_dim = pl.read_csv(os.path.join(litdata_dir, MC.fname.output_dim), has_header=True)[0,0]
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
        r"""Generate a dictionary of hyperparameters for DEM model training.
        """
        hparams = {
            MC.dkey.num_heads: n_heads,
            MC.dkey.num_encoders: n_encoders,
            MC.dkey.hidden_dim: hidden_dim,
            MC.dkey.lr: learning_rate,
            MC.dkey.dropout: dropout,
            MC.dkey.patience: patience,
            MC.dkey.max_epochs: max_epochs,
            MC.dkey.min_epochs: min_epochs,
            MC.dkey.bsize: batch_size,
        }
        return hparams

    def dem_fit(
            self,
            hparams:Dict[str, Any],
            devices: Union[List[int], str, int],
            accelerator: str,
        ):
        r"""Train DEM model with given hyperparameters and input/output paths.
        """
        _model = DEMLTN(
            omics_dim=self.omics_dims,
            n_heads=hparams[MC.dkey.num_heads],
            n_encoders=hparams[MC.dkey.num_encoders],
            hidden_dim=hparams[MC.dkey.hidden_dim],
            output_dim=self.model_out_dim,
            dropout=hparams[MC.dkey.dropout],
            learning_rate=hparams[MC.dkey.lr],
            is_regression=self.is_regression,
        )
        
        log_dir_uniq_model = os.path.join(self.log_dir, self.log_name, random_string())

        val_loss_min = train_model(
            model=_model,
            datamodule=self.datamodule,
            es_patience=hparams[MC.dkey.patience],
            max_epochs=hparams[MC.dkey.max_epochs],
            min_epochs=hparams[MC.dkey.min_epochs],
            log_dir=log_dir_uniq_model,
            devices=devices,
            accelerator=accelerator,
        )
        if val_loss_min is None:
            raise ValueError("Training failed.")
        return val_loss_min

    def manual_train(self):
        r"""Train DEM model with manually specified hyperparameters.
        """
        val_loss_min = self.dem_fit(
            hparams = self.hparams,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        return val_loss_min

    def objective(self, trial: optuna.Trial) -> float:
        r"""Objective function for DEM model training with Optuna.
        """
        print("Trial number:", trial.number)
        if self.n_jobs > 1:
            time_delay = (trial.number + self.n_jobs) % self.n_jobs * MC.default.time_delay
            time.sleep(time_delay)

        # Generate hyperparameters
        batch_size = trial.suggest_categorical(MC.dkey.bsize, MC.hparam_candidates.batch_size)
        n_heads = trial.suggest_categorical(MC.dkey.num_heads, MC.hparam_candidates.n_heads)
        n_encoders = trial.suggest_categorical(MC.dkey.num_encoders, MC.hparam_candidates.n_encoders)
        hidden_dim = trial.suggest_categorical(MC.dkey.hidden_dim, MC.hparam_candidates.hidden_dim)
        dropout = trial.suggest_float(MC.dkey.dropout, 0.0, MC.hparam_candidates.dropout_high, step=MC.hparam_candidates.dropout_step)
        lr = trial.suggest_categorical(MC.dkey.lr, MC.hparam_candidates.lr)

        # Update hyperparameters in DEMTrain object based on manual parameters in initialization
        hparams_tmp = self.hparams.copy()
        hparams_tmp[MC.dkey.bsize] = batch_size
        hparams_tmp[MC.dkey.num_heads] = n_heads
        hparams_tmp[MC.dkey.num_encoders] = n_encoders
        hparams_tmp[MC.dkey.hidden_dim] = hidden_dim
        hparams_tmp[MC.dkey.lr] = lr
        hparams_tmp[MC.dkey.dropout] = dropout
        
        val_loss_min = self.dem_fit(
            hparams = hparams_tmp,
            devices=self.devices,
            accelerator=self.accelerator,
        )
        
        return val_loss_min

    def optimize(
            self,
            n_trials: Optional[int] = None,
            storage: str = MC.default.optuna_db,
            gc_after_trial: bool = True,
        ):
        r"""Optimize hyperparameters of DEM model with Optuna.
        """
        study = optuna.create_study(
            storage = storage,
            study_name = self.log_name + "_" + time_string(),
            direction = "minimize",
            load_if_exists = True,
        )
        study.optimize(self.objective, n_trials=n_trials, n_jobs=self.n_jobs, gc_after_trial=gc_after_trial)


class DEMFitPipe:
    def __init__(
            self,
            litdata_dir: str,
            list_ncv: List[List[int]],
            log_dir: str,
            regression: bool,
            devices: Union[List[int], str, int] = MC.default.devices,
            accelerator: str = MC.default.accelerator,
            n_jobs: int = MC.default.n_jobs,
            n_trials: Optional[int] = MC.default.n_trials,
        ) -> None:
        r"""DEM model training pipeline with hyperparameter trials.
        
        Args:
            litdata_dir: Path to the directory containing the nested cross-validation litdata.

            list_ncv: List of lists containing the indices of the outer and inner folds for each data slice.

            log_dir: Path to the directory where the training logs and checkpoints will be saved.

            regression: Whether the task is regression or classification.

            devices: Device(s) to use.
                Default: ``"auto"``.
            
            accelerator: Accelerator to use.
                Default: ``"auto"``.
            
            n_jobs: Number of jobs to use for parallelization.
                Default: ``1``.
            
            n_trials: Number of trials to run for hyperparameter optimization.
                Default: ``10``.
        
        Usage:
        
            >>> from frn.dem.pipeline import DEMFitPipe
            >>> _pipe = DEMFitPipe(...)
            >>> _pipe.train_pipeline()
        
        """
        # Unique tag for the training log directory
        tag_str = time_string() + '_' + random_string()
        self.uniq_logdir = os.path.join(log_dir, MC.title_train + "_" + tag_str)
        os.makedirs(self.uniq_logdir, exist_ok=False)

        self.litdata_dir = litdata_dir
        self.list_ncv = list_ncv
        self.n_slice = len(list_ncv)
        
        self.regression = regression
        self.devices = devices
        self.accelerator = accelerator
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        
    def train_pipeline(self):
        r"""Train DEM model for each fold in nested cross-validation.
        """
        # Storage for optuna trials in self.log_dir
        path_storage = 'sqlite:///' + self.uniq_logdir + '/optuna.db'
        
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
        
        # Remove checkpoints of inferior models
        _collector = CollectFitLog(self.uniq_logdir)
        _collector.remove_inferior_models()


class DEMPredict:
    def __init__(self):
        r"""Prediction pipeline for DEM model.
        """

    def runs(
            self,
            litdata_dir: str,
            dir_fit_logs: str,
            dir_output: str,
            list_ncv: Optional[List[List[int]]] = None,
            overwrite_collected_log: bool = False,
            accelerator: str = MC.default.accelerator,
            batch_size: int = MC.default.batch_size,
            n_workers: int = MC.default.n_workers,
        ):
        r"""Run prediction for each fold in nested cross-validation.

        Args:
            litdata_dir: Path to the directory containing the nested cross-validation litdata.

            dir_fit_logs: Path to the directory containing the training logs.

            dir_output: Path to the directory where the prediction results will be saved.

            list_ncv: List of lists containing the indices of the outer and inner folds for each data slice.

            overwrite_collected_log: Whether to overwrite the collected log file.

            accelerator: Accelerator to use.
                Default: ``"auto"``.
            
            batch_size: Batch size to use.
                Default: ``32``.
            
            n_workers: Number of workers to use for dataloader.
                Default: ``0``.
        
        """
        os.makedirs(dir_output, exist_ok=True)
        if not hasattr(self, 'models_bv'):
            self.collect_models(dir_fit_logs, dir_output, overwrite_collected_log)
        
        if list_ncv is None:
            # Take the best model's path overall by searching the line min `val_loss` in models_bi.
            path_best_model = self.models_bi.filter(pl.col(MC.dkey.val_loss) == self.models_bi.select(MC.dkey.val_loss)).min().select(MC.dkey.ckpt_path)[0,0]
            output = self.predict(litdata_dir, path_best_model, dir_output, batch_size, accelerator, n_workers)
            output.write_parquet(os.path.join(dir_output, MC.fname.predicted_labels))

            return None
        
        # Else for each inner fold
        for data_xx in list_ncv:
            self.run_xo_xi(data_xx[0], data_xx[1], litdata_dir, dir_output, batch_size, accelerator, n_workers)
        
        return None
    
    def run_xo_xi(self, x_outer: int, x_inner: int, litdata_dir: str, dir_output: str, batch_size: int, accelerator: str = MC.default.accelerator, n_workers: int = MC.default.n_workers):
        os.makedirs(dir_output, exist_ok=True)
        path_o_pred_trn = os.path.join(dir_output, MC.fname.predicted_labels.replace(".parquet", f'_{x_outer}_{x_inner}_trn.parquet'))
        path_o_pred_val = os.path.join(dir_output, MC.fname.predicted_labels.replace(".parquet", f'_{x_outer}_{x_inner}_val.parquet'))
        path_o_pred_tst = os.path.join(dir_output, MC.fname.predicted_labels.replace(".parquet", f'_{x_outer}_{x_inner}_tst.parquet'))

        path_mdl = self.models_bv.filter((pl.col(MC.dkey.which_outer) == x_outer) & (pl.col(MC.dkey.which_inner) == x_inner)).select(MC.dkey.ckpt_path)[0,0]
        print(f'\nUsing model {path_mdl}\n')

        ncv_data = MyDataModule4Train(litdata_dir, x_outer, x_inner, batch_size, n_workers)
        dir_train, dir_valid, dir_test = ncv_data.get_dir_ncv_litdata()

        pred_trn = self.predict(dir_train, path_mdl, dir_output, batch_size, accelerator, n_workers)
        pred_trn.write_parquet(path_o_pred_trn)
        pred_val = self.predict(dir_valid, path_mdl, dir_output, batch_size, accelerator, n_workers)
        pred_val.write_parquet(path_o_pred_val)
        pred_tst = self.predict(dir_test, path_mdl, dir_output, batch_size, accelerator, n_workers)
        pred_tst.write_parquet(path_o_pred_tst)
        print(f'\nPredicted labels saved to {path_o_pred_trn}, {path_o_pred_val}, {path_o_pred_tst}\n')

    def load_model(self, model_path: str, map_location: Optional[str] = None):
        self._model = DEMLTN.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=get_map_location(map_location),
        )
        self._model.eval()
        self._model.freeze()

    def collect_models(self, dir_fit_logs: str, dir_output: str, overwrite_collected_log: bool = False):
        r"""Collect trained models for each fold in nested cross-validation.
        """
        os.makedirs(dir_output, exist_ok=True)
        collector = CollectFitLog(dir_fit_logs)
        models_bv, models_bi = collector.get_df_csv(dir_output, overwrite_collected_log)
        self.models_bv = models_bv
        self.models_bi = models_bi

    def predict(
            self,
            litdata_dir: str,
            path_model_ckpt: str,
            dir_log_predict: Optional[str],
            batch_size: int = MC.default.batch_size,
            accelerator: str = MC.default.accelerator,
            n_workers: int = MC.default.n_workers,
        ):
        r"""Predict phenotypes from omics data using a trained DEM model.

        Args:
            litdata_dir: Path to the directory containing the litdata.
                The directory should have a parent directory which is for nested cross-validation (e.g., ``ncv_test_0_val_0``).

            path_model_ckpt: Path to the trained DEM model.

            dir_log_predict: The directory to save the prediction logs.
        
        """
        parent_dir = os.path.dirname(litdata_dir)
        datamodule_ = MyDataModule4Uni(litdata_dir, batch_size, n_workers)
        datamodule_.setup()

        if not hasattr(self, "_model"):
            self.load_model(path_model_ckpt)

        available_devices = get_avail_nvgpu()

        trainer = Trainer(accelerator=accelerator, devices=available_devices, default_root_dir=dir_log_predict, logger=False)

        predictions = trainer.predict(model=self._model, datamodule=datamodule_)
        assert predictions is not None
        pred_array = np.concatenate(predictions)
        print(f"Shape of prediction results: {pred_array.shape}")

        # Output

        path_sample_ids = os.path.join(parent_dir, MC.fname.predata_ids)
        path_label_names = os.path.join(parent_dir, MC.fname.predata_label_names)

        data_dir_name = os.path.basename(litdata_dir)
        if data_dir_name.startswith(MC.title_train):
            path_sample_ids = os.path.join(parent_dir, MC.fname.predata_ids_trn)
        elif data_dir_name.startswith(MC.title_val):
            path_sample_ids = os.path.join(parent_dir, MC.fname.predata_ids_val)
        elif data_dir_name.startswith(MC.title_test):
            path_sample_ids = os.path.join(parent_dir, MC.fname.predata_ids_tst)
        else:
            raise ValueError(f'Unknown directory name: {litdata_dir}')
        
        df_sample_ids = pl.read_csv(path_sample_ids)
        assert len(df_sample_ids) == len(pred_array)
        print(f'Number of samples in predict_dataloader: {len(df_sample_ids)}')

        label_names = pl.read_csv(path_label_names)[MC.dkey.label].to_list()
        
        pred_df = pl.DataFrame(pred_array, schema=label_names)
        pred_df = df_sample_ids.hstack(pred_df)
        return pred_df

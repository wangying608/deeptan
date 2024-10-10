import os
import shutil
from typing import Optional, Union, List
import numpy as np
import polars as pl
from lightning import Trainer
from frn.dem.model import DEMLTN
import frn.constants as MC
from frn.utils.data_ncv import MyDataModule4Uni, read_omics_names
from frn.utils.uni import get_map_location, get_avail_nvgpu, CollectFitLog


class DEMFeatureRanking:
    def __init__(
            self,
            batch_size: int = MC.default.batch_size,
            n_workers: int = MC.default.n_workers,
            accelerator: str = MC.default.accelerator,
            map_location: Optional[str] = None,
        ):
        r"""Feature ranking using a trained DEM model.
        
        Args:
            model_path: Path to the trained DEM model.

            batch_size: Batch size for prediction.

            map_location: Map location for loading the model.
        
        """
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.accelerator = accelerator
        self.map_location = map_location
    
    def run_a_outer(self, ncv_litdata_dir: str, fit_log_dir: str, which_outer: int, output_path: str, random_states: list[int]):
        r"""Run feature ranking for a single outer fold in nested cross-validation.
        This function searches the best inner fold for the specified outer fold and runs feature ranking on the test set.

        Args:
            ncv_litdata_dir: Path to the directory containing the nested cross-validation litdata.

            fit_log_dir: Path to the directory containing the training logs.

            which_outer: Which outer fold to run.

            output_path: Path to the file to save the results.

            random_states: List of random states for shuffling values of each feature.
        
        """
        # Collect fit logs
        collector = CollectFitLog(fit_log_dir)
        fit_logs = collector.collect()
        log_best_inner_foreach_outer = fit_logs[MC.dkey.best_inner_folds]
        
        row_x_outer = log_best_inner_foreach_outer.filter(pl.col(MC.dkey.which_outer)==which_outer)
        _model_path = row_x_outer[MC.dkey.ckpt_path][0]
        _which_inner = row_x_outer[MC.dkey.which_inner][0]
        _litdata_dir = os.path.join(ncv_litdata_dir, f"ncv_test_{which_outer}_val_{_which_inner}", MC.title_test)

        self.run(_model_path, _litdata_dir, output_path, random_states)

    def run(self, model_path: str, litdata_dir: str, output_path: str, random_states: list[int]):
        r"""Rank features.

        Args:
            litdata_dir: Path to the directory containing the nested cross-validation litdata.

            output_path: Path to the file to save the results.

            random_states: List of random states for shuffling features.
        
        """
        # Load model
        self.load_model(model_path)

        # Get original prediction loss
        _datamodule = MyDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)
        _datamodule.setup()
        loss = self.trainer.test(model=self._model, datamodule=_datamodule)
        loss = np.mean(np.array(loss))
        
        # Shuffle features and get shuffled prediction loss.
        omics_names, omics_feat_paths = read_omics_names(litdata_dir, True)
        importance_scores = []
        feature_names: List[str] = []
        which_omics = []
        for x_om in range(len(omics_names)):
            _feat_names: List[str] = pl.read_csv(omics_feat_paths[x_om]).to_series().to_list()
            n_features = len(_feat_names)
            for x_feat in range(n_features):
                feature_names.append(_feat_names[x_feat])
                which_omics.append(omics_names[x_om])
                importance_scores.append(self.run_a_feat(litdata_dir, x_om, x_feat, random_states))

        # Rank features by their importance scores
        importance_scores = loss - np.array(importance_scores)
        importance_scores_abs = np.absolute(importance_scores)
        print(feature_names[:4])
        print(importance_scores_abs[:4])
        _sortperm = np.argsort(importance_scores_abs, order='descending')
        importance_scores = np.take(importance_scores, _sortperm)
        importance_scores_abs = np.take(importance_scores_abs, _sortperm)
        feature_names = [feature_names[i] for i in _sortperm.astype(int).tolist()]
        which_omics = [which_omics[i] for i in _sortperm.astype(int).tolist()]

        # Save feature ranking results
        df_o = pl.DataFrame({MC.dkey.omics: which_omics, MC.dkey.feature: feature_names, MC.dkey.feat_importance: importance_scores, MC.dkey.feat_importance_abs: importance_scores_abs})
        _filename = os.path.splitext(output_path)[0]
        df_o.write_csv(_filename + ".csv")
        df_o.write_parquet(_filename + ".parquet")

    def run_a_feat(self, litdata_dir: str, which_omics: Union[int, str], which_feature: int, random_states: List[int]) -> float:
        r""" Get average loss for a single shuffled feature.
        """
        # Load data
        losses: List[float] = []
        for random_state in random_states:
            _datamodule = MyDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)
            _datamodule.shuffle_a_feat(which_omics, which_feature, random_state)
            losses_shuffled = self.trainer.test(model=self._model, datamodule=_datamodule)
            losses.append(np.mean(np.array(losses_shuffled)).item())
        losses_avg = np.mean(losses).item()
        return losses_avg
        
    def load_model(self, model_path: str):
        r"""Load a model's checkpoint to specified device and define a trainer.

        Args:
            model_path: Path to the model's checkpoint.
        
        """
        self._model = DEMLTN.load_from_checkpoint(
            checkpoint_path=model_path,
            map_location=get_map_location(self.map_location),
        )
        self._model.eval()
        self._model.freeze()
        self.available_devices = get_avail_nvgpu()
        self.trainer = Trainer(accelerator=self.accelerator, devices=self.available_devices, default_root_dir=None, logger=False)

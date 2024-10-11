import os
import pickle
from typing import Optional, Union, List, Dict
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
        _filename = os.path.splitext(output_path)[0]

        # Load model
        self.load_model(model_path)

        # Load data
        self._datamodule = MyDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)

        # Get original prediction loss
        self._datamodule.setup()
        loss = self.trainer.test(self._model, self._datamodule)[0][MC.title_tst_loss]
        
        # Shuffle features and get shuffled prediction loss.
        omics_names, omics_feat_paths = read_omics_names(litdata_dir, True)
        importance_scores_list: List[float] = []
        predicted_labels: Dict[str, np.ndarray] = {}
        feature_names: List[str] = []
        which_omics = []
        for x_om in range(len(omics_names)):
            _feat_names: List[str] = pl.read_csv(omics_feat_paths[x_om]).to_series().to_list()
            n_features = len(_feat_names)
            for x_feat in range(n_features):
                feature_names.append(_feat_names[x_feat])
                which_omics.append(omics_names[x_om])
                _avg_loss, _avg_pred = self.run_a_feat(x_om, x_feat, random_states)
                importance_scores_list.append(_avg_loss)
                predicted_labels[f"{omics_names[x_om]}+{_feat_names[x_feat]}"] = _avg_pred

        # Write predicted labels to pickle file
        path_pkl = _filename + ".pkl"
        with open(path_pkl, "wb") as f:
            pickle.dump(predicted_labels, f)

        # Rank features by their importance scores
        importance_scores = np.array(importance_scores_list).flatten().astype(np.float64) * -1.0 / loss + 1.0
        importance_scores_abs = np.absolute(importance_scores).flatten().tolist()
        _sortperm = np.argsort(importance_scores_abs)

        importance_scores = importance_scores[_sortperm].tolist()
        importance_scores_abs = importance_scores_abs[_sortperm]
        feature_names = [feature_names[i] for i in _sortperm.astype(int).tolist()]
        which_omics = [which_omics[i] for i in _sortperm.astype(int).tolist()]

        # Save feature ranking results as CSV and Parquet
        df_o = pl.DataFrame({MC.dkey.omics: which_omics, MC.dkey.feature: feature_names, MC.dkey.feat_importance: importance_scores, MC.dkey.feat_importance_abs: importance_scores_abs})
        df_o.write_csv(_filename + ".csv")
        df_o.write_parquet(_filename + ".parquet")
    
    def run_a_feat(self, which_omics: Union[int, str], which_feature: int, random_states: List[int], litdata_dir: Optional[str]=None, model_path: Optional[str]=None):
        r""" Get average loss for a single shuffled feature.
        """
        if not hasattr(self, '_datamodule'):
            if litdata_dir is not None:
                self._datamodule = MyDataModule4Uni(litdata_dir, self.batch_size, self.n_workers)
            else:
                raise ValueError("Please specify litdata_dir.")
        if not hasattr(self, "trainer"):
            if model_path is not None:
                self.load_model(model_path)
            else:
                raise ValueError("Please specify model_path.")
        
        losses: List[float] = []
        predictions = []
        for random_state in random_states:
            _dataloader = self._datamodule.shuffle_a_feat(which_omics, which_feature, random_state)
            losses_shuffled = self.trainer.test(self._model, _dataloader)[0][MC.title_tst_loss]
            #
            predictions_shuffled = self.trainer.predict(self._model, _dataloader)
            assert predictions_shuffled is not None
            pred_array_shuffled = np.concatenate(predictions_shuffled)
            # print(f"Shape of prediction results: {pred_array_shuffled.shape}")
            #
            losses.append(losses_shuffled)
            predictions.append(pred_array_shuffled)
        losses_avg = np.mean(losses).item()
        predictions_avg = np.mean(predictions, axis=0)
        print(f"\nAverage loss: {losses_avg}")
        # print(f"Average prediction: {predictions_avg}\n")
        # print(f"Shape of average prediction: {predictions_avg.shape}\n")
        return losses_avg, predictions_avg
        
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

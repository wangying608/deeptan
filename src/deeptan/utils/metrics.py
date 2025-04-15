from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from scib.metrics import kBET
from scib.metrics import silhouette as ASW
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, pearsonr
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import f1_score as F1
from sklearn.metrics import homogeneity_score as HOM
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import silhouette_score


class RegressionMetricsCalculator:
    r"""
    Optimized class to compute metrics between two 2D numpy arrays with minimized code duplication.
    """

    def __init__(self, true_array: np.ndarray, pred_array: np.ndarray):
        self._true = true_array
        self._pred = pred_array
        self.n_samples, self.n_features = true_array.shape
        self._validate_arrays()

        # Define metric calculation functions
        self.metric_functions = {
            "mse": self._calculate_mse,
            "mae": self._calculate_mae,
            "jsd": self._calculate_jsd,
            "pcc": self._calculate_pcc,
        }

    def _validate_arrays(self):
        if self._true.shape != self._pred.shape:
            raise ValueError("Input arrays must have the same shape")
        if len(self._true.shape) != 2:
            raise ValueError("Input arrays must be 2-dimensional")

    def _normalize_for_jsd(self, arr: np.ndarray, axis: int) -> np.ndarray:
        arr = arr + 1e-10
        return arr / arr.sum(axis=axis, keepdims=True)

    def _calculate_mse(self, axis: int) -> np.ndarray:
        return ((self._true - self._pred) ** 2).mean(axis=axis)

    def _calculate_mae(self, axis: int) -> np.ndarray:
        return np.abs(self._true - self._pred).mean(axis=axis)

    def _calculate_jsd(self, axis: int) -> np.ndarray:
        # true_norm = self._normalize_for_jsd(self._true, axis)
        # pred_norm = self._normalize_for_jsd(self._pred, axis)

        # def jsd(p, q):
        #     m = 0.5 * (p + q)
        #     return 0.5 * (entropy(p, m) + entropy(q, m))

        # # Transpose if calculating along features to maintain consistency
        # if axis == 0:
        #     return np.array([jsd(p, q) for p, q in zip(true_norm.T, pred_norm.T)])
        # return np.array([jsd(p, q) for p, q in zip(true_norm, pred_norm)])

        # Use scipy's optimized implementation
        # if axis == 0:
        #     return np.array([jensenshannon(p, q) ** 2 for p, q in zip(true_norm.T, pred_norm.T)])
        # return np.array([jensenshannon(p, q) ** 2 for p, q in zip(true_norm, pred_norm)])

        return np.array(jensenshannon(self._true, self._pred, axis=axis) ** 2)

    def _calculate_pcc(self, axis: int) -> np.ndarray:
        if axis == 1:  # Sample-wise (rows)
            return np.array([pearsonr(t_row, p_row)[0] for t_row, p_row in zip(self._true, self._pred)])
        else:  # Feature-wise (columns)
            return np.array([pearsonr(self._true[:, i], self._pred[:, i])[0] for i in range(self.n_features)])

    def _calculate_metrics(self, axis: int) -> Dict[str, Any]:
        r"""
        Unified metric calculation for either samples or features.

        Args:
            axis (int): 0 for features, 1 for samples

        Returns:
            Dict[str, Any]: Dictionary of metric results
        """
        metrics = {}
        for name, func in self.metric_functions.items():
            values = func(axis)
            metrics[name] = {"values": values, "mean": np.mean(values)}
        return metrics

    def calculate_sample_metrics(self) -> Dict[str, Any]:
        """Calculate metrics row-wise (per sample)."""
        return self._calculate_metrics(axis=1)

    def calculate_feature_metrics(self) -> Dict[str, Any]:
        """Calculate metrics column-wise (per feature)."""
        return self._calculate_metrics(axis=0)

    def calculate_all_metrics(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        r"""
        Calculate all metrics and return as nested dictionary.

        Args:
            output_path (str, optional): Path to save parquet file

        Returns:
            dict: Nested dictionary with all results
            pl.DataFrame: Summary DataFrame
        """
        results = {"sample_metrics": self.calculate_sample_metrics(), "feature_metrics": self.calculate_feature_metrics(), "shape": {"n_samples": self.n_samples, "n_features": self.n_features}}

        # Create summary DataFrame
        summary_data = []
        for metric in self.metric_functions.keys():
            summary_data.append({"metric": metric, "sample_mean": results["sample_metrics"][metric]["mean"], "feature_mean": results["feature_metrics"][metric]["mean"]})
        df = pl.DataFrame(summary_data)

        results.update({"averaged": df})

        return results

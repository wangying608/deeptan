import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import polars as pl
from scib.metrics import kBET
from scib.metrics import silhouette as ASW
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import average_precision_score as AUPRC
from sklearn.metrics import f1_score as F1
from sklearn.metrics import homogeneity_score as HOM
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import roc_auc_score as AUROC


def format_ticks(x, pos):
    r"""
    Format the ticks on the x-axis of a plot.
    """
    return f"{x:.2f}"


class MetricsDictMaker:
    r"""
    Make a dictionary of metrics for predictions of **a dataset**.
    """

    def __init__(self, predictions_dir: str, true_data_dir: str, softmax_pred_labels: bool = True):
        self.predictions_dir = predictions_dir
        self.true_data_dir = true_data_dir
        self.softmax_pred_labels = softmax_pred_labels

        self.metrics_dict = {}

        self.ident = self.detect_pkl()
        self.fnames = self.ident["fname"].to_list()
        self.metrics_dict["identify"] = self.ident
        self.metrics_dict["prediction"] = {}
        self.metrics_dict["true"] = {}
        self.metrics_dict["metrics"] = {}

    def run(self):
        self.load_predictions()
        self.load_true()
        self.compute_all_metrics()
        self.make_metrics_summary()

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(a=x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def make_metrics_summary(self):
        # For recon
        _dfs = []
        for _fname in self.fnames:
            _tmp_df = self.metrics_dict["metrics"]["recon"][_fname]["averaged"]
            _tmp_df = pl.DataFrame({"fname": [_fname] * _tmp_df.shape[0]}).hstack(_tmp_df)
            _dfs.append(_tmp_df)
        _tmp: pl.DataFrame = pl.concat(_dfs)
        self.metrics_dict["summary_recon"] = _tmp.join(self.ident, on="fname", how="left")

        # For label
        _dfs = []
        for _fname in self.fnames:
            _tmp_df = self.metrics_dict["metrics"]["label"][_fname]
            _tmp_df = pl.DataFrame({"fname": [_fname] * _tmp_df.shape[0]}).hstack(_tmp_df)
            _dfs.append(_tmp_df)
        _tmp: pl.DataFrame = pl.concat(_dfs)
        self.metrics_dict["summary_label"] = _tmp.join(self.ident, on="fname", how="left")

    def compute_all_metrics(self):
        """Computes all metrics for the predictions."""
        # For recon
        self.metrics_dict["metrics"]["recon"] = {}
        for _fname in self.fnames:
            _seed = self.ident.filter(pl.col("fname") == _fname)["seed_num"].item()
            _calculator = RegressionMetricsCalculator(self.metrics_dict["true"][f"seed_{_seed}_tst"]["X"], self.metrics_dict["prediction"][_fname]["X"])
            self.metrics_dict["metrics"]["recon"][_fname] = _calculator.calculate_all_metrics()

        # For label
        self.metrics_dict["metrics"]["label"] = {}
        for _fname in self.fnames:
            _seed = self.ident.filter(pl.col("fname") == _fname)["seed_num"].item()
            n_class = self.metrics_dict["prediction"][_fname]["y"].shape[1]
            _calculator = MulticlassMetricsCalculator(self.metrics_dict["true"][f"seed_{_seed}_tst"]["y"], self.metrics_dict["prediction"][_fname]["y"], n_class)
            self.metrics_dict["metrics"]["label"][_fname] = pl.DataFrame(_calculator.calculate_metrics())

    def load_predictions(self):
        """Loads predictions from pickle files."""
        for _fname in self.fnames:
            _path = self.ident.filter(pl.col("fname") == _fname)["path"].item()
            with open(_path, "rb") as f:
                self.metrics_dict["prediction"][_fname] = pickle.load(f)
            self.metrics_dict["prediction"][_fname]["X"] = np.squeeze(self.metrics_dict["prediction"][_fname]["node_recon_all"], axis=-1)
            self.metrics_dict["prediction"][_fname]["y"] = self.metrics_dict["prediction"][_fname]["labels"]
            if self.softmax_pred_labels:
                self.metrics_dict["prediction"][_fname]["y"] = self.softmax(self.metrics_dict["prediction"][_fname]["y"])

    def load_true(self):
        seeds_uniq = self.ident["seed_num"].unique().sort().to_list()
        for _seed in seeds_uniq:
            self.metrics_dict["true"][f"seed_{_seed}_tst"] = {}

            # Use the 1st fname of seed xx to get feature names
            _fname = self.ident.filter(pl.col("seed_num") == _seed)["fname"].to_list()[0]
            feature_names_seed_xx = ["obs_names"] + list(self.metrics_dict["prediction"][_fname]["dict_node_names"].keys())
            test_data_df = pl.read_parquet(os.path.join(self.true_data_dir, f"split_{_seed}_2.parquet")).select(feature_names_seed_xx)
            test_data = test_data_df.drop(["obs_names"]).to_numpy()
            # Apply log1p
            self.metrics_dict["true"][f"seed_{_seed}_tst"]["X"] = np.log1p(test_data)

            # Load true labels
            _labels_df = pl.read_parquet(os.path.join(self.true_data_dir, "celltypes_onehot.parquet"))
            _labels_df = _labels_df.rename({"bc": "obs_names"})
            _labels_all = self.transform_ct_df(_labels_df)

            self.metrics_dict["true"][f"seed_{_seed}_tst"]["y_df_flatten"] = _labels_all.join(test_data_df.select(["obs_names"]), on="obs_names", how="right").select(["obs_names", "ct"])
            self.metrics_dict["true"][f"seed_{_seed}_tst"]["y_df"] = _labels_df.join(test_data_df.select(["obs_names"]), on="obs_names", how="right")
            self.metrics_dict["true"][f"seed_{_seed}_tst"]["y"] = self.metrics_dict["true"][f"seed_{_seed}_tst"]["y_df"].drop("obs_names").to_numpy()

    def detect_pkl(self):
        """Detects all pickle files in the predictions directory."""
        _fname = []
        _seeds = []
        _seeds_num = []
        _tasks = []
        _split = []
        _paths = []
        for _file in os.listdir(self.predictions_dir):
            if _file.endswith(".pkl"):
                _path = os.path.join(self.predictions_dir, _file)
                _prop = _path.strip(".pkl").split("+")
                _seeds.append(_prop[1])
                _seeds_num.append(int(_prop[1].replace("seed_", "")))
                _tasks.append(_prop[2])
                _split.append(_prop[3])
                _paths.append(_path)
                _fname.append(_file)
        return pl.DataFrame({"fname": _fname, "seed": _seeds, "seed_num": _seeds_num, "task": _tasks, "split": _split, "path": _paths})

    def transform_ct_df(self, df: pl.DataFrame):
        # 熔化数据框以使每个类别成为一行
        melted_df = df.unpivot(index=["obs_names"], on=df.columns[1:], variable_name="ct", value_name="value")

        # 过滤出值为1的行，因为每行只有一个1，所以这样可以得到正确的类别名称
        filtered_df = melted_df.filter(pl.col("value") == 1)

        # 选择需要的列
        result_df = filtered_df.select(["obs_names", "ct"])

        celltypes_ = result_df["ct"].to_list()
        celltypes = [ct.replace("ct_", "") for ct in celltypes_ if ct.startswith("ct_")]
        result_df = result_df.hstack(pl.DataFrame({"ct_": celltypes})).drop("ct").rename({"ct_": "ct"})

        return result_df


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

    def _calculate_mse(self, axis: int) -> np.ndarray:
        return ((self._true - self._pred) ** 2).mean(axis=axis)

    def _calculate_mae(self, axis: int) -> np.ndarray:
        return np.abs(self._true - self._pred).mean(axis=axis)

    def _calculate_jsd(self, axis: int) -> np.ndarray:
        # Ensure inputs are probability distributions
        def normalize(x: np.ndarray) -> np.ndarray:
            x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)  # softmax if needed
            # Alternatively, use simple normalization:
            # x = x / np.sum(x, axis=axis, keepdims=True)
            return x

        true_normalized = normalize(self._true)
        pred_normalized = normalize(self._pred)
        return np.array(jensenshannon(true_normalized, pred_normalized, axis=axis) ** 2)

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

    def calculate_all_metrics(self) -> Dict[str, Any]:
        r"""
        Calculate all metrics and return as nested dictionary.

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


class MulticlassMetricsCalculator:
    r"""
    Class to compute metrics for multi-class classification tasks.
    Supported metrics include weighted F1, macro F1, micro F1, AUROC, AUPRC, accuracy, precision, and recall.
    """

    def __init__(self, true_labels: np.ndarray, pred_probs: np.ndarray, num_classes: int):
        """
        Initialize the calculator with true labels and predicted probabilities.

        Args:
            true_labels (np.ndarray): 1D array of true class labels.
            pred_probs (np.ndarray): 2D array of predicted probabilities for each class.
            num_classes (int): Number of classes in the classification task.
        """
        self._true_df = true_labels
        self._pred_probs = pred_probs
        self._num_classes = num_classes
        self._labels = np.arange(num_classes).tolist()  # Default labels are 0, 1, ..., num_classes-1

        # Convert predicted probabilities to predicted labels
        self._pred_labels = np.argmax(pred_probs, axis=1)
        self._true = np.argmax(self._true_df, axis=1)

        self._validate_inputs()

        # Define metric calculation functions
        self.metric_functions = {
            "weighted_f1": self._calculate_weighted_f1,
            "macro_f1": self._calculate_macro_f1,
            "micro_f1": self._calculate_micro_f1,
            "auroc": self._calculate_auroc,
            "auprc": self._calculate_auprc,
            "accuracy": self._calculate_accuracy,
            "weighted_precision": self._calculate_weighted_precision,
            "weighted_recall": self._calculate_weighted_recall,
        }

    def _validate_inputs(self):
        """
        Validate the input arrays.
        """
        if len(self._true.shape) != 1:
            raise ValueError("True labels must be a 1-dimensional array.")
        if len(self._pred_probs.shape) != 2:
            raise ValueError("Predicted probabilities must be a 2-dimensional array.")
        if self._true.shape[0] != self._pred_probs.shape[0]:
            raise ValueError("True labels and predicted probabilities must have the same number of samples.")
        if self._pred_probs.shape[1] != self._num_classes:
            raise ValueError("Number of columns in predicted probabilities must match the number of classes.")
        if not np.allclose(self._pred_probs.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError("Predicted probabilities must be softmax normalized (rows should sum to 1)")

    def _calculate_weighted_f1(self) -> float:
        return F1(self._true, self._pred_labels, average="weighted", labels=self._labels, zero_division=0.0)

    def _calculate_macro_f1(self) -> float:
        return F1(self._true, self._pred_labels, average="macro", labels=self._labels, zero_division=0.0)

    def _calculate_micro_f1(self) -> float:
        return F1(self._true, self._pred_labels, average="micro", labels=self._labels, zero_division=0.0)

    def _calculate_auroc(self) -> float:
        try:
            return AUROC(self._true_df, self._pred_probs, multi_class="ovr", average="weighted", labels=self._labels)
        except ValueError:
            return np.nan  # Return NaN if AUROC cannot be computed

    def _calculate_auprc(self) -> float:
        return AUPRC(self._true_df, self._pred_probs, average="weighted")

    def _calculate_accuracy(self) -> float:
        return accuracy_score(self._true, self._pred_labels)

    def _calculate_weighted_precision(self) -> float:
        return precision_score(self._true, self._pred_labels, average="weighted", labels=self._labels, zero_division=0.0)

    def _calculate_weighted_recall(self) -> float:
        return recall_score(self._true, self._pred_labels, average="weighted", labels=self._labels, zero_division=0.0)

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate all metrics and return them as a dictionary.

        Returns:
            Dict[str, float]: Dictionary of metric results.
        """
        metrics = {}
        for name, func in self.metric_functions.items():
            metrics[name] = func()
        return metrics

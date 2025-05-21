import os
from typing import Any, Dict, List, Optional

import anndata
import h5py
import igraph as ig
import leidenalg
import numpy as np
import polars as pl
from scib_metrics import kbet, silhouette_label
from scib_metrics.nearest_neighbors import jax_approx_min_k
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import average_precision_score as AUPRC
from sklearn.metrics import f1_score as F1
from sklearn.metrics import normalized_mutual_info_score as NMI
from sklearn.metrics import roc_auc_score as AUROC

import deeptan.constants as const


def format_ticks(x, pos):
    r"""
    Format the ticks on the x-axis of a plot.
    """
    return f"{x:.2f}"


def transform_ct_df(df: pl.DataFrame):
    melted_df = df.unpivot(index=["obs_names"], on=df.columns[1:], variable_name="ct", value_name="value")

    filtered_df = melted_df.filter(pl.col("value") == 1)

    result_df = filtered_df.select(["obs_names", "ct"])

    celltypes_ = result_df["ct"].to_list()
    celltypes = [ct.replace("ct_", "") for ct in celltypes_ if ct.startswith("ct_")]
    result_df = result_df.hstack(pl.DataFrame({"ct_": celltypes})).drop("ct").rename({"ct_": "ct"})

    return result_df


def read_batch_from_h5ad(h5ad_path: str) -> pl.DataFrame:
    r"""
    Read batch info from an h5ad file.
    """
    adata = anndata.read_h5ad(h5ad_path)
    # batch_info = adata.obs["batch"].to_list()
    possible_batch_keys = ["batch", "orig.ident", "Orig.ident"]
    for _k in possible_batch_keys:
        if _k in adata.obs.columns:
            batch_info = adata.obs[_k].to_list()
            break
    else:
        raise ValueError("No batch information found in the h5ad file. Available keys: ", possible_batch_keys)

    result_df = pl.DataFrame({"obs_names": adata.obs_names.to_list(), "batch": batch_info})
    return result_df


def decode_item(item):
    if isinstance(item, h5py.Dataset):
        data = item[()]
        if isinstance(data, bytes):
            return data.decode("utf-8")  # Handle byte strings
        return data
    elif isinstance(item, h5py.Group):
        return {k: decode_item(v) for k, v in item.items()}
    return item


class MetricsDictMaker:
    r"""
    Make a dictionary of metrics for predictions of **a dataset**.
    """

    def __init__(
        self,
        predictions_dir: str,
        true_data_dir: str,
        softmax_pred_labels: bool = True,
        orig_h5ad: str | None = None,
        s_task: Optional[str] = None,
        s_split: Optional[str] = None,
    ):
        self.predictions_dir = predictions_dir
        self.true_data_dir = true_data_dir
        self.softmax_pred_labels = softmax_pred_labels
        self.orig_h5ad = orig_h5ad
        self.xxx_data_df = {}

        self.metrics_dict = {
            "identify": None,
            "prediction": {},  # Will store h5 file paths and metadata
            "true": {},
            "metrics": {
                "recon": {},
                "label": {},
                "cluster": {},
            },
            "summary_recon": None,
            "summary_label": None,
            "summary_clustering": None,
        }

        self.ident = self.detect_predictions(s_task, s_split)
        self.fnames = self.ident["fname"].to_list()
        self.metrics_dict["identify"] = self.ident

    def run(self):
        self.load_predictions()
        self.load_true()
        self.compute_all_metrics()
        self.make_metrics_summary()

    def load_predictions(self):
        """Loads predictions from pickle files."""
        for _fname in self.fnames:
            _path = self.ident.filter(pl.col("fname") == _fname)["path"].item()
            self.metrics_dict["prediction"][_fname] = {"path": _path}
            # with open(_path, "rb") as f:
            #     self.metrics_dict["prediction"][_fname] = pickle.load(f)
            # self.metrics_dict["prediction"][_fname]["X"] = np.squeeze(self.metrics_dict["prediction"][_fname]["node_recon_all"], axis=-1)
            # self.metrics_dict["prediction"][_fname]["node_recon_all"] = None
            # self.metrics_dict["prediction"][_fname]["y"] = self.metrics_dict["prediction"][_fname]["labels"]
            # if self.softmax_pred_labels:
            #     self.metrics_dict["prediction"][_fname]["y"] = self.softmax(self.metrics_dict["prediction"][_fname]["y"])

    def load_true(self):
        """Load true data from parquet files."""
        for i, _split in enumerate(const.dkey.splits):
            seeds_uniq = self.ident["seed_num"].unique().sort().to_list()
            for _seed in seeds_uniq:
                self.metrics_dict["true"][f"seed_{_seed}_{_split}"] = {}

                # Use the 1st fname of seed xx to get feature names
                _fname = self.ident.filter(pl.col("seed_num") == _seed)["fname"].to_list()[0]
                feature_names_seed_xx = ["obs_names"] + list(self._get_dict_node_names(_fname).keys())

                _path = os.path.join(self.true_data_dir, f"split_{_seed}_{i}.parquet")
                self.xxx_data_df[f"seed_{_seed}_{_split}"] = pl.read_parquet(_path).select(feature_names_seed_xx)
                xxx_data = self.xxx_data_df[f"seed_{_seed}_{_split}"].drop(["obs_names"]).to_numpy()
                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["X"] = np.log1p(xxx_data)

                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["obs_names"] = self.xxx_data_df[f"seed_{_seed}_{_split}"]["obs_names"].to_list()
                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["feature_names"] = feature_names_seed_xx[1:]

                # Load true labels
                _labels_df = pl.read_parquet(os.path.join(self.true_data_dir, "celltypes_onehot.parquet")).rename({"bc": "obs_names"})
                _labels_all = transform_ct_df(_labels_df)

                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y_df_flatten"] = _labels_all.join(self.xxx_data_df[f"seed_{_seed}_{_split}"].select(["obs_names"]), on="obs_names", how="right").select(["obs_names", "ct"])
                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y_df"] = _labels_df.join(self.xxx_data_df[f"seed_{_seed}_{_split}"].select(["obs_names"]), on="obs_names", how="right")
                self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y"] = self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y_df"].drop("obs_names").to_numpy()

    def _get_dict_node_names(self, fname: str) -> Dict[str, int]:
        """Read dict_node_names from the h5 file."""
        path = self.metrics_dict["prediction"][fname]["path"]
        with h5py.File(path, "r") as f:
            if "dict_node_names" not in f:
                raise KeyError("dict_node_names not found in the h5 file.")
            return {k: decode_item(v) for k, v in f["dict_node_names"].items()}

    def _get_label_names(self, fname: str) -> list:
        """Read label_names from the h5 file."""
        path = self.metrics_dict["prediction"][fname]["path"]
        with h5py.File(path, "r") as f:
            if "label_names" in f:
                return [n.decode("utf-8") if isinstance(n, bytes) else n for n in f["label_names"][()]]
        return []

    def _read_h5_dataset(self, fname: str, dataset_name: str) -> np.ndarray:
        """Lazily read a specific dataset from an h5 file."""
        path = self.metrics_dict["prediction"][fname]["path"]
        with h5py.File(path, "r") as f:
            if dataset_name in f:
                return f[dataset_name][()]

    def make_metrics_summary(self):
        # For recon
        _dfs = []
        for _fname in self.fnames:
            _tmp_df = self.metrics_dict["metrics"]["recon"][_fname]["averaged"]
            _tmp_df = pl.DataFrame({"fname": [_fname] * _tmp_df.shape[0]}).hstack(_tmp_df)
            _dfs.append(_tmp_df)
        _tmp: pl.DataFrame = pl.concat(_dfs)
        _tmp = pl.DataFrame({"Capability": ["Feature Reconstruction"] * _tmp.shape[0]}).hstack(_tmp)
        _tmp = pl.DataFrame({"Method": ["DeepTAN"] * _tmp.shape[0]}).hstack(_tmp)
        self.metrics_dict["summary_recon"] = _tmp.join(self.ident, on="fname", how="left")

        # For label
        _dfs = []
        _confusion_matrices = []
        _label_names = []
        for _fname in self.fnames:
            _tmp_df = self.metrics_dict["metrics"]["label"][_fname]["df"]
            _tmp_df = pl.DataFrame({"fname": [_fname] * _tmp_df.shape[0]}).hstack(_tmp_df)
            _dfs.append(_tmp_df)

            _tmp_cm = self.metrics_dict["metrics"]["label"][_fname]["confusion_matrix"]
            _confusion_matrices.append(_tmp_cm)
            _label_names.append(self.metrics_dict["metrics"]["label"][_fname]["label_names"])

        _tmp: pl.DataFrame = pl.concat(_dfs)
        _tmp = pl.DataFrame({"Capability": ["Labelling"] * _tmp.shape[0]}).hstack(_tmp)
        _tmp = pl.DataFrame({"Method": ["DeepTAN"] * _tmp.shape[0]}).hstack(_tmp)
        self.metrics_dict["summary_label"] = _tmp.join(self.ident, on="fname", how="left")
        self.metrics_dict["confusion_matrices"] = _confusion_matrices
        self.metrics_dict["label_names"] = _label_names

        # For clustering
        _dfs = []
        for _fname in self.fnames:
            _tmp_df = self.metrics_dict["metrics"]["cluster"][_fname]
            _tmp_df = pl.DataFrame({"fname": [_fname] * _tmp_df.shape[0]}).hstack(_tmp_df)
            _dfs.append(_tmp_df)
        _tmp: pl.DataFrame = pl.concat(_dfs)
        _tmp = pl.DataFrame({"Capability": ["Clustering"] * _tmp.shape[0]}).hstack(_tmp)
        _tmp = pl.DataFrame({"Method": ["DeepTAN"] * _tmp.shape[0]}).hstack(_tmp)
        self.metrics_dict["summary_clustering"] = _tmp.join(self.ident, on="fname", how="left")

    def compute_all_metrics(self):
        """Computes all metrics for the predictions."""

        # For reconstruction
        print("\nComputing metrics for recon...")
        for _fname in self.fnames:
            _seed = self.ident.filter(pl.col("fname") == _fname)["seed_num"].item()
            _split = self.ident.filter(pl.col("fname") == _fname)["split"].item()

            X_true = self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["X"]
            X_pred = np.squeeze(self._read_h5_dataset(_fname, "node_recon_all"), axis=-1)

            _calculator = RegressionMetricsCalculator(X_true, X_pred)
            self.metrics_dict["metrics"]["recon"][_fname] = _calculator.calculate_all_metrics()

        # For label
        print("Computing metrics for label...")
        for _fname in self.fnames:
            _seed = self.ident.filter(pl.col("fname") == _fname)["seed_num"].item()
            _split = self.ident.filter(pl.col("fname") == _fname)["split"].item()

            y_true = self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y"]
            y_pred = self._read_h5_dataset(_fname, "labels")
            label_names = self._get_label_names(_fname)

            if len(y_pred) == 0:
                continue

            if self.softmax_pred_labels:
                y_pred = self.softmax(y_pred)

            _calculator = MulticlassMetricsCalculator(y_true, y_pred, label_names)
            _metrics, _confusion_matrix = _calculator.calculate_metrics()
            self.metrics_dict["metrics"]["label"][_fname] = {
                "df": pl.DataFrame(_metrics),
                "confusion_matrix": _confusion_matrix,
                "label_names": _calculator._label_names,
            }

        # For cluster
        print("Calculating cluster metrics...\n")
        for _fname in self.fnames:
            _seed = self.ident.filter(pl.col("fname") == _fname)["seed_num"].item()
            _split = self.ident.filter(pl.col("fname") == _fname)["split"].item()

            if self.orig_h5ad is not None:
                batch_info = self._read_batch_from_h5ad(self.orig_h5ad, _seed, _split)["batch"].to_numpy()
            else:
                batch_info = np.repeat("batch_0", len(self.xxx_data_df[f"seed_{_seed}_{_split}"]))

            y_true = self.metrics_dict["true"][f"seed_{_seed}_{_split}"]["y"]

            g_embedding = self._read_h5_dataset(_fname, "g_embedding")

            _calculator = ClusteringMetricsCalculator(
                true_labels=y_true,
                pred_labels=None,
                batches=batch_info,
                X_emb=g_embedding,
            )
            self.metrics_dict["metrics"]["cluster"][_fname] = pl.DataFrame(_calculator.calculate_metrics())

    def detect_predictions(self, s_task: Optional[str], s_split: Optional[str]):
        """Detects .h5 files in the predictions directory."""
        _fnames = []
        _seeds = []
        _seeds_num = []
        _tasks = []
        _splits = []
        _paths = []
        for _file in os.listdir(self.predictions_dir):
            if not _file.endswith(".h5"):
                continue
            _path = os.path.join(self.predictions_dir, _file)
            _prop = _file.removesuffix(".h5").split("+")

            _task = _prop[2]
            _split = _prop[3]
            if s_task is not None and _task != s_task:
                print(f"Skipping task {_task} as it does not match the specified task {s_task}.")
                continue
            if s_split is not None and _split != s_split:
                print(f"Skipping split {_split} as it does not match the specified split {s_split}.")
                continue
            print(f"Found file📄 {_file} with task🎯 {_task} and split🍰 {_split}.")

            _seeds.append(_prop[1])
            _seeds_num.append(int(_prop[1].replace("seed_", "")))
            _tasks.append(_task)
            _splits.append(_split)
            _paths.append(_path)
            _fnames.append(_file)
        return pl.DataFrame(
            {
                "fname": _fnames,
                "seed": _seeds,
                "seed_num": _seeds_num,
                "task": _tasks,
                "split": _splits,
                "path": _paths,
            }
        )

    def _read_batch_from_h5ad(self, h5ad_path: str, _seed: int, _split: str) -> pl.DataFrame:
        r"""
        Read batch info from an h5ad file.
        """
        result_df = read_batch_from_h5ad(h5ad_path)
        return result_df.join(self.xxx_data_df[f"seed_{_seed}_{_split}"].select(["obs_names"]), on="obs_names", how="right")

    def softmax(self, x, axis=-1):
        e_x = np.exp(x - np.max(a=x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)


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
            "spearman": self._calculate_spearman,
        }

    def _validate_arrays(self):
        if self._true.shape != self._pred.shape:
            raise ValueError(f"Input arrays must have the same shape. True array shape: {self._true.shape}, Pred array shape: {self._pred.shape}")
        if len(self._true.shape) != 2:
            raise ValueError(f"Input arrays must be 2-dimensional. True array shape: {self._true.shape}, Pred array shape: {self._pred.shape}")

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

    def _calculate_spearman(self, axis: int) -> np.ndarray:
        if axis == 1:  # Sample-wise (rows)
            return np.array([spearmanr(t_row, p_row)[0] for t_row, p_row in zip(self._true, self._pred)])
        else:  # Feature-wise (columns)
            return np.array([spearmanr(self._true[:, i], self._pred[:, i])[0] for i in range(self.n_features)])

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
        results = {
            "sample_metrics": self.calculate_sample_metrics(),
            "feature_metrics": self.calculate_feature_metrics(),
            "shape": {"n_samples": self.n_samples, "n_features": self.n_features},
        }

        # Create summary DataFrame
        summary_data = []
        for metric in self.metric_functions.keys():
            summary_data.append(
                {
                    "metric": metric,
                    "sample_mean": results["sample_metrics"][metric]["mean"],
                    "feature_mean": results["feature_metrics"][metric]["mean"],
                }
            )
        df = pl.DataFrame(summary_data)

        results.update({"averaged": df})

        return results


class MulticlassMetricsCalculator:
    r"""
    Class to compute metrics for multi-class classification tasks.
    Supported metrics include weighted F1, macro F1, micro F1, AUROC, AUPRC, accuracy, precision, and recall.
    """

    def __init__(self, true_labels: np.ndarray, pred_probs: np.ndarray, label_names: List[str]):
        """
        Initialize the calculator with true labels and predicted probabilities.

        Args:
            true_labels (np.ndarray): 1D array of true class labels.
            pred_probs (np.ndarray): 2D array of predicted probabilities for each class.
            num_classes (int): Number of classes in the classification task.
        """
        self._true_df = true_labels
        self._pred_probs = pred_probs
        self._label_names = label_names
        self._num_classes = len(label_names)

        # Convert predicted probabilities to predicted labels
        self._pred_labels = np.argmax(pred_probs, axis=1)
        self._true = np.argmax(self._true_df, axis=1)

        # Find unique labels to avoid errors in confusion matrix
        self._unique_labels = np.unique(np.concatenate((self._true, self._pred_labels)))
        self._unique_labels.sort()
        self._label_names = [self._label_names[i] for i in self._unique_labels]
        _label_map = {label: i for i, label in enumerate(self._unique_labels)}
        self._true = np.array([_label_map[label] for label in self._true])
        self._pred_labels = np.array([_label_map[label] for label in self._pred_labels])
        self._num_classes = len(self._unique_labels)

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
        # if self._true.shape[0] != self._pred_probs.shape[0]:
        #     raise ValueError("True labels and predicted probabilities must have the same number of samples.")
        # if self._pred_probs.shape[1] != self._num_classes:
        #     raise ValueError("Number of columns in predicted probabilities must match the number of classes.")
        if not np.allclose(self._pred_probs.sum(axis=1), 1.0, atol=1e-3):
            raise ValueError("Predicted probabilities must be softmax normalized (rows should sum to 1)")

    def _calculate_weighted_f1(self):
        return F1(self._true, self._pred_labels, average="weighted", zero_division=0.0)

    def _calculate_macro_f1(self):
        return F1(self._true, self._pred_labels, average="macro", zero_division=0.0)

    def _calculate_micro_f1(self):
        return F1(self._true, self._pred_labels, average="micro", zero_division=0.0)

    def _calculate_auroc(self):
        try:
            return AUROC(self._true_df, self._pred_probs, multi_class="ovr", average="weighted")
        except ValueError:
            return np.nan  # Return NaN if AUROC cannot be computed

    def _calculate_auprc(self):
        return AUPRC(self._true_df, self._pred_probs, average="weighted")

    def _calculate_accuracy(self):
        return accuracy_score(self._true, self._pred_labels)

    def _calculate_weighted_precision(self):
        return precision_score(self._true, self._pred_labels, average="weighted", zero_division=0.0)

    def _calculate_weighted_recall(self):
        return recall_score(self._true, self._pred_labels, average="weighted", zero_division=0.0)

    def _confusion_matrix(self):
        return confusion_matrix(self._true, self._pred_labels)

    def calculate_metrics(self):
        """
        Calculate all metrics and return them as a dictionary.

        Returns:
            Dict[str, float]: Dictionary of metric results.
        """
        metrics = {}
        for name, func in self.metric_functions.items():
            metrics[name] = func()
        return metrics, self._confusion_matrix()
        # return metrics


class ClusteringMetricsCalculator:
    r"""
    Class to compute clustering metrics such as ARI, ASW, NMI, and kBET.
    """

    def __init__(self, true_labels: np.ndarray, pred_labels: Optional[np.ndarray], batches: np.ndarray, X_emb: np.ndarray, n_neighbors=50):
        """
        Initialize the calculator with true labels and predicted labels.

        Args:
            true_labels (np.ndarray): 2D array of true cluster labels.
            pred_labels (np.ndarray): 2D array of predicted cluster labels.
            batches (np.ndarray): 1D array representing batch information for each cell.
            X_emb (np.ndarray): 2D array of embedded data.
            n_neighbors (int): Number of neighbors to consider for nearest neighbor calculations. Defaults to 50.
        """
        self._true = true_labels.argmax(axis=1)

        self.batch = batches
        self.X_emb = X_emb
        self.nn_result = jax_approx_min_k(X=self.X_emb, n_neighbors=n_neighbors)

        # Calculate Leiden labels
        self.leiden_labels = self._calculate_leiden_labels()

        if pred_labels is None:
            self._pred = self.leiden_labels
        else:
            self._pred = pred_labels.argmax(axis=1)

        # Define metric calculation functions
        self.metric_functions = {
            "kbet": self._calculate_kbet,
            "asw_true_label": self._calculate_asw_true_label,
            "asw_pred_label": self._calculate_asw_pred_label,
            "ari": self._calculate_ari,
            "nmi": self._calculate_nmi,
            "ami": self._calculate_ami,
            "ari_leiden": self._calculate_ari_leiden,
            "nmi_leiden": self._calculate_nmi_leiden,
            "ami_leiden": self._calculate_ami_leiden,
        }

    def _calculate_asw_true_label(self):
        return silhouette_label(X=self.X_emb, labels=self._true)

    def _calculate_asw_pred_label(self):
        return silhouette_label(X=self.X_emb, labels=self._pred)

    def _calculate_kbet(self):
        _result = kbet(X=self.nn_result, batches=self.batch)
        return _result[0]

    def _calculate_ari(self) -> float:
        return ARI(self._true, self._pred)

    def _calculate_nmi(self):
        return NMI(self._true, self._pred)

    def _calculate_ami(self):
        return AMI(self._true, self._pred)

    def _calculate_leiden_labels(self):
        """
        Calculate Leiden clustering labels from the embedding.

        Returns:
            np.ndarray: Leiden cluster labels.
        """
        indices = self.nn_result.indices
        n_samples = self.X_emb.shape[0]

        # Create a sparse adjacency matrix
        rows = np.repeat(np.arange(n_samples), self.nn_result.n_neighbors)
        cols = indices.ravel()  # Assuming indices is a 1D array of indices
        data = np.ones_like(rows)
        adj_matrix = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        adj_matrix = adj_matrix + adj_matrix.T

        dense_adj_matrix = adj_matrix.toarray()
        g = ig.Graph.Adjacency(dense_adj_matrix.astype(bool).tolist())
        partition_type = leidenalg.RBConfigurationVertexPartition
        partition = leidenalg.find_partition(g, partition_type, n_iterations=-1)
        return np.array(partition.membership)

    def _calculate_ari_leiden(self) -> float:
        return ARI(self._true, self.leiden_labels)

    def _calculate_nmi_leiden(self):
        return NMI(self._true, self.leiden_labels)

    def _calculate_ami_leiden(self):
        return AMI(self._true, self.leiden_labels)

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

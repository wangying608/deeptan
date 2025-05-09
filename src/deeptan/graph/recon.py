import os
import pickle
from typing import Any, List, Optional

import h5py
import numpy as np
import polars as pl
import torch
from lightning import Trainer
from litdata import StreamingDataLoader, StreamingDataset
from tqdm import tqdm

import deeptan.constants as const
from deeptan.graph.model import DeepTAN
from deeptan.utils.uni import collate_fn, get_map_location


def predict(
    model_ckpt_path: str,
    litdata_dir: str,
    output_pkl_path: str,
    map_location: Optional[str] = None,
    batch_size: int = 8,
    save_h5: bool = True,
):
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    # Load a DeepTAN model
    path_hparams = os.path.join(os.path.dirname(model_ckpt_path), "version_0", "hparams.yaml")
    if os.path.exists(path_hparams):
        model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location), hparams_file=path_hparams)
    else:
        model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location))

    # Freeze the model
    model.eval()
    model.freeze()

    # Load the LitData dataset
    dataloader = StreamingDataLoader(StreamingDataset(litdata_dir, max_cache_size="10GB"), batch_size=batch_size, collate_fn=collate_fn)

    # Predict
    trainer = Trainer(logger=False)
    results = trainer.predict(model=model, dataloaders=dataloader)

    assert results is not None, "No results returned from prediction"

    # Read feature names and label names
    with open(os.path.join(os.path.dirname(litdata_dir), const.fname.litdata_others2save_pkl), "rb") as f:
        feature_dict_and_label_dim: dict = pickle.load(f)
    label_names = pl.read_parquet(os.path.join(os.path.dirname(litdata_dir), const.fname.label_class_onehot)).columns
    feature_dict_and_label_dim.update({"label_names": label_names})

    process_results(results, output_pkl_path, feature_dict_and_label_dim, save_h5, False)
    return None


def process_results(pickle_file: str | Any, output_pkl: str, others2save: Optional[dict] = None, save_h5: bool = True, only_return: bool = False):
    r"""
    Process the results of DeepTAN from the pickle file and save them to a numpy pickle file.
    """
    if isinstance(pickle_file, str):
        # Load the results
        with open(pickle_file, "rb") as f:
            results = pickle.load(f)
    else:
        results = pickle_file

    g_embedding = []
    node_recon = []
    node_recon_for_loss = []
    node_recon_all = []
    labels = []

    for i_batch in range(len(results)):
        g_embedding.append(results[i_batch]["embedding"])
        node_recon.append(results[i_batch]["node_recon"])
        node_recon_for_loss.append(results[i_batch]["node_recon_for_loss"])
        node_recon_all.append(results[i_batch]["node_recon_for_loss_all"])
        labels.append(results[i_batch]["label_pred"])

    g_embedding = torch.cat(g_embedding, dim=0)
    node_recon = torch.cat(node_recon, dim=0)
    node_recon_all = torch.cat(node_recon_all, dim=0)
    labels = torch.cat(labels, dim=0)

    # Convert to numpy arrays for further processing
    g_embedding_np = g_embedding.detach().cpu().numpy()
    node_recon_np = node_recon.detach().cpu().numpy()
    node_recon_all_np = node_recon_all.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # Save the results as a dictionary in a pickle file
    results_dict = {
        "g_embedding": g_embedding_np,
        "node_recon": node_recon_np,
        "node_recon_all": node_recon_all_np,
        "labels": labels_np,
    }

    # For each key in the results dictionary, print data shape
    for key in results_dict.keys():
        print(f"Key: {key}, Shape: {results_dict[key].shape}")
    # EXAMPLE OUTPUT:
    # dict_keys(['g_embedding', 'node_recon', 'node_recon_all', 'labels'])
    # Key: g_embedding, Shape: (150, 256)
    # Key: node_recon, Shape: (150, 13461, 128)
    # Key: node_recon_all, Shape: (150, 13461, 1)
    # Key: labels, Shape: (150, 1)

    if others2save is not None:
        results_dict.update(others2save)

    if only_return:
        return results_dict

    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    if save_h5:
        if not output_pkl.endswith(".h5"):
            output_pkl = output_pkl.replace(".pkl", ".h5") if output_pkl.endswith(".pkl") else output_pkl + ".h5"
        print(f"Saving results to {output_pkl}")
        with h5py.File(output_pkl, "w") as f:
            for key, value in results_dict.items():
                f.create_dataset(key, data=value, compression="gzip", compression_opts=5)

    else:
        output_pkl = output_pkl if output_pkl.endswith(".pkl") else output_pkl + ".h5"
        print(f"Saving results to {output_pkl}")
        with open(output_pkl, "wb") as f:
            pickle.dump(results_dict, f)


class FeaturePerturbationStreamingDataset(StreamingDataset):
    def __getitem__(self, index):
        _gdata = super().__getitem__(index)
        node_names = _gdata.node_names[0]

        # Get indices of features to perturb
        feat_idx = [i for i, _name in enumerate(node_names) if _name in self.feature_names2perturb]
        if len(feat_idx) == 0:
            return _gdata

        # Get the values to overwrite
        _feat_names = [node_names[i] for i in feat_idx]
        _overwrite_values = []
        for _name in _feat_names:
            # Search for the position of _name in self.feature_names2perturb and get the corresponding overwrite value
            _overwrite_values.append(self.overwrite_values[self.feature_names2perturb.index(_name)])

        # Clone the graph data
        _gdata_clone = _gdata.clone()
        _x = _gdata_clone.x.clone()
        assert _x.ndim == 2, f"Expected _x.ndim to be 2, but got {_x.ndim}"

        # Perturb the features
        _x[feat_idx, :] = torch.tensor(_overwrite_values, dtype=_gdata.x.dtype, device=_gdata.x.device)
        _gdata_clone.x = _x
        return _gdata_clone

    def _perturb(self, feature_names: List[str], overwrite_values: List[Any]):
        if len(feature_names) == 0:
            raise ValueError("feature_idx cannot be empty")
        if len(feature_names) != len(overwrite_values):
            raise ValueError("feature_idx and overwrite_values must have the same length")
        self.feature_names2perturb = feature_names
        self.overwrite_values = overwrite_values


def predict_perturbation(
    model_ckpt_path: str,
    litdata_dir: str,
    output_path: str,
    n_perturbations: int = 5,
    map_location: Optional[str] = None,
    overwrite_files: bool = True,
):
    r"""
    Predict with feature perturbations by analyzing original feature ranges and perturbing each feature multiple times.

    Args:
        model_ckpt_path: Path to model checkpoint
        litdata_dir: Path to LitData directory
        output_path: Output path for results
        n_perturbations: Number of perturbations per feature
        map_location: Device to load model on
        overwrite_files: Whether to overwrite existing files
    """
    batch_size: int = 1
    # Load original dataset to analyze feature statistics
    orig_dataloader = StreamingDataLoader(StreamingDataset(litdata_dir, max_cache_size="10GB"), batch_size=batch_size, collate_fn=collate_fn)
    feature_stats = {}
    all_feat_values = {}

    for _gdata in orig_dataloader:
        _node_names = _gdata.node_names[0]
        _x = _gdata.x
        for i, _name in enumerate(_node_names):
            if _name not in all_feat_values:
                all_feat_values[_name] = _x[i].unsqueeze(0)
            else:
                all_feat_values[_name] = torch.cat([all_feat_values[_name], _x[i].unsqueeze(0)], dim=0)

    node_names = list(all_feat_values.keys())

    # Compute stats for each feature
    for feat_idx, feat_name in enumerate(node_names):
        feat_vals = all_feat_values[feat_name]
        feature_stats[feat_name] = {
            "mean": feat_vals.mean().item(),
            "std": feat_vals.std().item(),
            "min": feat_vals.min().item(),
            "max": feat_vals.max().item(),
        }

    # Prepare perturbation values for each feature
    perturbation_plans = []
    for feat_name in node_names:
        stats = feature_stats[feat_name]
        # Generate perturbation values within ±2 std of mean, clamped to feature range
        base_values = torch.linspace(
            max(stats["min"], stats["mean"] - 2 * stats["std"]),
            min(stats["max"], stats["mean"] + 2 * stats["std"]),
            n_perturbations,
        ).tolist()
        perturbation_plans.append((feat_name, base_values))

    # =========================================================================
    # Load additional metadata like in predict()
    with open(os.path.join(os.path.dirname(litdata_dir), const.fname.litdata_others2save_pkl), "rb") as f:
        feature_dict_and_label_dim = pickle.load(f)
    label_names = pl.read_parquet(os.path.join(os.path.dirname(litdata_dir), const.fname.label_class_onehot)).columns
    feature_dict_and_label_dim.update({"label_names": label_names})
    # print(feature_dict_and_label_dim.keys())
    # for _k in feature_dict_and_label_dim.keys():
    #     print(_k, type(feature_dict_and_label_dim[_k]))
    # =================
    # dict_keys(['dict_node_names', 'output_g_label_dim', 'label_names'])
    # dict_node_names <class 'dict'>
    # output_g_label_dim <class 'int'>
    # label_names <class 'list'>

    # Prepare output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _output_path = output_path if output_path.endswith(".h5") else output_path + ".h5"
    if os.path.exists(_output_path):
        if overwrite_files:
            os.remove(_output_path)
        else:
            raise FileExistsError(f"File {_output_path} already exists. Set overwrite_files=True to overwrite.")

    # Load model
    path_hparams = os.path.join(os.path.dirname(model_ckpt_path), "version_0", "hparams.yaml")
    if os.path.exists(path_hparams):
        model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location), hparams_file=path_hparams)
    else:
        model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location))
    model.eval()
    model.freeze()

    with h5py.File(_output_path, "w") as f:
        # Save feature stats and metadata first
        metadata_grp = f.create_group("metadata")

        for key, value in feature_dict_and_label_dim.items():
            try:
                if isinstance(value, (list, tuple)) and all(isinstance(x, str) for x in value):
                    # Handle string lists
                    dt = h5py.string_dtype(encoding="utf-8")
                    metadata_grp.create_dataset(key, data=np.array(value, dtype=object), dtype=dt)
                elif isinstance(value, dict):
                    # Handle nested dictionaries
                    sub_grp = metadata_grp.create_group(key)
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, (list, tuple)) and all(isinstance(x, str) for x in subvalue):
                            dt = h5py.string_dtype(encoding="utf-8")
                            sub_grp.create_dataset(subkey, data=np.array(subvalue, dtype=object), dtype=dt)
                        else:
                            sub_grp.create_dataset(subkey, data=subvalue)
                else:
                    # Handle arrays and other numeric data
                    metadata_grp.create_dataset(key, data=value)
            except Exception as e:
                print(f"Warning: Failed to save {key} to HDF5: {str(e)}")
                continue

        # Save feature stats
        feat_stats_grp = f.create_group("feature_stats")
        for feat_name, stats in feature_stats.items():
            feat_grp = feat_stats_grp.create_group(feat_name)
            for stat_name, stat_val in stats.items():
                feat_grp.create_dataset(stat_name, data=stat_val)

        # Save perturbation plans
        perturb_plans_grp = f.create_group("perturbation_plans")
        for i, (feat_name, values) in enumerate(perturbation_plans):
            perturb_plans_grp.create_dataset(f"{i}_feature", data=feat_name)
            perturb_plans_grp.create_dataset(f"{i}_values", data=values)

        # Create group for results
        results_grp = f.create_group("perturbation_results")

        # Process each feature perturbation
        for feat_idx, (feat_name, perturb_values) in enumerate(tqdm(perturbation_plans, desc="Perturbing features")):
            # Create perturbed dataset for this feature
            perturb_dataset = FeaturePerturbationStreamingDataset(litdata_dir, max_cache_size="10GB")

            # Process each perturbation value
            for val_idx, value in enumerate(perturb_values):
                perturb_dataset._perturb(feature_names=[feat_name], overwrite_values=[value])
                dataloader = StreamingDataLoader(perturb_dataset, batch_size=batch_size, collate_fn=collate_fn)
                trainer = Trainer(logger=False, devices=1)
                results = trainer.predict(model=model, dataloaders=dataloader)

                if results:
                    # Save results immediately
                    results_dict = process_results(results, "", feature_dict_and_label_dim, save_h5=False, only_return=True)
                    result_grp = results_grp.create_group(f"{feat_idx}_{val_idx}")
                    result_grp.create_dataset("perturbed_feature", data=feat_name)
                    result_grp.create_dataset("perturb_value", data=value)

                    for key, value in results_dict.items():
                        try:
                            if isinstance(value, np.ndarray) and value.dtype == object:
                                # Convert object arrays to string arrays
                                dt = h5py.string_dtype(encoding="utf-8")
                                result_grp.create_dataset(key, data=np.array(value, dtype=object), dtype=dt)
                            else:
                                result_grp.create_dataset(key, data=value, compression="gzip", compression_opts=5)
                        except Exception as e:
                            print(f"Warning: Failed to save {key} to HDF5: {str(e)}")
                            continue

    print(f"Saved output to {_output_path}")

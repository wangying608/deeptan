import os
import pickle
from typing import Any, Optional

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
    dataloader = StreamingDataLoader(StreamingDataset(litdata_dir), batch_size=batch_size, collate_fn=collate_fn)

    # Predict
    trainer = Trainer(logger=False)
    results = trainer.predict(model=model, dataloaders=dataloader)

    assert results is not None, "No results returned from prediction"

    # Read feature names and label names
    with open(os.path.join(os.path.dirname(litdata_dir), const.fname.litdata_others2save_pkl), "rb") as f:
        feature_dict_and_label_dim: dict = pickle.load(f)
    label_names = pl.read_parquet(os.path.join(os.path.dirname(litdata_dir), const.fname.label_class_onehot)).columns
    feature_dict_and_label_dim.update({"label_names": label_names})

    process_results(results, output_pkl_path, feature_dict_and_label_dim)
    return None


def process_results(pickle_file: str | Any, output_pkl: str, others2save: Optional[dict] = None):
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
    print(results_dict.keys())

    if not output_pkl.endswith(".pkl"):
        output_pkl += ".pkl"
    print(f"Saving results to {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(results_dict, f)

    # return output_pkl


def compute_feature_correlations(
    output_npz: str,
    pickle_path: Optional[str] = None,
    node_recon: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    device: Optional[str] = None,
):
    """
    Compute feature correlation matrix.

    Args:
        output_npy: Path to output npy file containing correlation matrix
        pickle_path: Path to pickle file containing processed results
        node_recon: 3D array of shape (n_samples, n_features, dim)
        labels: 1D array of shape (n_samples,)
        device: Device to use for computation

    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
    device = get_map_location(device)
    # --- Input Validation ---
    if pickle_path is not None:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        node_recon = data["node_recon"]
        labels = data["labels"].squeeze()
    assert node_recon is not None and labels is not None, "Both node_recon and labels must be provided"

    n_samples, n_features, dim = node_recon.shape

    # Convert data to PyTorch tensors and move to device
    labels_tensor = torch.from_numpy(labels).float().to(device)

    # Precompute label statistics
    y_mean = torch.mean(labels_tensor)
    y_centered = labels_tensor - y_mean
    sum_y_sq = torch.sum(y_centered**2)

    # Initialize accumulators on device
    sum_x = torch.zeros((n_features, n_features), device=device)
    sum_x_sq = torch.zeros((n_features, n_features), device=device)
    sum_xy = torch.zeros((n_features, n_features), device=device)

    # Process each sample
    for s in tqdm(range(n_samples), desc="Processing samples"):
        sample_feat = torch.from_numpy(node_recon[s]).float().to(device)

        # Compute absolute dot products using matrix multiplication
        # dot_products = torch.abs(torch.mm(sample_feat, sample_feat.T))  # (n_features, n_features)
        dot_products = torch.mm(sample_feat, sample_feat.T)

        # Update accumulators
        sum_x += dot_products
        sum_x_sq += dot_products**2
        sum_xy += dot_products * y_centered[s]

    # Compute final statistics
    sum_x_centered_sq = sum_x_sq - (sum_x**2) / n_samples

    # Compute correlation matrix
    denominator = torch.sqrt(sum_x_centered_sq * sum_y_sq)
    with torch.no_grad():
        corr_matrix = torch.where(denominator != 0, sum_xy / denominator, torch.zeros_like(denominator))

    # Compute weighted correlation matrix
    mean_x_matrix = sum_x / n_samples
    mean_x_matrix = torch.abs(mean_x_matrix)
    # [0,1]
    mean_x_matrix = mean_x_matrix / mean_x_matrix.max()
    corr_weighted = corr_matrix * mean_x_matrix
    output_weighted = corr_weighted.cpu().numpy()
    print(f"\n🔥Correlation matrix shape: {output_weighted.shape}")

    # Move results back to CPU and convert to numpy
    output = corr_matrix.cpu().numpy()

    # Save output_weighted and output to a npz
    np.savez(output_npz, corr_matrix=output, corr_weighted=output_weighted)

import os
import pickle
from typing import Optional

import numpy as np
import torch
from lightning import Trainer
from litdata import StreamingDataLoader, StreamingDataset
from tqdm import tqdm

from deeptan.graph.model import DeepTAN
from deeptan.utils.uni import collate_fn, get_map_location


def predict(
    model_ckpt_path: str,
    litdata_dir: str,
    output_pickle_path: str,
    map_location: Optional[str] = None,
    batch_size: int = 1,
):
    # Load a DeepTAN model
    model = DeepTAN.load_from_checkpoint(model_ckpt_path, map_location=get_map_location(map_location))
    # Freeze the model
    model.eval()
    model.freeze()

    # Load the LitData dataset
    dataloader = StreamingDataLoader(StreamingDataset(litdata_dir), batch_size=batch_size, collate_fn=collate_fn)

    # Predict
    trainer = Trainer(logger=False)
    results = trainer.predict(model=model, dataloaders=dataloader)

    assert results is not None
    # Save the results to a pickle file
    with open(output_pickle_path, "wb") as f:
        pickle.dump(results, f)


def process_results(pickle_path: str, output_pkl: str):
    # Load the results
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
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

    print(results_dict.keys())
    # For each key in the results dictionary, print data shape
    for key in results_dict.keys():
        print(f"Key: {key}, Shape: {results_dict[key].shape}")
    # EXAMPLE OUTPUT:
    # dict_keys(['g_embedding', 'node_recon', 'node_recon_all', 'labels'])
    # Key: g_embedding, Shape: (150, 256)
    # Key: node_recon, Shape: (150, 13461, 128)
    # Key: node_recon_all, Shape: (150, 13461, 1)
    # Key: labels, Shape: (150, 1)

    if not output_pkl.endswith(".pkl"):
        output_pkl += ".pkl"
    print(f"Saving results to {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(results_dict, f)

    # return output_pkl


def compute_feature_correlations(
    pickle_path: Optional[str] = None,
    node_recon: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compute feature correlation matrix.

    Args:
        pickle_path: Path to pickle file containing processed results
        node_recon: 3D array of shape (n_samples, n_features, dim)
        labels: 1D array of shape (n_samples,)
        device: Device to use for computation

    Returns:
        Correlation matrix of shape (n_features, n_features)
    """
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

    # Move results back to CPU and convert to numpy
    output = corr_matrix.cpu().numpy()
    print(f"\n🔥Correlation matrix shape: {output.shape}")
    np.save("correlation_matrix.npy", output)

    return output


# Example usage
# if __name__ == "__main__":
#     # Test with small synthetic data
#     node_recon = np.random.randn(100, 50, 16)  # 100 samples, 50 features, 16-dim
#     labels = np.random.randn(100)

#     # Should return (50, 50) matrix with values in [-1, 1]
#     corr_matrix = compute_feature_correlations(None, node_recon, labels)
#     print(f"\nCorrelation matrix shape: {corr_matrix.shape}")
#     print(f"Correlation range: ({corr_matrix.min():.2f}, {corr_matrix.max():.2f})")

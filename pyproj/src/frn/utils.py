import numpy as np
import pandas as pd
import random
import string
import os
import pickle
import gzip


def random_string(length: int = 7):
    letters = string.ascii_letters + string.digits
    result = ''.join(random.choice(letters) for _ in range(length))
    return result


def idx_convert(indices:list[int], len_one_hot_vec:int=10) -> list[int]:
    """
    Convert the indices to the corresponding indices in the one-hot vector.
    """
    converted_indices = [(i * len_one_hot_vec + nx) for nx in range(len_one_hot_vec) for i in indices]
    return sorted(converted_indices)


def one_hot_encode_snp_matrix(
        snp_matrix: np.ndarray,
        len_one_hot_vec: int = 10,
        genes_snps: list[list[int]] | None = None,
    ):
    """
    One-hot encode the SNP matrix.
    """
    if genes_snps is not None:
        num_genes = len(genes_snps)
        indices_snp = []
        for i_gene in range(num_genes):
            indices_snp.append(idx_convert(genes_snps[i_gene], len_one_hot_vec))
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_one_hot_vec + 1)[snp_vec][:, 1:].reshape(-1)
            snp_vec_genes = [snp_vec[indices_snp[i_gene]].astype(np.float32) for i_gene in range(num_genes)]
            snp_data.append(snp_vec_genes)
    else:
        snp_data = []
        for i_sample in range(snp_matrix.shape[0]):
            snp_vec = snp_matrix[i_sample].astype(int)
            snp_vec = np.eye(len_one_hot_vec + 1)[snp_vec][:, 1:].reshape(-1).astype(np.float32)
            snp_data.append(snp_vec)

    return snp_data


def read_pkl_gv(path_pkl: str) -> dict:
    """
    Read processed VCF data from a pickle file.
    """
    with gzip.open(path_pkl, 'rb') as file:
        # Initialize an empty list to hold all the deserialized vectors
        _vectors = []

        # While there is data in the file, load it
        while True:
            try:
                # Load the next pickled object from the file
                _data = pickle.load(file)
                # Append the loaded data to the list
                _vectors.append(_data)
            except EOFError:
                # An EOFError is raised when there is no more data to read
                break

    _sample_ids = _vectors[0]
    _snp_ids = _vectors[1]
    _block_ids = _vectors[2]
    _block2gtype = _vectors[3]
    _mat_vec = _vectors[4]
    _mat_shape = (len(_snp_ids), len(_sample_ids))

    # Reshape the matrix to the correct shape
    vcf_mat = np.reshape(_mat_vec, _mat_shape).transpose()

    return {
        'gt_mat': vcf_mat,
        'block2gtype': _block2gtype,
        'sample_ids': _sample_ids,
        'snp_ids': _snp_ids,
        'block_ids': _block_ids,
    }

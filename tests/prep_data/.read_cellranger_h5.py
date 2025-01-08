r"""
Read cellranger h5 file.
"""
import os
import sys
import scanpy
import polars as pl


if __name__ == "__main__":
    path_h5 = sys.argv[1]
    adata = scanpy.read_10x_h5(path_h5)
    adata.var_names_make_unique(join="_")
    print(f"\nadata.shape: {adata.shape}\n")
    print(f"n_obs: {adata.n_obs}\n")
    print(f"Feature types:\n{adata.var["feature_types"].unique()}\n")
    print(f"Features 0 to 4:\n{adata.var[:5]}\n")
    print(f"Features -1 to -5:\n{adata.var[-5:]}\n")

    path_atac_peaks_bed = os.path.join(os.path.dirname(path_h5), "atac_peaks.bed")
    peak_bed = pl.read_csv(path_atac_peaks_bed, separator='\t', has_header=False, comment_prefix='#')
    # print(f"Peaks bed shape: {peak_bed.shape}")
    peak_bed.columns = ['chrom', 'chromStart', 'chromEnd']
    n_feature_atac = peak_bed.shape[0]
    print(f"n_feature_atac: {n_feature_atac}")
    
    n_feature_rna = adata.n_vars - n_feature_atac
    print(f"n_feature_rna: {n_feature_rna}")

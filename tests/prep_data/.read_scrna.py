import scanpy
import pandas as pd
import polars as pl


if __name__ == '__main__':
    path_h5 = '/mnt/bank/sc_sn/GSE235510_RAW/GSM7504011.h5ad'
    adata = scanpy.read_h5ad(path_h5)
    print(f"adata.shape: {adata.shape}")
    print(f"num obs: {adata.obs.shape}")
    print(f"num var: {adata.var.shape}")
    print(adata.var_names)

    # print("\nFirst 5 var:\n", adata[:, :5].to_df())
    # print("\nLast 5 var:\n", adata[:, -5:].to_df())

    # Delete vars with zero counts
    adata = adata[:, adata.X.sum(axis=0) > 0]
    print(f"Matrix shape after delete vars with zero counts: {adata.shape}")
    # Delete obs with zero counts
    adata = adata[adata.X.sum(axis=1) > 0, :]
    print(f"Matrix shape after delete obs with zero counts: {adata.shape}")

    # print("\nFirst 5 var:\n", adata[:, :5].to_df())
    # print("\nLast 5 var:\n", adata[:, -5:].to_df())

    # Plot the histogram of the RNA values
    # Read peaks
    path_atac_peaks_bed = '/mnt/bank/sc_sn/GSE235510_RAW/GSM7504011_ATAC_Control2_atac_peaks.bed'
    peak_bed = pl.read_csv(path_atac_peaks_bed, separator='\t', has_header=False,
                        comment_prefix='#')
    # print(f"Peaks bed shape: {peak_bed.shape}")
    peak_bed.columns = ['chrom', 'chromStart', 'chromEnd']
    n_feature_atac = peak_bed.shape[0]
    
    n_feature_rna = adata.shape[1] - n_feature_atac
    # scanpy.pl.highest_expr_genes(adata[:, :n_feature_rna], n_top=20)

    # Find the maximum value in the matrix
    print(f"Max value of RNA: {adata[:, :n_feature_rna].X.max()}")
    print(f"Max value of ATAC: {adata[:, n_feature_rna:].X.max()}")

    # obtain nuclei with high qualities of both RNA-seq and ATAC-seq
    # (nCount_ATAC < 100,000, nCount_RNA < 25,000, nCount_ATAC > 1,000, nCount_RNA > 1,000, nucleosome_signal < 2 and TSS.enrichment > 1).
    adata = adata[adata[:, n_feature_rna:].X.sum(axis=1) > 1000, :]
    adata = adata[adata[:, n_feature_rna:].X.sum(axis=1) < 25000, :]
    adata = adata[adata[:, :n_feature_rna].X.sum(axis=1) > 1000, :]
    adata = adata[adata[:, :n_feature_rna].X.sum(axis=1) < 100000, :]
    # Filter blank features
    adata = adata[:, adata.X.sum(axis=0) > 0]
    
    print(f"Matrix shape after filter: {adata.shape}")

import os
import anndata
import polars as pl
# import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    data_dir = '/mnt/bank/sc_sn/GSE235510_RAW'
    gsm = 'GSM7504011'
    path_write = Path(os.path.join(data_dir, f'{gsm}.h5ad'))
    gsm_files = [f for f in os.listdir(data_dir) if f.startswith(gsm)]
    path_atac_mtx = Path(os.path.join(data_dir, [f for f in gsm_files if f.endswith('.mtx.gz')][0]))
    path_atac_barcodes = os.path.join(data_dir, [f for f in gsm_files if f.endswith('_barcodes.tsv.gz')][0])
    # path_atac_peak_bed =
    path_features = os.path.join(data_dir, 'GSM7504015_Control2_features.tsv')

    # Read matrix
    adata: anndata.AnnData = anndata.io.read_mtx(path_atac_mtx).T
    print(f"Matrix shape: {adata.shape}")
    # Delete vars with zero counts
    # adata = adata[:, adata.X.sum(axis=0) > 0]
    # print(f"Matrix shape after delete vars with zero counts: {adata.shape}")
    # Delete obs with zero counts
    # adata = adata[adata.X.sum(axis=1) > 0, :]
    # print(f"Matrix shape after delete obs with zero counts: {adata.shape}")

    # Read barcodes
    barcodes: list[str] = pl.read_csv(path_atac_barcodes, separator='\t', has_header=False).get_column('column_1').to_list()
    print(f"Barcodes num: {barcodes.__len__()}")
    
    # Read peaks
    # peak_bed = pl.read_csv(path_atac_peak_bed, separator='\t', has_header=False,
    #                     comment_prefix='#')
    # print(f"Peaks bed shape: {peak_bed.shape}")
    # peak_bed.columns = ['chrom', 'chromStart', 'chromEnd']

    # Read features
    features = pl.read_csv(path_features, separator='\t', has_header=False)
    print(f"Features shape: {features.shape}")
    var_names: list[str] = features.get_column('column_1').to_list()
    print(f"The first var_name: {var_names[0]}")
    print(f"The last var_name: {var_names[-1]}")

    # Set obs and var names
    adata.obs_names = barcodes
    # adata.var_names = peak_bed['peak_id']
    adata.var_names = var_names

    adata.write_h5ad(path_write, compression='gzip')

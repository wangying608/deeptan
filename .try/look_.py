import os

import polars as pl

if __name__ == "__main__":
    dir_ = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df"
    path_parq = "snRNA/ath_snrna_balanced_flower_seedling_rosette_split_full/nmic_g_mi01win001/split_42_0.parquet"
    _path = os.path.join(dir_, path_parq)
    _df = pl.read_parquet(_path)

    colnames = _df.columns
    print("Number of nodes: ", len(colnames))

    # gene_list = ["ECT5", "PRK1", "AT1G75945", "DAG2", "AP1", "AG", "ECT11", "PRK6", "SUS3", "SUS4", "SWEET3"]
    gene_list = ["AT3G13060", "AT5G35390", "AT5G10140"]

    # Search for the gene list in the column names
    for gene in gene_list:
        if gene in colnames:
            print(f"Gene {gene} found in column names.")
        else:
            print(f"Gene {gene} not found in column names.")

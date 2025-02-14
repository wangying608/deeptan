import polars as pl
from deeptan.utils.data import read_h5ad


if __name__ == "__main__":
    h5_path = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scRNA/SRP273996.h5ad"
    celltypes = read_h5ad(h5_path).obs["Celltype"]
    print(celltypes)
    celltypes_pl = pl.DataFrame({"bc": celltypes.index, "ct": celltypes.values})
    print(celltypes_pl)
    celltypes_onehot = celltypes_pl.to_dummies(columns=["ct"])

    # Add a new column "ct_unknown" which all values are 0 (type: u8)
    col_unk = pl.Series("ct_unknown", [0] * len(celltypes_onehot), dtype=pl.UInt8)
    celltypes_onehot = celltypes_onehot.with_columns(col_unk)

    # Save the one-hot encoded DataFrame
    celltypes_onehot.write_parquet(
        h5_path.replace(".h5ad", "_celltypes_onehot.parquet")
    )

    print(celltypes_onehot)

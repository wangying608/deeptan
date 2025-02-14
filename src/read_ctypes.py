import polars as pl


if __name__ == "__main__":
    # Load the CSV file into a Polars DataFrame
    path_csv = "/mnt/hdd2/homext/wuch/xn2p/data/raw_df/scMultiome/cell_type_annotations.csv"
    df = pl.read_csv(
        path_csv
    )
    df.columns = columns = ["bc", "ct"]
    print(df)

    # One-hot encode the 'ct' column
    df_one_hot = df_one_hot = df.to_dummies(columns=["ct"])

    # Add a new column "ct_unknown" which all values are 0 (type: u8)
    col_unk = pl.Series("ct_unknown", [0] * len(df_one_hot), dtype=pl.UInt8)
    df_one_hot = df_one_hot.with_columns(col_unk)

    # SORT
    df_one_hot = df_one_hot.sort("bc")

    print(df_one_hot)
    
    # Save the one-hot encoded DataFrame
    df_one_hot.write_parquet(path_csv.replace(".csv", "_one_hot.parquet"))

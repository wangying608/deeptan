import polars as pl
import sys


if __name__ == "__main__":
    path_result_parq = sys.argv[1]
    df1 = pl.read_parquet(path_result_parq)
    print(df1)

r"""
Get features UNION from the results of mi2graph.
"""
import os
import sys
import polars as pl


if __name__ == "__main__":
    # path_result_parq = sys.argv[1]
    # df1 = pl.read_parquet(path_result_parq)
    # print(df1)
    dir_results = sys.argv[1]
    files_parq = [f for f in os.listdir(dir_results) if f.endswith(".parquet")]
    # colnames = [pl.read_parquet_schema(os.path.join(dir_results, f)).keys() for f in files_parq]
    set_colnames = set()
    for f in files_parq:
        tmp_colnames = pl.read_parquet_schema(os.path.join(dir_results, f)).keys()
        tmp_1st_col = pl.read_parquet(os.path.join(dir_results, f), columns=[0])
        if len(tmp_colnames) < 2:
            continue
        print(f"{f}: num_var = {len(tmp_colnames)-1}, num_obs = {tmp_1st_col.shape[0]}")
        set_colnames.update(tmp_colnames)
    # Remove 'obs_names'
    set_colnames.remove('obs_names')
    # print(set_colnames.__len__())

    feature_names_list: list[str] = sorted(list(set_colnames))
    # Write to a parquet file
    df_var_names = pl.DataFrame({'var_names': feature_names_list})
    print("\n", df_var_names)
    # df_var_names.write_parquet(os.path.join(dir_results, "_union_var_names.parquet"))

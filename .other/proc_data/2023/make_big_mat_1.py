import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

# def discrSth(x, sth):
#     if x is np.nan and sth is np.nan:
#         return True
#     elif x is np.nan or sth is np.nan:
#         return False
#     else:
#         return x == sth

def delSthRowInDf(dfIn: pd.DataFrame, Sth, skipCols=None, maxPercSth=1.0):
    # print(dfIn.shape[0])
    # chunk_size = dfIn.shape[0] // nthreads
    # print(nthreads, chunk_size)

    if maxPercSth > 1.0:
        maxPercSth = 1.0
    if skipCols is None:
        skipCols = []
    
    rows2del = []
    # availCols = dfIn.shape[1] - len(skipCols)
    
    executor = ThreadPoolExecutor(max_workers=(os.cpu_count() or 1) * 5)
    # Use map to apply the function to each row
    results = list(executor.map(lambda xr: _process_row(dfIn, xr, Sth, skipCols, maxPercSth), range(dfIn.shape[0])))
        
    for should_delete, xr in results:
        if should_delete:
            rows2del.append(xr)
    
    dfOut = dfIn.drop(dfIn.index[rows2del])
    return dfOut

def _process_row(dfIn: pd.DataFrame, xr: int, Sth, skipCols: list[int], maxPercSth: float):
    row = dfIn.iloc[xr]
    row_vals = [element for index, element in enumerate(row) if index not in skipCols] if len(skipCols) > 0 else row
    
    # Check if all row elements are sth or missing
    # print(row_vals[0].isna())
    if np.all(np.logical_or(row_vals.isnull(), row_vals == Sth)):
        return True, xr
    
    # Check if percentage of sth or missing is greater than maxPercSth
    if maxPercSth < 1.0:
        # row_vals = row_vals.dropna()
        if len(row_vals) > 0:
            if (np.sum(row_vals.isnull()) / len(row_vals)) > maxPercSth:
                return True, xr
    
    return False, xr


# Example usage
# df = pd.DataFrame({
#     "A": [1, 2, None, 3, None, 4],
#     "B": [None, None, 2, None, None, 4],
#     "C": [3, 4, None, 5, None, 6],
# })

# print(df)

# Delete rows with all missing values
# df_filtered = delSthRowInDf(df, None)

# Delete rows with all values equal to 2 in "B"
# df_filtered = delSthRowInDf(df, 2, ["A"])

# Delete rows with more than 50% missing values in any column
# df_filtered = delSthRowInDf(df, None, maxPercSth=0.5)

# print(df_filtered)

xdir = "/home/wuch/prjs/XRN2P/data_tmp/test_data_big1"
file_omics = ["atac_matrix_1215.csv", "data_06_ibaq_phos_rnaseq.csv", "exp.matrix.csv", "m6a_matrix2.tsv", "prot_matrix.csv"]

df_atac = pd.read_csv(os.path.join(xdir, file_omics[0]), index_col=3).iloc[:,3:]
# df_m6A = pd.read_csv(os.path.join(xdir, file_omics[-2]), delimiter="\t", index_col=0, na_values=["NA"])

print(df_atac.shape)
# print(df_atac)
# print(df.iloc[0])

df_atac_filtered = delSthRowInDf(df_atac, None, maxPercSth=0.95)

print(df_atac_filtered.shape)

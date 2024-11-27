import polars as pl
import numpy as np


path_parq = "/mnt/bank/scPlantDB/ath/mic_g_init/SRP338044.h5ad.parquet"
mat_sc = pl.read_parquet(path_parq)
print(mat_sc)
mat_sc_np = mat_sc.drop(["obs_names"]).to_numpy()

# Compute coefficient of variation of each column
cv = np.std(mat_sc_np, axis=0) / np.mean(mat_sc_np, axis=0)
print(cv)

# Plot histogram of CV using seaborn
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(cv, bins=50)
plt.xlabel("Coefficient of Variation")
plt.ylabel("Frequency")
plt.show()

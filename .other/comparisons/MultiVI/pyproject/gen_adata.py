import numpy as np
import pandas as pd
import anndata as ad
from string import ascii_uppercase


# 设置观测值数量
n_obs = 1000

# 生成观察时间
obs = pd.DataFrame()
obs['time'] = np.random.choice(['day 1', 'day 2', 'day 4', 'day 8'], n_obs)

# 设置特征名
var_names = [i*letter for i in range(1, 10) for letter in ascii_uppercase]

# 特征数量
n_vars = len(var_names)

# 特征注释数据框
var = pd.DataFrame(index=var_names)

# 生成数据矩阵
X = np.arange(n_obs*n_vars).reshape(n_obs, n_vars)


# 初始化 AnnoData 对象
# AnnoData 对象默认使用数据类型为 `float32`, 可以更精确的存储数据
# 这里设置为整数，为了演示方便
adata = ad.AnnData(X, obs=obs, var=var, dtype='int32')
# 一般默认将变量或特征存储在数据框的行
# 查看数据
# print(adata)

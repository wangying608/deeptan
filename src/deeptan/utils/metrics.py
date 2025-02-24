import numpy as np
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import f1_score as F1
from sklearn.metrics import adjusted_mutual_info_score as AMI
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.metrics import homogeneity_score as HOM
from sklearn.metrics import normalized_mutual_info_score as NMI
# from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score as AUROC
from scipy.stats import pearsonr, entropy
from scib.metrics import kBET
from scib.metrics import silhouette as ASW


def calculate_mse(x_true, x_pred):
    mse = MSE(x_true, x_pred)
    return mse


def calculate_pcc(x_true, x_pred):
    pcc = []
    for i in range(x_true.shape[1]):
        if np.all(x_true[:, i] == x_true[:, i][0]) or np.all(
            x_pred[:, i] == x_pred[:, i][0]
        ):
            # If all elements in the column are the same, the correlation is undefined. Set it to 0.0.
            pcc.append(0.0)
        else:
            corr, _ = pearsonr(x_true[:, i], x_pred[:, i])
            pcc.append(corr)
    return np.mean(pcc)


def normalize_to_distribution(data):
    """
    将数据标准化为概率分布。
    如果数据是常量（所有值相同），返回均匀分布。
    """
    data_min = np.min(data)
    data_range = np.max(data) - data_min

    if data_range == 0:
        # 如果数据是常量，返回均匀分布
        return np.ones_like(data) / len(data)

    # 归一化到非负范围
    data_normalized = data - data_min

    # 确保数据和为 1
    data_sum = np.sum(data_normalized)
    if data_sum == 0:
        # 如果归一化后总和仍为零，返回均匀分布
        return np.ones_like(data) / len(data)

    return data_normalized / data_sum


def calculate_jsd(x_true, x_pred):
    """
    计算 Jensen-Shannon 散度 (JSD)。
    """
    jsd = np.zeros(x_true.shape[1])
    for i in range(x_true.shape[1]):
        true_col = x_true[:, i]
        pred_col = x_pred[:, i]

        # 标准化为概率分布
        true_dist = normalize_to_distribution(true_col)
        pred_dist = normalize_to_distribution(pred_col)

        # 计算平均分布
        m_dist = 0.5 * (true_dist + pred_dist)

        # 计算 JSD
        jsd[i] = 0.5 * (entropy(true_dist, m_dist) + entropy(pred_dist, m_dist))

    # 返回 JSD 的均值
    return np.mean(jsd)

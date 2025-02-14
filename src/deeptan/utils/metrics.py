import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.metrics import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import silhouette_score, roc_auc_score
from scipy.stats import pearsonr, entropy


def compute_mse(x_true, x_pred):
    # mse = np.array(
    #     [mean_squared_error(x_true[:, i], x_pred[:, i]) for i in range(x_true.shape[1])]
    # )
    mse = mean_squared_error(x_true, x_pred)
    return mse


def compute_pcc(x_true, x_pred):
    pcc = np.zeros(x_true.shape[1])
    for i in range(x_true.shape[1]):
        pcc[i], _ = pearsonr(x_true[:, i], x_pred[:, i])
    return np.mean(pcc)


def compute_rsquared(x_true, x_pred):
    r2 = r2_score(x_true, x_pred)
    return np.mean(r2)


def normalize_to_distribution(data):
    data = data - np.min(data)
    return data / np.sum(data)


def compute_jsd(x_true, x_pred):
    """
    Jensen-Shannon divergence
    """
    jsd = np.zeros(x_true.shape[1])
    for i in range(x_true.shape[1]):
        true_col = x_true[:, i]
        pred_col = x_pred[:, i]

        true_dist = normalize_to_distribution(true_col)
        pred_dist = normalize_to_distribution(pred_col)

        m_dist = 0.5 * (true_dist + pred_dist)

        jsd[i] = 0.5 * (entropy(true_dist, m_dist) + entropy(pred_dist, m_dist))
    return np.mean(jsd)

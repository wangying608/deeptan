import numpy as np
import time
from module_graph_init import mi_graph_init


if __name__ == '__main__':
    
    n_th = 20
    thresh_std = 0.05
    thresh_pcc = 0.95
    # matx = np.random.rand(807, 5003)# Time used for MI matrix initialization: 70638.85545778275 seconds
    matx = np.random.rand(807, 47)

    feature_names = [f"feature{i}" for i in range(matx.shape[1])]
    print("matx.shape:", matx.shape, "\n")

    mig_initializer = mi_graph_init(matx, feature_names, thresh_std, thresh_pcc, n_processes=n_th)

    time_start = time.time()
    mig_initializer.mi_matrix_init_with_feat_select()
    time_end = time.time()
    print("\nTime used for MI matrix initialization:", time_end - time_start, "seconds\n")

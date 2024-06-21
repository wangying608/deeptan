import numpy as np


if __name__ == '__main__':
    # read npz file
    path_npz = "test_mi_graph.npz"
    data = np.load(path_npz)

    # print the keys
    print("Keys in the npz file:", data.files)

    # access the data
    print(data['thre_pcc'])
    print(data['thre_sd'])
    print(data['thre_mi'])
    print(data['mi_values'])
    print(data['mi_values'].shape)
    print(data['feat_pairs'])
    print(data['feat_pairs'].shape)

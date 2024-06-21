import os
import numpy as np

'''
Generate a random matrix of size mxn with float elements in the range [0, 1]
Save the matrix to a file
'''
def rand_mat(m, n):
    mat = np.random.rand(m, n)
    np.save('test_rand_mat.npy', mat)


if __name__ == '__main__':
    # rand_mat(15000, 2000)
    rand_mat(203, 1001)
    os.system('mi2graph/target/release/mi2graph -i test_rand_mat.npy -o test_mi_graph.npz --thresd 0.001 --threpcc 0.85 --thremi 0.1')

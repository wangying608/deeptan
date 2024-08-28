# https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/sparse/matmul_cn.html#daimashili

import paddle

paddle.device.set_device('gpu')


# csr @ dense -> dense
crows = [0, 1, 2, 3]
cols = [1, 2, 0]
values = [1., 2., 3.]
csr = paddle.sparse.sparse_csr_tensor(crows, cols, values, [3, 3])
print(csr)

dense = paddle.ones([3, 2])
out = paddle.sparse.matmul(csr, dense)
print(out)

# coo @ dense -> dense
indices = [[0, 1, 2], [1, 2, 0]]
values = [1., 2., 3.]
coo = paddle.sparse.sparse_coo_tensor(indices, values, [3, 3])
print(coo)

dense = paddle.ones([3, 2])
out = paddle.sparse.matmul(coo, dense)
print(out)

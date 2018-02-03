import mxnet as mx
import numpy as np
import time

mx.random.seed(1)
np.random.seed(1)
num_rows = 1024*770
idx = np.random.randint(low=0, high=num_rows-1, size=9*1024)
sorted_idx = np.unique(idx)
print(sorted_idx.shape)

idx_nd = mx.nd.array(idx, dtype=np.int64)
data = mx.nd.ones((num_rows, 1024)).tostype('row_sparse')

mx.nd.waitall()
a = time.time()
for i in range(1):
    out = mx.nd.sparse.retain(data=data, indices=idx_nd)
mx.nd.waitall()
b = time.time()
print(b - a)
mx.nd.waitall()
c = time.time()
for i in range(1000):
    out = mx.nd.sparse.retain(data=data, indices=idx_nd)
mx.nd.waitall()
d = time.time()
print(d - c)

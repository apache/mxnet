import mxnet as mx

shape = (3, 4)
data_list = [7, 8, 9]
indices_list = [0, 2, 1]
indptr_list = [0, 2, 2, 3]
a = mx.nd.sparse.csr_matrix(data_list, indptr_list, indices_list, shape, ctx=mx.gpu())
b = mx.nd.sparse.slice(a,begin=0, end=2)
print b.asnumpy()

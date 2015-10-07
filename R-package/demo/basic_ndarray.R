require(mxnet)
require(methods)


x = as.array(c(1,2,3))
mat = mx.nd.array(x, mx.cpu(0))
mat = mat + 1.0
mat = mat + mat
oldmat = mat
mat = mx.nd.internal.plus.scalar(mat, 1, out=mat)
xx = mat$as.array()

# This will result in an error,  becase mat has been moved
oldmat + 1

print(xx)


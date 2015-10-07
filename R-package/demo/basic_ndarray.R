require(mxnet)
require(methods)
x = as.array(c(1,2,3))

mat = mx.nd.array(x, mx.cpu(0))
mat = mx.nd.internal.plus(mat, mat)
xx = mx.nd.internal.as.array(mat)
print(class(mat))
print(xx)


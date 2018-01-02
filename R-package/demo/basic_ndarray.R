require(mxnet)

x <- 1:3
mat <- mx.nd.array(x)

mat <- mat + 1.0
mat <- mat + mat
mat <- mat - 5
mat <- 10 / mat
mat <- 7 * mat
mat <- 1 - mat + (2 * mat) / (mat + 0.5)
as.array(mat)

x <- as.array(matrix(1:4, 2, 2))

mx.ctx.default(mx.cpu(1))
print(mx.ctx.default())
print(is.mx.context(mx.cpu()))
mat <- mx.nd.array(x)
mat <- (mat * 3 + 5) / 10
as.array(mat)

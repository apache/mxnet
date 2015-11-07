require(mxnet)
require(methods)


shape = c(1, 1)
lr = 0.01
x = mx.nd.ones(shape)
y = mx.nd.zeros(shape)
print(x)
n = 1000


tic = proc.time()
for (i in 1 : n) {
  y = y + x *lr
}
toc = proc.time() - tic
as.array(y)
print(toc)

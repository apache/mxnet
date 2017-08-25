require(mxnet)
# TODO(KK, tong) think about setter getter interface(which breaks immutability, or current set and move interface.
# We need to make a choice between
# exec_old = exec
# exec$arg.arrays = some.array, this changes exec_old$arg.arrays as well, user won't aware
# V.S.
# exec_old = exec
# exec = mx.exec.set.arg.arrays(exec, some.array)
# exec_old is moved, user get an error when use exec_old

A = mx.symbol.Variable('A')
B = mx.symbol.Variable('B')
C = A + B
a = mx.nd.zeros(c(2), mx.cpu())
b = mx.nd.array(as.array(c(1, 2)), mx.cpu())

exec = mxnet:::mx.symbol.bind(
  symbol=C,
  ctx=mx.cpu(),
  arg.arrays = list(A=a, B=b),
  aux.arrays = list(),
  grad.reqs = list("null", "null"))

# calculate outputs
mx.exec.forward(exec)
out = as.array(exec$outputs[[1]])
print(out)

mx.exec.update.arg.arrays(exec, list(A=b, B=b))
mx.exec.forward(exec)

out = as.array(exec$outputs[[1]])
print(out)


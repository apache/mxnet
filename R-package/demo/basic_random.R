require(mxnet)

mx.set.seed(10)

print(mx.runif(c(2,2), -10, 10))

# Test initialization module for neural nets.
uinit <- mx.init.uniform(0.1)
print(uinit("fc1_weight", c(2, 2), mx.cpu()))
print(uinit("fc1_gamma", c(2, 2), mx.cpu()))

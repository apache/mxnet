import mxnet as mx

a = mx.zeros_shared((3,4))
b = mx.zeros_shared((3,4))
a.numpy[:] = 10
b.numpy[:] = 11
print(a.numpy)
c = b + a

print(c.to_numpy())


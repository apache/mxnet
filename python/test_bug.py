import mxnet as mx
inputs = mx.np.array([1,1,1])
fake_data = mx.np.array([1,1,0])
print(type(inputs))
print(type(fake_data))
#print(1 - mx.np.equal(fake_data, inputs))


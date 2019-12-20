import mxnet as mx
from mxnet.gluon.probability import Normal
from mxnet import np, npx

loc = np.zeros((2,2))
scale = np.ones((2,2))
dist = Normal(loc, scale)

print(dist.sample())
print(dist.sample_n((2,2)).shape)
print(dist.mean)
print(dist.variance)
print(dist.log_prob(np.random.uniform(size=(2,2))))

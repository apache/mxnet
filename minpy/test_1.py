import mxnet
import numpy as np

a = mxnet.nd.ones(2)
b = mxnet.nd.ones(2)
print((a + b).asnumpy())

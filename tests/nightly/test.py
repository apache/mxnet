import mxnet as mx
grad = mx.nd.array([-6.3, -2.1, 3.4, 1.2, 10.5, 5.1, -3.2, 2.0, -8.9, 0])
residual = mx.nd.array([-3.1, 1.2, -1.3, 5.4, -2.1, 2.9, 3.0, -7.0, -2.9, -100.3])
neg_threshold = mx.nd.array([-4.0])
pos_threshold = mx.nd.array([4.0])
out = mx.contrib.nd.create_2bit(grad)
import pdb
#pdb.set_trace()
mx.contrib.ndarray.quantize_2bit(grad, residual, neg_threshold, pos_threshold, out)

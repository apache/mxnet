from __future__ import print_function
import numpy as np
import mxnet as mx
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
import unittest
import timeit

shape = [(268435456)] #(25,),(16,),(1121),(14400),(144000),

# for shape in orig_shape:
grad = mx.nd.random_uniform(-0.9,0.9, shape=shape, ctx=default_context())
residual = mx.nd.random_uniform(-0.6,0.6, shape=shape, ctx=default_context())
res = mx.nd.array(residual)
mx.nd.waitall()

def run():
    compr = mx.contrib.nd.create_2bit(grad)
    decompr = mx.nd.array(grad.shape)
    mx.contrib.ndarray.quantize_2bit(grad, res, compr, -0.5, 0.5)
    mx.contrib.ndarray.dequantize_2bit(compr, decompr)
    mx.nd.waitall()
d = timeit.repeat(run, repeat=10, number=1)
print(d)
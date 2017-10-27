from __future__ import print_function
import numpy as np
import mxnet as mx
import random
import itertools
from numpy.testing import assert_allclose, assert_array_equal
from mxnet.test_utils import *
import unittest
import timeit
mx.profiler.profiler_set_config(mode='all',filename='profiler.json')
#shape = [(268435456)] #(25,),(16,),(1121),(14400),(144000),
grad = mx.nd.load('example/image-classification/quant_data')[0]
print(grad.shape)
#grad = mx.nd.random_uniform(-2,2,shape=shape, ctx=default_context())
residual = mx.nd.random_uniform(-0.4,0.4,shape=grad.shape, ctx=default_context())
res = mx.nd.array(residual)
compressed = mx.contrib.nd.create_2bit(grad)
def run():
    compr = mx.nd.zeros(compressed.shape)
    decompr = mx.nd.zeros(grad.shape)
    mx.contrib.ndarray.quantize_2bit(grad, res, compr, -0.5, 0.5)
    mx.contrib.ndarray.dequantize_2bit(compr, decompr)
    mx.nd.waitall()
mx.profiler.profiler_set_state('run')
d = timeit.repeat(run, repeat=10, number=1)
#mx.profiler.profiler_set_state('stop')
print(d)



def run_mshadow():
    compr = mx.nd.zeros(compressed.shape)
    decompr = mx.nd.zeros(grad.shape)
    mx.contrib.ndarray.quantize_mshadow_2bit(grad, res, compr, -0.5, 0.5)
    mx.contrib.ndarray.dequantize_mshadow_2bit(compr, decompr)
    mx.nd.waitall()
d2 = timeit.repeat(run_mshadow, repeat=25, number=1)
mx.profiler.profiler_set_state('stop')
# print( d2)

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import random
import mxnet as mx
import numpy as np
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import with_seed

def check_fused_symbol(sym, **kwargs):
    inputs = sym.list_inputs()
    shapes = {inp : kwargs[inp].shape for inp in inputs}
    test_sym = mx.sym.Group([mx.sym.identity(s) for s in sym])
    rtol = {'float16' : 1e-2,
            'float32' : 1.5e-6,
            'float64' : 1.5e-6,
            }
    atol = {'float16' : 1e-3,
            'float32' : 1e-7,
            'float64' : 1e-7,
            }
    for dtype in ['float16', 'float32', 'float64']:
        data = {inp : kwargs[inp].astype(dtype) for inp in inputs}
        for grad_req in ['write', 'add']:
            type_dict = {inp : dtype for inp in inputs}
            os.environ["MXNET_USE_FUSION"] = "0"
            orig_exec = test_sym.simple_bind(ctx=mx.gpu(0), grad_req=grad_req, type_dict=type_dict, **shapes)
            os.environ["MXNET_USE_FUSION"] = "1"
            fused_exec = test_sym.simple_bind(ctx=mx.gpu(0), grad_req=grad_req, type_dict=type_dict, **shapes)
            fwd_orig = orig_exec.forward(is_train=True, **data)
            out_grads = [mx.nd.ones_like(arr) for arr in fwd_orig]
            orig_exec.backward(out_grads=out_grads)
            fwd_fused = fused_exec.forward(is_train=True, **data)
            fused_exec.backward(out_grads=out_grads)
            for orig, fused in zip(fwd_orig, fwd_fused):
                np.testing.assert_allclose(orig.asnumpy(), fused.asnumpy(), rtol=rtol[dtype], atol=atol[dtype])
            for orig, fused in zip(orig_exec.grad_arrays, fused_exec.grad_arrays):
                if orig is None and fused is None:
                    continue
                assert orig is not None
                assert fused is not None
                np.testing.assert_allclose(orig.asnumpy(), fused.asnumpy(), rtol=rtol[dtype], atol=atol[dtype])

def check_unary_ops():
    unary_ops = [
            'relu',
            'sigmoid',
            'softsign',
            'exp',
            'expm1',
            'log',
            'log10',
            'log2',
            'log1p',
            'degrees',
            'radians',
            'sin',
            'cos',
            'tan',
            'arcsin',
            'arccos',
            'arctan',
            'sinh',
            'cosh',
            'tanh',
            'arcsinh',
            'arctanh',
            'sqrt',
            'rsqrt',
            'cbrt',
            'rcbrt',
            'square',
            'squeeze',
            'zeros_like',
            'ones_like',
            'flatten',
            'round',
            'rint',
            'fix',
            'floor',
            'ceil',
            'trunc',
            'sign',
            'reciprocal',
            'abs',
            'gamma',
            'gammaln',
            'erf',
            'erfinv',
            'negative',
            ]
    arr = mx.random.uniform(shape=rand_shape_2d())
    a = mx.sym.Variable('a')
    for op_name in unary_ops:
        print("Checking fusion of " + op_name)
        op = getattr(mx.sym, op_name)
        sym = op(a)
        check_fused_symbol(sym, a=arr)

    # unary ops requiring special treatment

    # arccosh needs input to be >= 1
    arr2 = arr + 1
    check_fused_symbol(mx.sym.arccosh(a), a=arr2)

    # Activation requires act_type attribute
    for act_type in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
        check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=arr)

    # Cast requires dtype
    for dtype in ['float16', 'float32', 'float64', 'int32']:
        check_fused_symbol(mx.sym.Cast(a, dtype=dtype), a=arr)

    # reshape requires shape
    check_fused_symbol(mx.sym.reshape(a, shape=(-1,)), a=arr)

    # expand_dims requires axis
    check_fused_symbol(mx.sym.expand_dims(a, axis=1), a=arr)

    # clip requires a_min, a_max
    check_fused_symbol(mx.sym.clip(a, a_min=0.3, a_max=0.7), a=arr)

    # smooth_l1 requires a scalar
    check_fused_symbol(mx.sym.smooth_l1(a, scalar=0.3), a=arr)

def check_binary_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    check_fused_symbol(a+b, a=arr1, b=arr2)
    check_fused_symbol(a+3, a=arr1)
    check_fused_symbol(a-b, a=arr1, b=arr2)
    check_fused_symbol(a-3, a=arr1)
    check_fused_symbol(3-a, a=arr1)
    check_fused_symbol(a*b, a=arr1, b=arr2)
    check_fused_symbol(a*3, a=arr1)
    check_fused_symbol(a/b, a=arr1, b=arr2)
    check_fused_symbol(a/3, a=arr1)
    check_fused_symbol(3/a, a=arr1)
    check_fused_symbol(a**b, a=arr1, b=arr2)
    check_fused_symbol(a**3, a=arr1)
    check_fused_symbol(mx.sym.pow(3,a), a=arr1)
    check_fused_symbol(mx.sym.maximum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.minimum(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,b), a=arr1, b=arr2)
    check_fused_symbol(mx.sym.hypot(a,3), a=arr1)

def check_other_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    c = mx.sym.Variable('c')
    shape = rand_shape_2d()
    shape = (5,) + shape
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)
    arr3 = mx.random.uniform(shape=shape)

    check_fused_symbol(mx.sym.add_n(a,b,c), a=arr1, b=arr2, c=arr3)

    check_fused_symbol(mx.sym.slice_axis(a, axis=0, begin=1, end=4), a=arr1)

    begin = (random.randint(0, shape[0]-1),
             random.randint(0, shape[1]-1),
             random.randint(0, shape[2]-1))
    end = (random.randint(begin[0]+1, shape[0]),
           random.randint(begin[1]+1, shape[1]),
           random.randint(begin[2]+1, shape[2]))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

@with_seed()
def test_fusion():
    check_unary_ops()
    check_binary_ops()
    check_other_ops()

if __name__ == '__main__':
    import nose
    nose.runmodule()

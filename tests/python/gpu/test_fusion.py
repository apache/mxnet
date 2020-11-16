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

import sys
import os
import random
import itertools
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
from mxnet.test_utils import *

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))

def check_fused_symbol(sym, **kwargs):
    inputs = sym.list_inputs()
    shapes = {inp : kwargs[inp].shape for inp in inputs}
    ctx = kwargs.get('ctx', mx.gpu(0))
    # Double identity so that there is always something to fuse
    test_sym = mx.sym.Group([mx.sym.identity(mx.sym.identity(s)) for s in sym])
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
            with environment('MXNET_USE_FUSION', '0'):
                orig_exec = test_sym._simple_bind(ctx=ctx, grad_req=grad_req, type_dict=type_dict, **shapes)
            with environment('MXNET_USE_FUSION', '1'):
                fused_exec = test_sym._simple_bind(ctx=ctx, grad_req=grad_req, type_dict=type_dict, **shapes)
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
            'negative',
            'logical_not',
            ]

    def announce_check(op_name):
        print("Checking fusion of " + op_name)

    arr = mx.random.uniform(shape=rand_shape_2d())
    a = mx.sym.Variable('a')
    for op_name in unary_ops:
        announce_check(op_name)
        op = getattr(mx.sym, op_name)
        sym = op(a)
        check_fused_symbol(sym, a=arr)

    # unary ops requiring special treatment

    # arccosh needs input to be >= 1
    arr2 = arr + 1
    announce_check('arccosh')
    check_fused_symbol(mx.sym.arccosh(a), a=arr2)

    # erfinv needs -1 < input < 1, but we avoid the limits of this range where the slope nears +inf.
    arr2 = (arr - 0.5) * 1.99
    announce_check('erfinv')
    check_fused_symbol(mx.sym.erfinv(a), a=arr2)

    # Activation requires act_type attribute
    for act_type in ['relu', 'sigmoid', 'tanh', 'softrelu', 'softsign']:
        announce_check("Activation(act_type='{}')".format(act_type))
        check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=arr)
        if act_type == 'softrelu':
            # Check that softrelu implementation doesn't overflow on large inputs
            check_fused_symbol(mx.sym.Activation(a, act_type=act_type), a=1000 * arr)

    # Cast requires dtype
    for dtype in ['float16', 'float32', 'float64', 'int32']:
        announce_check("Cast(dtype='{}')".format(dtype))
        check_fused_symbol(mx.sym.Cast(a, dtype=dtype), a=arr)

    # reshape requires shape
    announce_check('reshape')
    check_fused_symbol(mx.sym.reshape(a, shape=(-1,)), a=arr)

    # expand_dims requires axis
    announce_check('expand_dims')
    check_fused_symbol(mx.sym.expand_dims(a, axis=1), a=arr)

    # clip requires a_min, a_max
    announce_check('clip')
    check_fused_symbol(mx.sym.clip(a, a_min=0.3, a_max=0.7), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=-np.inf, a_max=0.7), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=-np.inf, a_max=np.inf), a=arr)
    check_fused_symbol(mx.sym.clip(a, a_min=0, a_max=np.nan), a=arr)

    # smooth_l1 requires a scalar
    announce_check('smooth_l1')
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
    check_fused_symbol(a/(b+1), a=arr1, b=arr2)
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
    shape = list((5,) + shape)
    # Make sure there is at least 2 elements for the test with negative indices
    shape[1] += 1
    shape[2] += 1
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)
    arr3 = mx.random.uniform(shape=shape)

    check_fused_symbol(mx.sym.add_n(a,b,c), a=arr1, b=arr2, c=arr3)

    check_fused_symbol(mx.sym.slice_axis(a, axis=0, begin=1, end=4), a=arr1)

    # Testing handling of negative axis
    check_fused_symbol(mx.sym.slice_axis(a, axis=-3, begin=1, end=4), a=arr1)

    begin = (random.randint(0, shape[0]-1),
             random.randint(0, shape[1]-1),
             random.randint(0, shape[2]-1))
    end = (random.randint(begin[0]+1, shape[0]),
           random.randint(begin[1]+1, shape[1]),
           random.randint(begin[2]+1, shape[2]))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    begin = (random.randint(-shape[0], -2),
             random.randint(-shape[1], -2),
             random.randint(-shape[2], -2))
    end = (random.randint(begin[0]+1, -1),
           random.randint(begin[1]+1, -1),
           random.randint(begin[2]+1, -1))
    check_fused_symbol(mx.sym.slice(a, begin=begin, end=end), a=arr1)

    arr1 = mx.random.uniform(shape=(2,3,4,5))
    arr2 = mx.random.uniform(shape=(1,2,3))
    check_fused_symbol(mx.sym.slice_like(a,b, axes=[-2, 0]), a=arr1, b=arr2)

    arr1 = mx.random.uniform(shape=(1,1,2,3))
    arr2 = mx.random.uniform(shape=(2,2,2,3))
    check_fused_symbol(mx.sym.broadcast_like(a, b, lhs_axes=[0], rhs_axes=[0]), a=arr1, b=arr2)


def check_leakyrelu_ops():
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    # Testing gelu
    print("Checking fusion of LeakyReLU:gelu")
    check_fused_symbol(mx.sym.LeakyReLU(a+b, act_type='gelu'), a=arr1, b=arr2)


def test_fusion():
    check_unary_ops()
    check_binary_ops()
    check_other_ops()
    check_leakyrelu_ops()

def test_fusion_compiler_cache():
    # Stresses the internal cache of CUfunctions by creating the same kernel multiple times and
    # on multiple GPUs if available.
    a = mx.sym.Variable('a')
    b = mx.sym.Variable('b')
    shape = rand_shape_2d()
    arr1 = mx.random.uniform(shape=shape)
    arr2 = mx.random.uniform(shape=shape)

    # Invoke the same model twice, second time will exercise compile cache
    check_fused_symbol(a+b, ctx=mx.gpu(0), a=arr1, b=arr2)
    check_fused_symbol(a+b, ctx=mx.gpu(0), a=arr1, b=arr2)

    # On multi-GPU systems, invoke the same model on other GPUs
    num_gpus = mx.context.num_gpus()
    if num_gpus > 1:
        check_fused_symbol(a+b, ctx=mx.gpu(1), a=arr1, b=arr2)

@use_np
def test_fusion_boolean_inputs():
    from mxnet.gluon import HybridBlock

    class Foo(HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()

        def hybrid_forward(self, F, valid_length):
            mask = valid_length.astype(np.float32)
            mask2 = valid_length.astype(np.float32)
            mask = mask * F.np.expand_dims(mask2, axis=-1)
            return mask

    foo = Foo()
    foo.hybridize(static_alloc=True)
    out = foo(mx.np.ones((10,), ctx=mx.gpu(), dtype=np.bool))
    mx.npx.waitall()

def test_fusion_different_dimensions():
    from mxnet.gluon import HybridBlock

    class Foo(HybridBlock):
        def __init__(self):
            super(Foo, self).__init__()

        def hybrid_forward(self, F, x):
            mask2 = x.astype(np.float32)
            mask = F.expand_dims(mask2, axis=-1)
            return mask

    foo = Foo()
    foo.hybridize(static_alloc=True)
    # Pass 1-D data
    out = foo(mx.nd.ones((10,), ctx=mx.gpu()))
    assert np.all(out.asnumpy() == np.ones((10,1)))
    assert out.shape == (10,1)
    # Pass 2-D data
    out = foo(mx.nd.ones((10,10), ctx=mx.gpu()))
    assert np.all(out.asnumpy() == np.ones((10,10)))
    assert out.shape == (10,10,1)

def test_input_reorder():
    class Block(gluon.HybridBlock):
        def __init__(self, **kwargs):
            super(Block, self).__init__(**kwargs)

        def hybrid_forward(self, F, x, y, z):
            s = x * 2
            s2 = s + z
            s = F.broadcast_add(s, y * y)
            return F.dot(s, s2)

    for static_alloc in (False, True):
        arg_shapes = [(10, 10), (10, 1), (10, 10)]
        arg_data = [mx.random.uniform(shape=s) for s in arg_shapes]

        arrays = {}
        for use_fusion in ('0', '1'):
            with environment('MXNET_USE_FUSION', use_fusion):
                arrays[use_fusion] = {}
                n = Block()
                n.hybridize(static_alloc=static_alloc)
                args = [arg.copyto(mx.gpu()) for arg in arg_data]
                for arg in args:
                    arg.attach_grad()
                with autograd.record():
                    r = n(*args)
                arrays[use_fusion]['result'] = r
                r.backward()
                for i, arg in enumerate(args):
                    arrays[use_fusion][i] = arg.grad
        for key in ['result'] + list(range(len(arg_data))):
            assert_allclose(arrays['0'][key].asnumpy(), arrays['1'][key].asnumpy())

def test_fusion_cycle():
    class Test(gluon.nn.HybridBlock):
        def __init__(self, **kwargs):
            super(Test, self).__init__(**kwargs)

        def hybrid_forward(self, F, x, y):
            x = F.relu(x)
            y = F.relu(y)
            z1 = F.expand_dims(F.sum_axis(x, axis=1), axis=1)
            z2 = F.expand_dims(F.sum_axis(y, axis=1), axis=1)
            return x + z2, y + z1

    t = Test()
    a = mx.nd.zeros(shape=(10,1), ctx=mx.gpu())
    b = mx.nd.zeros(shape=(10,1), ctx=mx.gpu())
    t.hybridize(static_alloc=True, static_shape=True)
    out = t(a, b)
    mx.nd.waitall()

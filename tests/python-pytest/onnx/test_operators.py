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

import mxnet as mx
from mxnet.gluon import HybridBlock, nn
import numpy as np
import onnxruntime as rt
from mxnet.test_utils import assert_almost_equal
import pytest
import tempfile

def op_export_test(op_name, Model, inputs, tmp_path):
    def export_to_onnx(model, op_name, inputs):
        model_path = '{}/{}'.format(tmp_path, op_name)
        model.export(model_path, epoch=0)
        sym_file = '{}-symbol.json'.format(model_path)
        params_file = '{}-0000.params'.format(model_path)
        dtype = inputs[0].dtype
        onnx_file = '{}/{}.onnx'.format(tmp_path, op_name)
        mx.contrib.onnx.export_model(sym_file, params_file, [i.shape for i in inputs],
                                     dtype, onnx_file)
        return onnx_file
    def onnx_rt(onnx_file, inputs):
        sess = rt.InferenceSession(onnx_file)
        input_dict = dict((sess.get_inputs()[i].name, inputs[i].asnumpy()) for i in range(len(inputs)))
        pred = sess.run(None, input_dict)[0]
        return pred

    # create a new model 
    model = Model()
    model.initialize(ctx=mx.cpu(0))
    model.hybridize()
    pred_nat = model(*inputs)
    onnx_file = export_to_onnx(model, op_name, inputs)
    pred_onx = onnx_rt(onnx_file, inputs)
    assert_almost_equal(pred_nat, pred_onx)


def test_onnx_export_abs():
    with tempfile.TemporaryDirectory() as tmp_path:
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x):
                out = F.abs(x)
                return out
        x = mx.nd.array([[-2, -1], [0, 99]], dtype='float32')
        op_export_test('abs', Model, [x], tmp_path)

def test_onnx_export_slice():
    with tempfile.TemporaryDirectory() as tmp_path:
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x):
                out = F.slice(x, begin=(0,1), end=(2,4))
                return out
        x = mx.nd.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]], dtype='float32')
        op_export_test('slice', Model, [x], tmp_path)

def test_onnx_export_stack():
    with tempfile.TemporaryDirectory() as tmp_path:
        dtype = 'float32'
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x, y):
                out = F.stack(x, y)
                return out
        x = mx.nd.array([1, 2], dtype=dtype)
        y = mx.nd.array([3, 4], dtype=dtype)
        op_export_test('stack', Model, [x, y], tmp_path)

def test_onnx_export_zeros_like():
    with tempfile.TemporaryDirectory() as tmp_path:
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x):
                out = F.zeros_like(x)
                return out
        x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype='float32')
        op_export_test('zeros_like', Model, [x], tmp_path)

@pytest.mark.parametrize("dtype", ["float32", "double"])
def test_onnx_export_arange_like(dtype):
    with tempfile.TemporaryDirectory() as tmp_path:
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x):
                out = F.contrib.arange_like(x)
                return out
        x = mx.nd.array([[-2,-1,0],[0,50,99],[4,5,6],[7,8,9]], dtype=dtype)
        op_export_test('arange_like', Model, [x], tmp_path)

def test_onnx_export_layernorm():
    with tempfile.TemporaryDirectory() as tmp_path:
        dtype = 'float32'
        class Model(HybridBlock):
            def __init__(self, **kwargs):
                super(Model, self).__init__(**kwargs)
            def hybrid_forward(self, F, x, gamma, beta):
                out = F.LayerNorm(x, gamma, beta, axis=1)
                return out
        x = mx.nd.array([[1,3],[2,4]], dtype=dtype)
        gamma = mx.random.uniform(0, 1, x[0].shape).astype(dtype)
        beta = mx.random.uniform(0, 1, x[0].shape).astype(dtype)
        op_export_test('LayerNorm', Model, [x, gamma, beta], tmp_path)


if __name__ == '__main__':
    test_onnx_export_abs()
    test_onnx_export_slice()
    test_onnx_export_stack()
    test_onnx_export_zeros_like()
    test_onnx_export_arange_like('float32')
    test_onnx_export_arange_like('double')
    test_onnx_export_layernorm()


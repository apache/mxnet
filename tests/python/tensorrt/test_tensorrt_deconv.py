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
from mxnet.test_utils import assert_almost_equal

def get_params():
    arg_params = {}
    aux_params = {}
    arg_params["trt_bn_test_conv_weight"] = mx.nd.ones((1, 1, 3, 3))
    arg_params["trt_bn_test_deconv_weight"] = mx.nd.ones((1, 1, 3, 3))
    return arg_params, aux_params

def get_symbol():
    data = mx.sym.Variable("data")
    conv = mx.sym.Convolution(data=data, kernel=(3,3), no_bias=True, num_filter=1, num_group=1,
                              name="trt_bn_test_conv")
    deconv = mx.sym.Deconvolution(data=conv, kernel=(3, 3), no_bias=True, num_filter=1,
                                  num_group=1, name="trt_bn_test_deconv")
    return deconv

def test_deconvolution_produce_same_output_as_tensorrt():
    arg_params, aux_params = get_params()
    arg_params_trt, aux_params_trt = get_params()

    sym = get_symbol()
    sym_trt = get_symbol().get_backend_symbol("TensorRT")

    mx.contrib.tensorrt.init_tensorrt_params(sym_trt, arg_params_trt, aux_params_trt)

    executor = sym.simple_bind(ctx=mx.gpu(), data=(1, 1, 3, 3), grad_req='null', force_rebind=True)
    executor.copy_params_from(arg_params, aux_params)

    executor_trt = sym_trt.simple_bind(ctx=mx.gpu(), data=(1, 1, 3, 3), grad_req='null',
                                  force_rebind=True)
    executor_trt.copy_params_from(arg_params_trt, aux_params_trt)

    input_data = mx.nd.random.uniform(low=0, high=1, shape=(1, 1, 3, 3))

    y = executor.forward(is_train=False, data=input_data)
    y_trt = executor_trt.forward(is_train=False, data=input_data)

    print(y[0].asnumpy())
    print(y_trt[0].asnumpy())
    assert_almost_equal(y[0].asnumpy(), y_trt[0].asnumpy(), 1e-4, 1e-4)

if __name__ == '__main__':
    import nose
    nose.runmodule()

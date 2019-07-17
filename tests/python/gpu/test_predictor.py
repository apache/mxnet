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

from __future__ import print_function
import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append(os.path.join(curr_path, "../../../amalgamation/python/"))
from mxnet_predict import Predictor, load_ndarray_file

import ctypes
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from mxnet.ndarray import NDArray
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal, download_model
from mxnet.contrib.amp import amp
from mxnet.base import NDArrayHandle, py_str
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
from common import setup_module, with_seed, teardown

@with_seed()
def test_predictor_with_dtype():
    prefix = 'test_predictor_simple_dense'
    symbol_file = "%s-symbol.json" % prefix
    param_file = "%s-0000.params" % prefix

    input1 = np.random.uniform(size=(1, 3))
    input1 = input1.astype(np.float16)

    block = mx.gluon.nn.HybridSequential()
    block.add(mx.gluon.nn.Dense(7))
    block.add(mx.gluon.nn.Dense(3))
    block.cast(np.float16)
    block.hybridize()
    block.initialize(ctx=mx.gpu(0))
    tmp = mx.nd.array(input1, dtype=np.float16, ctx=mx.gpu(0))
    out1 = block.forward(tmp)
    block.export(prefix)

    predictor = Predictor(open(symbol_file, "r").read(),
                          open(param_file, "rb").read(),
                          {"data": input1.shape},
                          dev_type="gpu",
                          dev_id=0,
                          type_dict={"data": input1.dtype})
    predictor.forward(data=input1)
    predictor_out1 = predictor.get_output(0)

    assert_almost_equal(out1.asnumpy(), predictor_out1, rtol=1e-5, atol=1e-6)

def compare_module_cpredict(result_sym, result_arg_params, result_aux_params, monitor_callback=False):
    # Dummmy inputs
    input1 = np.ones((1, 3, 224, 224))
    input1 = input1.astype(np.float32)
    nd_dict = {}
    def pred_mon_callback(name, arr):
        nd_dict[name] = arr
    mod = mx.mod.Module(result_sym, data_names=["data"], label_names=["softmax_label"], context=mx.gpu())
    mod.bind(data_shapes=[['data', (1, 3, 224, 224)]], label_shapes=[['softmax_label', (1,)]], for_training=False)
    mod.set_params(result_arg_params, result_aux_params)
    mod.forward(mx.io.DataBatch(data=[mx.nd.array(input1, ctx=mx.gpu())],
                                label=[mx.nd.ones((1,), ctx=mx.gpu())]))
    prefix = "test_predictor_amp"
    mod.save_checkpoint(prefix, 0, remove_amp_cast=False)
    sym_file = "{}-symbol.json".format(prefix)
    params_file = "{}-0000.params".format(prefix)
    predictor = Predictor(open(sym_file, "r").read(),
                          open(params_file, "rb").read(),
                          {'data': (1, 3, 224, 224),
                           'softmax_label': (1,)},
                          dev_type="gpu",
                          dev_id=0)
    if monitor_callback:
        predictor.set_monitor_callback(pred_mon_callback, monitor_all=True)
    predictor.forward(data=input1, softmax_label=mx.nd.ones((1,)).asnumpy())
    predictor_out1 = predictor.get_output(0)
    if monitor_callback:
        assert len(nd_dict) > 0, "Callback not called"
    assert_almost_equal(mod.get_outputs()[0].asnumpy(), predictor_out1, atol=1e-1, rtol=1e-1)


@with_seed()
def test_predictor_amp():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, 'model')
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    prefix, epoch = download_model("imagenet1k-resnet-18", dst_dir=model_path)

    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)


    # Convert model to mixed precision model, params in FP32
    result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                         arg_params,
                                                                         aux_params,
                                                                         target_dtype="float16",
                                                                         target_dtype_ops=["Convolution"])
    compare_module_cpredict(result_sym, result_arg_params, result_aux_params)

    # Convert model to mixed precision model, params in FP16
    result_sym, result_arg_params, result_aux_params = amp.convert_model(sym,
                                                                         arg_params,
                                                                         aux_params,
                                                                         target_dtype="float16",
                                                                         target_dtype_ops=["Convolution"],
                                                                         cast_optional_params=True)
    compare_module_cpredict(result_sym, result_arg_params, result_aux_params, monitor_callback=True)


if __name__ == '__main__':
    import nose
    nose.runmodule()

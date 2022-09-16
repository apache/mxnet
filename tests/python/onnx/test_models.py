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
import numpy as np
import onnxruntime
import pytest
import shutil
from mxnet import gluon
from mxnet.test_utils import assert_almost_equal

@pytest.mark.skip(reason='Gluon no long support v1.x models since https://github.com/apache/incubator-mxnet/pull/20262')
def test_resnet50_v2(tmp_path):
    try:
        ctx = mx.cpu()
        model = gluon.model_zoo.vision.resnet50_v2(pretrained=True, ctx=ctx)
        BS = 1
        inp = mx.random.uniform(0, 1, (1, 3, 224, 224))
        model.hybridize(static_alloc=True)
        out = model(inp)

        prefix = f"{tmp_path}/resnet50"
        model.export(prefix)

        sym_file = f"{prefix}-symbol.json"
        params_file = f"{prefix}-0000.params"
        onnx_file = f"{prefix}.onnx"
    
        dynamic_input_shapes = [('batch', 3, 224, 224)]
        input_shapes = [(1, 3, 224, 224)]
        input_types = [np.float32]

        converted_model_path = mx.onnx.export_model(sym_file, params_file, input_shapes,
                                                    input_types, onnx_file,
                                                    dynamic=True,
                                                    dynamic_input_shapes=dynamic_input_shapes)

        ses_opt = onnxruntime.SessionOptions()
        ses_opt.log_severity_level = 3
        session = onnxruntime.InferenceSession(onnx_file, ses_opt)

        BS = 10
        inp = mx.random.uniform(0, 1, (1, 3, 224, 224))

        mx_out = model(inp)
        onnx_inputs = [inp]
        input_dict = dict((session.get_inputs()[i].name, onnx_inputs[i].asnumpy())
                           for i in range(len(onnx_inputs)))
        on_out = session.run(None, input_dict)

        assert_almost_equal(mx_out, on_out, rtol=0.001, atol=0.01)
    finally:
        shutil.rmtree(tmp_path)

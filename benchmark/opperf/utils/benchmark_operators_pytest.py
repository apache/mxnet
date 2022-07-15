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
import pytest

from benchmark.opperf.utils.benchmark_utils import run_benchmark_operator

test_cases = {
    "reshape"           : [((128,128,128), {"newshape": (128,256,-1)}),
                           ((256,256,256), {"newshape": (256,512,-1)}),
                           ((512,512,512), {"newshape": (512,1024,-1)}),],
    "swapaxes"          : [((64,128,64), {"axis1": 1, "axis2": 2}),
                           ((128,256,128), {"axis1": 1, "axis2": 2}),
                           ((256,512,256), {"axis1": 1, "axis2": 2})],
    "activation"        : [((128,128,128), {"actType": "relu"}),
                           ((256,256,256), {"actType": "relu"}),
                           ((512,512,512), {"actType": "relu"})],
    "batch_norm"        : [((128,128,128), {}),
                           ((256,256,256), {}),
                           ((512,512,512), {})],
    "convolution"       : [((16,16,16,16,16), {"numFilter": 8, "kernel": (3,3,3)}),
                           ((32,32,16,16,16), {"numFilter": 16, "kernel": (5,5,5)}),
                           ((32,32,32,32,32), {"numFilter": 16, "kernel": (7,7,7)})],
    "add"               : [((128,128,128), {}),
                           ((256,256,256), {}),
                           ((512,512,512), {})],
    "masked_softmax"    : [((128,128,128), {}),
                           ((256,256,256), {}),
                           ((512,512,512), {})],
    "slice"             : [((128,128,128), {"begin": (32,32,32), "end": (-32,-32,-32)}),
                           ((256,256,256), {"begin": (64,64,64), "end": (-64,-64,-64)}),
                           ((512,512,512), {"begin": (96,96,96), "end": (-96,-96,-96)})],
    "fully_connected"   : [((20,20,20,20), {"numHidden": 30}),
                           ((60,60,60,60), {"numHidden": 60}),
                           ((90,90,90,90), {"numHidden": 90}),],
    "batch_dot"         : [((10,10,10), {"matrix1": (20,30), "matrix2": (30,40)}),
                           ((20,20,20), {"matrix1": (40,50), "matrix2": (50,60)}),
                           ((40,40,40), {"matrix1": (60,70), "matrix2": (70,80)})]
}

def generate_test_cases():
    tests = []
    for op_name, cases in test_cases.items():
        for case in cases:
            tests.append((op_name, case[0], case[1]))
    return tests

def generate_test_ids():
    test_ids = []
    for op_name, cases in test_cases.items():
        for case in cases:
            s = op_name + "-shape_"
            for i in range(len(case[0])):
                s += str(case[0][i])
                if (i != len(case[0])-1):
                    s += "x"
            params = case[1].items()
            if len(params) != 0:
                s += "-params"
                for key, value in params:
                    s += "_" + str(key) + "_"
                    if isinstance(value, tuple):
                        for i in range(len(value)):
                            s += str(value[i])
                            if (i != len(value)-1):
                                s += "x"
                    else:
                        s += str(value)
            test_ids.append(s)
    return test_ids

generate_inputs = {
    "reshape"               : lambda shape, metadata: {"newshape": metadata["newshape"], "shape": metadata["newshape"]},
    "swapaxes"              : lambda shape, metadata: {"axis1": metadata["axis1"], "axis2": metadata["axis2"],
                                                       "dim1": metadata["axis1"], "dim2": metadata["axis2"]},
    "activation"            : lambda shape, metadata: {"act_type": metadata["actType"]},
    "batch_norm"            : lambda shape, metadata: {"gamma": (shape[1],), "beta": (shape[1],), "running_mean": (shape[1],), "running_var": (shape[1],),
                                                       "moving_mean": (shape[1],), "moving_var": (shape[1],)},
    "convolution"           : lambda shape, metadata: {"weight": (metadata["numFilter"], shape[1]) + metadata["kernel"], "kernel": metadata["kernel"],
                                                       "bias": (metadata["numFilter"],), "num_filter": metadata["numFilter"]},
    "masked_softmax"        : lambda shape, metadata: {"mask": mx.np.array(round(mx.np.random.rand(*shape)), dtype="bool")},
    "fully_connected"       : lambda shape, metadata: {"weight": (metadata["numHidden"], shape[-1]), "bias": (metadata["numHidden"],), 
                                                       "num_hidden": metadata["numHidden"], "flatten": False},
    "batch_dot"             : lambda shape, metadata: {"lhs": shape + metadata["matrix1"], "a": shape + metadata["matrix1"],
                                                       "rhs": shape + metadata["matrix2"], "b": shape + metadata["matrix2"]},
    "slice"                 : lambda shape, metadata: {"begin": metadata["begin"], "end": metadata["end"]}
}

@pytest.mark.parametrize(argnames=("op_name, shape, params"), argvalues=generate_test_cases(), ids=generate_test_ids())
def test(op_name, shape, params):
    if op_name in generate_inputs.keys():
        additional_inputs = generate_inputs[op_name](shape,params)
    else:
        additional_inputs = {}
    run_benchmark_operator(name=op_name, size=shape, additional_inputs=additional_inputs, profiler="python")

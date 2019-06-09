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

"""
Tests for individual operators
This module contains operator tests which currently do not exist on
ONNX backend test framework. Once we have PRs on the ONNX repo and get
those PRs merged, this file will get EOL'ed.
"""
# pylint: disable=too-many-locals,wrong-import-position,import-error
from __future__ import absolute_import
import sys
import os
import unittest
import logging
import tarfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import numpy_helper, helper, load_model
from onnx import TensorProto
from mxnet.test_utils import download
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
import backend

CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../../python/unittest'))

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def get_rnd(shape, low=-1.0, high=1.0, dtype=np.float32):
    if dtype == np.float32:
        return (np.random.uniform(low, high,
                                  np.prod(shape)).reshape(shape).astype(np.float32))
    elif dtype == np.int32:
        return (np.random.randint(low, high,
                                  np.prod(shape)).reshape(shape).astype(np.float32))
    elif dtype == np.bool_:
        return np.random.choice(a=[False, True], size=shape).astype(np.float32)


def forward_pass(sym, arg, aux, data_names, input_data):
    """ Perform forward pass on given data
    :param sym: Symbol
    :param arg: Arg params
    :param aux: Aux params
    :param data_names: Input names (list)
    :param input_data: Input data (list). If there is only one input,
                        pass it as a list. For example, if input is [1, 2],
                        pass input_data=[[1, 2]]
    :return: result of forward pass
    """
    data_shapes = []
    data_forward = []
    for idx in range(len(data_names)):
        val = input_data[idx]
        data_shapes.append((data_names[idx], np.shape(val)))
        data_forward.append(mx.nd.array(val))
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)
    if not arg and not aux:
        mod.init_params()
    else:
        mod.set_params(arg_params=arg, aux_params=aux,
                       allow_missing=True, allow_extra=True)
    # run inference
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch(data_forward), is_train=False)

    return mod.get_outputs()[0].asnumpy()


class TestNode(unittest.TestCase):
    """ Tests for models.
    Tests are dynamically added.
    Therefore edit test_models to add more tests.
    """

    def test_import_export(self):
        def get_input_tensors(input_data):
            input_tensor = []
            input_names = []
            input_sym = []
            for idx, ip in enumerate(input_data):
                name = "input" + str(idx + 1)
                input_sym.append(mx.sym.Variable(name))
                input_names.append(name)
                input_tensor.append(helper.make_tensor_value_info(name,
                                                                  TensorProto.FLOAT, shape=np.shape(ip)))
            return input_names, input_tensor, input_sym

        def get_onnx_graph(testname, input_names, inputs, output_name, output_shape, attr):
            outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=output_shape)]

            nodes = [helper.make_node(output_name, input_names, ["output"], **attr)]

            graph = helper.make_graph(nodes, testname, inputs, outputs)

            model = helper.make_model(graph)
            return model

        for test in test_cases:
            test_name, mxnet_op, onnx_name, inputs, attrs, mxnet_specific = test
            with self.subTest(test_name):
                names, input_tensors, inputsym = get_input_tensors(inputs)
                test_op = mxnet_op(*inputsym, **attrs)
                mxnet_output = forward_pass(test_op, None, None, names, inputs)
                outputshape = np.shape(mxnet_output)

                if mxnet_specific:
                    onnxmodelfile = onnx_mxnet.export_model(test_op, {}, [np.shape(ip) for ip in inputs],
                                                            np.float32,
                                                            onnx_name + ".onnx")
                    onnxmodel = load_model(onnxmodelfile)
                else:
                    onnxmodel = get_onnx_graph(test_name, names, input_tensors, onnx_name, outputshape, attrs)

                bkd_rep = backend.prepare(onnxmodel, operation='export')
                output = bkd_rep.run(inputs)

                npt.assert_almost_equal(output[0], mxnet_output)


# test_case = ("test_case_name", mxnet op, "ONNX_op_name", [input_list], attribute map, MXNet_specific=True/False)
test_cases = [
    ("test_equal", mx.sym.broadcast_equal, "Equal", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False),
    ("test_greater", mx.sym.broadcast_greater, "Greater", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False),
    ("test_less", mx.sym.broadcast_lesser, "Less", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False),
    ("test_and", mx.sym.broadcast_logical_and, "And",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False),
    ("test_xor", mx.sym.broadcast_logical_xor, "Xor",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False),
    ("test_or", mx.sym.broadcast_logical_or, "Or",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False),
    ("test_not", mx.sym.logical_not, "Not", [get_rnd((3, 4, 5), dtype=np.bool_)], {}, False),
    ("test_square", mx.sym.square, "Pow", [get_rnd((2, 3), dtype=np.int32)], {}, True),
    ("test_spacetodepth", mx.sym.space_to_depth, "SpaceToDepth", [get_rnd((1, 1, 4, 6))],
     {'block_size': 2}, False),
    ("test_softmax", mx.sym.SoftmaxOutput, "Softmax", [get_rnd((1000, 1000)), get_rnd(1000)],
     {'ignore_label': 0, 'use_ignore': False}, True),
    ("test_fullyconnected", mx.sym.FullyConnected, "Gemm", [get_rnd((4,3)), get_rnd((4, 3)), get_rnd(4)],
     {'num_hidden': 4, 'name': 'FC'}, True)
]

if __name__ == '__main__':
    unittest.main()

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
import sys
import os
import unittest
import logging
import tarfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import checker, numpy_helper, helper, load_model
from onnx import TensorProto
from mxnet.test_utils import download
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


def _fix_attributes(attrs, attribute_mapping):
    new_attrs = attrs
    attr_modify = attribute_mapping.get('modify', {})
    for k, v in attr_modify.items():
        new_attrs[v] = new_attrs.pop(k, None)

    attr_add = attribute_mapping.get('add', {})
    for k, v in attr_add.items():
        new_attrs[k] = v

    attr_remove = attribute_mapping.get('remove', [])
    for k in attr_remove:
        if k in new_attrs:
            del new_attrs[k]

    return new_attrs


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

class TestNode(unittest.TestCase):
    """ Tests for models.
    Tests are dynamically added.
    Therefore edit test_models to add more tests.
    """
    def test_imports(self):
        for bk in ['mxnet', 'gluon']:
            for test in import_test_cases:
                test_name, onnx_name, inputs, np_op, attrs = test
                with self.subTest(test_name):
                    names, input_tensors, inputsym = get_input_tensors(inputs)
                    np_out = [np_op(*inputs, **attrs)]
                    output_shape = np.shape(np_out)
                    onnx_model = get_onnx_graph(test_name, names, input_tensors, onnx_name, output_shape, attrs)
                    bkd_rep = backend.prepare(onnx_model, operation='import', backend=bk)
                    mxnet_out = bkd_rep.run(inputs)
                    npt.assert_almost_equal(np_out, mxnet_out, decimal=4)

    def test_exports(self):
        for test in export_test_cases:
            test_name, onnx_name, mx_op, input_shape, attrs = test
            input_sym = mx.sym.var('data')
            if isinstance(mx_op, type) and issubclass(mx_op, (mx.gluon.HybridBlock, mx.gluon.SymbolBlock)):
                mx_op = mx_op(**attrs)
                mx_op.initialize()
                mx_op(mx.nd.zeros(input_shape))
                params = {p.var().name: p.data() for p in mx_op.collect_params().values()}
                outsym = mx_op(input_sym)
            else:
                params = {}
                outsym = mx_op(input_sym, **attrs)
            converted_model = mx.contrib.onnx.export_model(outsym, params, [input_shape], np.float32,
                                                           onnx_file_path=outsym.name + ".onnx")
            model = load_model(converted_model)
            checker.check_model(model)


# test_case = ("test_case_name", mxnet op, "ONNX_op_name", [input_list], attribute map, MXNet_specific=True/False,
# fix_attributes = {'modify': {mxnet_attr_name: onnx_attr_name},
#                   'remove': [attr_name],
#                   'add': {attr_name: value},
# check_value=True/False, check_shape=True/False)
test_cases = [
    ("test_equal", mx.sym.broadcast_equal, "Equal", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_greater", mx.sym.broadcast_greater, "Greater", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_less", mx.sym.broadcast_lesser, "Less", [get_rnd((1, 3, 4, 5)), get_rnd((1, 5))], {}, False, {}, True,
     False),
    ("test_and", mx.sym.broadcast_logical_and, "And",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_xor", mx.sym.broadcast_logical_xor, "Xor",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_or", mx.sym.broadcast_logical_or, "Or",
     [get_rnd((3, 4, 5), dtype=np.bool_), get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_not", mx.sym.logical_not, "Not", [get_rnd((3, 4, 5), dtype=np.bool_)], {}, False, {}, True, False),
    ("test_square", mx.sym.square, "Pow", [get_rnd((2, 3), dtype=np.int32)], {}, True, {}, True, False),
    ("test_spacetodepth", mx.sym.space_to_depth, "SpaceToDepth", [get_rnd((1, 1, 4, 6))],
     {'block_size': 2}, False, {}, True, False),
    ("test_fullyconnected", mx.sym.FullyConnected, "Gemm", [get_rnd((4, 3)), get_rnd((4, 3)), get_rnd(4)],
     {'num_hidden': 4, 'name': 'FC'}, True, {}, True, False),
    ("test_lppool1", mx.sym.Pooling, "LpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 1, 'pool_type': 'lp'}, False,
     {'modify': {'kernel': 'kernel_shape', 'pad': 'pads', 'stride': 'strides', 'p_value': 'p'},
      'remove': ['pool_type']}, True, False),
    ("test_lppool2", mx.sym.Pooling, "LpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 2, 'pool_type': 'lp'}, False,
     {'modify': {'kernel': 'kernel_shape', 'pad': 'pads', 'stride': 'strides', 'p_value': 'p'},
      'remove': ['pool_type']}, True, False),
    ("test_globallppool1", mx.sym.Pooling, "GlobalLpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 1, 'pool_type': 'lp', 'global_pool': True}, False,
     {'modify': {'p_value': 'p'},
      'remove': ['pool_type', 'kernel', 'pad', 'stride', 'global_pool']}, True, False),
    ("test_globallppool2", mx.sym.Pooling, "GlobalLpPool", [get_rnd((2, 3, 20, 20))],
     {'kernel': (4, 5), 'pad': (0, 0), 'stride': (1, 1), 'p_value': 2, 'pool_type': 'lp', 'global_pool': True}, False,
     {'modify': {'p_value': 'p'},
      'remove': ['pool_type', 'kernel', 'pad', 'stride', 'global_pool']}, True, False),
    ("test_roipool", mx.sym.ROIPooling, "MaxRoiPool",
     [[[get_rnd(shape=(8, 6), low=1, high=100, dtype=np.int32)]], [[0, 0, 0, 4, 4]]],
     {'pooled_size': (2, 2), 'spatial_scale': 0.7}, False,
     {'modify': {'pooled_size': 'pooled_shape'}}, True, False),

    # since results would be random, checking for shape alone
    ("test_multinomial", mx.sym.sample_multinomial, "Multinomial",
     [np.array([0, 0.1, 0.2, 0.3, 0.4]).astype("float32")],
     {'shape': (10,)}, False, {'modify': {'shape': 'sample_size'}}, False, True),
    ("test_random_normal", mx.sym.random_normal, "RandomNormal", [],
     {'shape': (2, 2), 'loc': 0, 'scale': 1}, False, {'modify': {'loc': 'mean'}}, False, True),
    ("test_random_uniform", mx.sym.random_uniform, "RandomUniform", [],
     {'shape': (2, 2), 'low': 0.5, 'high': 1.0}, False, {}, False, True)
]

test_scalar_ops = ['Add', 'Sub', 'rSub' 'Mul', 'Div', 'Pow']

# test_case = ("test_case_name", "ONNX_op_name", [input_list], np_op, attribute map)
import_test_cases = [
    ("test_lpnormalization_default", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':2, 'axis':-1}),
    ("test_lpnormalization_ord1", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':1, 'axis':-1}),
    ("test_lpnormalization_ord2", "LpNormalization", [get_rnd([5, 3, 3, 2])], np.linalg.norm, {'ord':2, 'axis':1})
]

# test_case = ("test_case_name", "ONNX_op_name", mxnet_op, input_shape, attribute map)
export_test_cases = [
    ("test_expand", "Expand", mx.sym.broadcast_to, (2,1,3,1), {'shape': (2,1,3,1)}),
    ("test_tile", "Tile", mx.sym.tile, (2,1,3,1), {'reps': (2,3)}),
    ("test_topk", "TopK", mx.sym.topk, (2, 10, 2), {'k': 3, 'axis': 1, 'ret_typ': 'both', 'dtype': np.int64}),
    ("test_slice_axis", "Slice", mx.sym.slice_axis, (2, 10, 2), {'begin': 3, 'end': 7, 'axis': 1}),
    ("test_LSTM", "LSTM", mx.gluon.rnn.LSTM, (3,1,2), {'hidden_size': 3}),
    ("test_BiLSTM", "LSTM", mx.gluon.rnn.LSTM, (3,1,2), {'hidden_size': 3, 'bidirectional': True}),
]

if __name__ == '__main__':
    unittest.main()

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
import tempfile
from collections import namedtuple
import numpy as np
import numpy.testing as npt
from onnx import numpy_helper, helper
from onnx import TensorProto
from mxnet import nd, sym
from mxnet.gluon import nn
from mxnet.test_utils import download
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
CURR_PATH = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(CURR_PATH, '../../../python/unittest'))
import backend
from common import with_seed

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
URLS = {
    'bvlc_googlenet':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_googlenet.tar.gz',
    'bvlc_reference_caffenet':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_caffenet.tar.gz',
    'bvlc_reference_rcnn_ilsvrc13':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/bvlc_reference_rcnn_ilsvrc13.tar.gz',
    'inception_v1':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/inception_v1.tar.gz',
    'inception_v2':
        'https://s3.amazonaws.com/onnx-mxnet/model-zoo/inception_v2.tar.gz'
}

def get_test_files(name):
    """Extract tar file and returns model path and input, output data"""
    tar_name = download(URLS.get(name), dirname=CURR_PATH.__str__())
    # extract tar file
    tar_path = os.path.join(CURR_PATH, tar_name)
    tar = tarfile.open(tar_path.__str__(), "r:*")
    tar.extractall(path=CURR_PATH.__str__())
    tar.close()
    data_dir = os.path.join(CURR_PATH, name)
    model_path = os.path.join(data_dir, 'model.onnx')

    inputs = []
    outputs = []
    # get test files
    for test_file in os.listdir(data_dir):
        case_dir = os.path.join(data_dir, test_file)
        # skip the non-dir files
        if not os.path.isdir(case_dir):
            continue
        input_file = os.path.join(case_dir, 'input_0.pb')
        input_tensor = TensorProto()
        with open(input_file, 'rb') as proto_file:
            input_tensor.ParseFromString(proto_file.read())
        inputs.append(numpy_helper.to_array(input_tensor))

        output_tensor = TensorProto()
        output_file = os.path.join(case_dir, 'output_0.pb')
        with open(output_file, 'rb') as proto_file:
            output_tensor.ParseFromString(proto_file.read())
        outputs.append(numpy_helper.to_array(output_tensor))

    return model_path, inputs, outputs


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
    # create module
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=mx.cpu(), label_names=None)

    data_shapes = []
    data_forward = []
    for idx in range(len(data_names)):
        val = input_data[idx]
        data_shapes.append((data_names[idx], np.shape(val)))
        data_forward.append(mx.nd.array(val))

    mod.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)
    mod.set_params(arg_params=arg, aux_params=aux,
                   allow_missing=True, allow_extra=True)

    # run inference
    batch = namedtuple('Batch', ['data'])
    mod.forward(batch(data_forward), is_train=False)

    return mod.get_outputs()[0].asnumpy()


def test_models(model_name, input_shape, output_shape):
    """ Tests Googlenet model for both onnx import and export"""
    model_path, inputs, outputs = get_test_files(model_name)
    logging.info("Translating model from ONNX model zoo to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)
    params = {}
    params.update(arg_params)
    params.update(aux_params)

    dir_path = os.path.dirname(model_path)
    new_model_name = "exported_" + model_name + ".onnx"
    onnx_file = os.path.join(dir_path, new_model_name)

    logging.info("Translating converted model from mxnet to ONNX")
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model_path)

    metadata = onnx_mxnet.get_model_metadata(converted_model_path)
    assert len(metadata) == 2
    assert metadata.get('input_tensor_data')
    assert metadata.get('input_tensor_data')[0][1] == input_shape
    assert metadata.get('output_tensor_data')
    assert metadata.get('output_tensor_data')[0][1] == output_shape
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    logging.info("Running inference on onnx re-import model in mxnet")
    # run test for each test file
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, [input_data])

        # verify the results
        npt.assert_equal(result.shape, output_data.shape)
        npt.assert_almost_equal(output_data, result, decimal=3)
    logging.info(model_name + " conversion successful")


def test_model_accuracy(model_name, input_shape):
    """ Imports ONNX model, runs inference, exports and imports back
        run inference, compare result with the previous inference result"""
    model_path, inputs, outputs = get_test_files(model_name)
    logging.info("Translating model from ONNX model zoo to Mxnet")
    sym, arg_params, aux_params = onnx_mxnet.import_model(model_path)

    metadata = onnx_mxnet.get_model_metadata(model_path)
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    expected_result= []
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, [input_data])
        expected_result.append(result)

    params = {}
    params.update(arg_params)
    params.update(aux_params)

    dir_path = os.path.dirname(model_path)
    new_model_name = "exported_" + model_name + ".onnx"
    onnx_file = os.path.join(dir_path, new_model_name)

    logging.info("Translating converted model from mxnet to ONNX")
    converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32,
                                                   onnx_file)

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model_path)

    metadata = onnx_mxnet.get_model_metadata(converted_model_path)
    data_names = [input_name[0] for input_name in metadata.get('input_tensor_data')]

    actual_result = []
    for input_data, output_data in zip(inputs, outputs):
        result = forward_pass(sym, arg_params, aux_params, data_names, [input_data])
        actual_result.append(result)

    # verify the results
    for expected, actual in zip(expected_result, actual_result):
        npt.assert_equal(expected.shape, actual.shape)
        npt.assert_almost_equal(expected, actual, decimal=3)

@with_seed()
def test_spacetodepth():
    n, c, h, w = shape = (1, 1, 4, 6)
    input1 = np.random.rand(n, c, h, w).astype("float32")
    blocksize = 2
    inputs = [helper.make_tensor_value_info("input1", TensorProto.FLOAT, shape=shape)]

    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=(1, 4, 2, 3))]

    nodes = [helper.make_node("SpaceToDepth", ["input1"], ["output"], block_size=blocksize)]

    graph = helper.make_graph(nodes,
                              "spacetodepth_test",
                              inputs,
                              outputs)

    spacetodepth_model = helper.make_model(graph)

    bkd_rep = backend.prepare(spacetodepth_model)
    output = bkd_rep.run([input1])

    tmp = np.reshape(input1, [n, c,
                    h // blocksize, blocksize,
                    w // blocksize, blocksize])
    tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
    numpy_op = np.reshape(tmp, [n, c * (blocksize**2),
                    h // blocksize,
                    w // blocksize])

    npt.assert_almost_equal(output[0], numpy_op)

@with_seed()
def test_square():
    input1 = np.random.randint(1, 10, (2, 3)).astype("float32")

    ipsym = mx.sym.Variable("input1")
    square = mx.sym.square(data=ipsym)
    model = mx.mod.Module(symbol=square, data_names=['input1'], label_names=None)
    model.bind(for_training=False, data_shapes=[('input1', np.shape(input1))], label_shapes=None)
    model.init_params()

    args, auxs = model.get_params()
    params = {}
    params.update(args)
    params.update(auxs)

    converted_model = onnx_mxnet.export_model(square, params, [np.shape(input1)], np.float32, "square.onnx")

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model)
    result = forward_pass(sym, arg_params, aux_params, ['input1'], [input1])

    numpy_op = np.square(input1)

    npt.assert_almost_equal(result, numpy_op)


@with_seed()
def test_fully_connected():
    def random_arrays(*shapes):
        """Generate some random numpy arrays."""
        arrays = [np.random.randn(*s).astype("float32")
                  for s in shapes]
        if len(arrays) == 1:
            return arrays[0]
        return arrays

    data_names = ['x', 'w', 'b']

    dim_in, dim_out = (3, 4)
    input_data = random_arrays((4, dim_in), (dim_out, dim_in), (dim_out,))

    ipsym = []
    data_shapes = []
    data_forward = []
    for idx in range(len(data_names)):
        val = input_data[idx]
        data_shapes.append((data_names[idx], np.shape(val)))
        data_forward.append(mx.nd.array(val))
        ipsym.append(mx.sym.Variable(data_names[idx]))

    op = mx.sym.FullyConnected(data=ipsym[0], weight=ipsym[1], bias=ipsym[2], num_hidden=dim_out, name='FC')

    model = mx.mod.Module(op, data_names=data_names, label_names=None)
    model.bind(for_training=False, data_shapes=data_shapes, label_shapes=None)

    model.init_params()

    args, auxs = model.get_params()
    params = {}
    params.update(args)
    params.update(auxs)

    converted_model = onnx_mxnet.export_model(op, params, [shape[1] for shape in data_shapes], np.float32, "fc.onnx")

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model)
    result = forward_pass(sym, arg_params, aux_params, data_names, input_data)

    numpy_op = np.dot(input_data[0], input_data[1].T) + input_data[2]

    npt.assert_almost_equal(result, numpy_op)


def test_softmax():
    input1 = np.random.rand(1000, 1000).astype("float32")
    label1 = np.random.rand(1000)
    input_nd = mx.nd.array(input1)
    label_nd = mx.nd.array(label1)

    ipsym = mx.sym.Variable("ipsym")
    label = mx.sym.Variable('label')
    sym = mx.sym.SoftmaxOutput(data=ipsym, label=label, ignore_label=0, use_ignore=False)
    ex = sym.bind(ctx=mx.cpu(0), args={'ipsym': input_nd, 'label': label_nd})
    ex.forward(is_train=True)
    softmax_out = ex.outputs[0].asnumpy()

    converted_model = onnx_mxnet.export_model(sym, {}, [(1000, 1000), (1000,)], np.float32, "softmaxop.onnx")

    sym, arg_params, aux_params = onnx_mxnet.import_model(converted_model)
    result = forward_pass(sym, arg_params, aux_params, ['ipsym'], input1)

    # Comparing result of forward pass before using onnx export, import
    npt.assert_almost_equal(result, softmax_out)

@with_seed()
def test_comparison_ops():
    """Test greater, lesser, equal"""
    def test_ops(op_name, inputs, input_tensors, numpy_op):
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=np.shape(inputs[0]))]
        nodes = [helper.make_node(op_name, ["input"+str(i+1) for i in range(len(inputs))], ["output"])]
        graph = helper.make_graph(nodes,
                                  op_name + "_test",
                                  input_tensors,
                                  outputs)
        model = helper.make_model(graph)
        bkd_rep = backend.prepare(model)
        output = bkd_rep.run(inputs)
        npt.assert_almost_equal(output[0], numpy_op)
    input_data = [np.random.rand(1, 3, 4, 5).astype("float32"),
                  np.random.rand(1, 5).astype("float32")]
    input_tensor = []
    for idx, ip in enumerate(input_data):
        input_tensor.append(helper.make_tensor_value_info("input" + str(idx + 1),
                                                          TensorProto.FLOAT, shape=np.shape(ip)))
    test_ops("Greater", input_data, input_tensor,
             np.greater(input_data[0], input_data[1]).astype(np.float32))
    test_ops("Less", input_data, input_tensor,
             np.less(input_data[0], input_data[1]).astype(np.float32))
    test_ops("Equal", input_data, input_tensor,
             np.equal(input_data[0], input_data[1]).astype(np.float32))


def get_int_inputs(interval, shape):
    """Helper to get integer input of given shape and range"""
    assert len(interval) == len(shape)
    inputs = []
    input_tensors = []
    for idx in range(len(interval)):
        low, high = interval[idx]
        inputs.append(np.random.randint(low, high, size=shape[idx]).astype("float32"))
        input_tensors.append(helper.make_tensor_value_info("input"+str(idx+1),
                                                        TensorProto.FLOAT, shape=shape[idx]))
    return inputs, input_tensors


@with_seed()
def test_logical_ops():
    """Test for logical and, or, not, xor operators"""
    def test_ops(op_name, inputs, input_tensors, numpy_op):
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=np.shape(inputs[0]))]
        nodes = [helper.make_node(op_name, ["input"+str(i+1) for i in range(len(inputs))], ["output"])]
        graph = helper.make_graph(nodes,
                                  op_name + "_test",
                                  input_tensors,
                                  outputs)
        model = helper.make_model(graph)
        bkd_rep = backend.prepare(model)
        output = bkd_rep.run(inputs)
        npt.assert_almost_equal(output[0], numpy_op)
    input_data, input_tensor = get_int_inputs([(0, 2), (0, 2)], [(3, 4, 5), (3, 4, 5)])
    test_ops("And", input_data, input_tensor,
             np.logical_and(input_data[0], input_data[1]).astype(np.float32))
    test_ops("Or", input_data, input_tensor,
             np.logical_or(input_data[0], input_data[1]).astype(np.float32))
    test_ops("Xor", input_data, input_tensor,
             np.logical_xor(input_data[0], input_data[1]).astype(np.float32))
    test_ops("Not", [input_data[0]], [input_tensor[0]],
             np.logical_not(input_data[0]).astype(np.float32))


def _assert_sym_equal(lhs, rhs):
    assert lhs.list_inputs() == rhs.list_inputs()  # input names must be identical
    assert len(lhs.list_outputs()) == len(rhs.list_outputs())  # number of outputs must be identical


def _force_list(output):
    if isinstance(output, nd.NDArray):
        return [output]
    return list(output)


def _optional_group(symbols, group=False):
    if group:
        return sym.Group(symbols)
    else:
        return symbols


def _check_onnx_export(net, group_outputs=False, shape_type=tuple, extra_params={}):
    net.initialize()
    data = nd.random.uniform(0, 1, (1, 1024))
    output = _force_list(net(data))  # initialize weights
    net_sym = _optional_group(net(sym.Variable('data')), group_outputs)
    net_params = {name:param._reduce() for name, param in net.collect_params().items()}
    net_params.update(extra_params)
    with tempfile.TemporaryDirectory() as tmpdirname:
        onnx_file_path = os.path.join(tmpdirname, 'net.onnx')
        export_path = onnx_mxnet.export_model(
            sym=net_sym,
            params=net_params,
            input_shape=[shape_type(data.shape)],
            onnx_file_path=onnx_file_path)
        assert export_path == onnx_file_path
        # Try importing the model to symbol
        _assert_sym_equal(net_sym, onnx_mxnet.import_model(export_path)[0])

        # Try importing the model to gluon
        imported_net = onnx_mxnet.import_to_gluon(export_path, ctx=None)
        _assert_sym_equal(net_sym, _optional_group(imported_net(sym.Variable('data')), group_outputs))

        # Confirm network outputs are the same
        imported_net_output = _force_list(imported_net(data))
        for out, imp_out in zip(output, imported_net_output):
            mx.test_utils.assert_almost_equal(out.asnumpy(), imp_out.asnumpy())


@with_seed()
def test_onnx_export_single_output():
    net = nn.HybridSequential(prefix='single_output_net')
    with net.name_scope():
        net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
    _check_onnx_export(net)


@with_seed()
def test_onnx_export_multi_output():
    class MultiOutputBlock(nn.HybridBlock):
        def __init__(self):
            super(MultiOutputBlock, self).__init__()
            with self.name_scope():
                self.net = nn.HybridSequential()
                for i in range(10):
                    self.net.add(nn.Dense(100 + i * 10, activation='relu'))

        def hybrid_forward(self, F, x):
            out = tuple(block(x) for block in self.net._children.values())
            return out

    net = MultiOutputBlock()
    assert len(sym.Group(net(sym.Variable('data'))).list_outputs()) == 10
    _check_onnx_export(net, group_outputs=True)


@with_seed()
def test_onnx_export_list_shape():
    net = nn.HybridSequential(prefix='list_shape_net')
    with net.name_scope():
        net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
    _check_onnx_export(net, shape_type=list)


@with_seed()
def test_onnx_export_extra_params():
    net = nn.HybridSequential(prefix='extra_params_net')
    with net.name_scope():
        net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
    _check_onnx_export(net, extra_params={'extra_param': nd.array([1, 2])})


if __name__ == '__main__':
    test_models("bvlc_googlenet", (1, 3, 224, 224), (1, 1000))
    test_models("bvlc_reference_caffenet", (1, 3, 224, 224), (1, 1000))
    test_models("bvlc_reference_rcnn_ilsvrc13", (1, 3, 224, 224), (1, 200))

    # Comparing MXNet inference result, since MXNet results don't match
    # ONNX expected results due to AveragePool issue github issue(#10194)
    test_model_accuracy("inception_v1", (1, 3, 224, 224))
    test_model_accuracy("inception_v2", (1, 3, 224, 224))

    unittest.main()

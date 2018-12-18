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


# pylint: disable=too-many-locals,wrong-import-position,import-error
from __future__ import absolute_import
import os
import unittest
import logging
import tempfile
from mxnet import nd, sym
from mxnet.gluon import nn
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
    net_params = {name: param._reduce() for name, param in net.collect_params().items()}
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


class TestExport(unittest.TestCase):
    """ Tests ONNX export.
    """

    def test_onnx_export_single_output(self):
        net = nn.HybridSequential(prefix='single_output_net')
        with net.name_scope():
            net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
        _check_onnx_export(net)

    def test_onnx_export_multi_output(self):
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

    def test_onnx_export_list_shape(self):
        net = nn.HybridSequential(prefix='list_shape_net')
        with net.name_scope():
            net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
        _check_onnx_export(net, shape_type=list)

    def test_onnx_export_extra_params(self):
        net = nn.HybridSequential(prefix='extra_params_net')
        with net.name_scope():
            net.add(nn.Dense(100, activation='relu'), nn.Dense(10))
        _check_onnx_export(net, extra_params={'extra_param': nd.array([1, 2])})


if __name__ == '__main__':
    unittest.main()

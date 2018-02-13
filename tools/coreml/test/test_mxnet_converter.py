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

import unittest
import mxnet as mx
import numpy as np
import sys
import os
current_working_directory = os.getcwd()
sys.path.append(current_working_directory + "/..")
sys.path.append(current_working_directory + "/../converter/")
import _mxnet_converter as mxnet_converter
from collections import namedtuple
from converter import utils

def _mxnet_remove_batch(input_data):
    for blob in input_data:
        input_data[blob] = np.reshape(input_data[blob], input_data[blob].shape[1:])
    return input_data


def _get_mxnet_module(net, data_shapes, mode, label_names, input_names=None):
    """ Given a symbolic graph, input shape and the initialization mode,
        returns an MXNet module.
    """
    mx.random.seed(1993)

    mod = utils.create_module(sym=net, data_shapes=data_shapes, label_shapes=input_names, label_names=label_names)

    if mode == 'random':
        mod.init_params(
            initializer=mx.init.Uniform(scale=.1)
        )
    elif mode == 'zeros':
        mod.init_params(
            initializer=mx.init.Zero()
        )
    elif mode == 'ones':
        mod.init_params(
            initializer=mx.init.One()
        )
    else:
        Exception(KeyError("%s is not a valid initialization mode" % mode))

    return mod


class SingleLayerTest(unittest.TestCase):
    """
    Unit test class for testing where converter is able to convert individual layers or not.
    In order to do so, it converts model and generates preds on both CoreML and MXNet and check they are the same.
    """
    def _test_mxnet_model(self, net, input_shape, mode, class_labels=None, coreml_mode=None, label_names=None, delta=1e-3,
                          pre_processing_args=None, input_name='data'):
        """ Helper method that convert the CoreML model into CoreML and compares the predictions over random data.

        Parameters
        ----------
        net: MXNet Symbol Graph
            The graph that we'll be converting into CoreML.

        input_shape: tuple of ints
            The shape of input data. Generally of the format (batch-size, channels, height, width)

        mode: (random|zeros|ones)
            The mode to use in order to set the parameters (weights and biases).

        label_names: list of strings
            The names of the output labels. Default: None

        delta: float
            The maximum difference b/w predictions of MXNet and CoreML that is tolerable.

        input_name: str
            The name of the input variable to the symbolic graph.
        """

        data_shapes=[(input_name, input_shape)]

        mod = _get_mxnet_module(net, data_shapes, mode, label_names)

        # Generate some dummy data
        input_data = {input_name: np.random.uniform(-10., 10., input_shape)}
        Batch = namedtuple('Batch', ['data'])
        mod.forward(Batch([mx.nd.array(input_data[input_name])]))
        mxnet_preds = mod.get_outputs()[0].asnumpy().flatten()

        # Get predictions from coreml
        coreml_model = mxnet_converter.convert(
            model=mod,
            class_labels=class_labels,
            mode=coreml_mode,
            input_shape={input_name: input_shape},
            preprocessor_args=pre_processing_args
        )
        coreml_preds = coreml_model.predict(_mxnet_remove_batch(input_data)).values()[0].flatten()

        # Check prediction accuracy
        self.assertEquals(len(mxnet_preds), len(coreml_preds))
        for i in range(len(mxnet_preds)):
            self.assertAlmostEquals(mxnet_preds[i], coreml_preds[i], delta = delta)

    def test_tiny_inner_product_zero_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        self._test_mxnet_model(net, input_shape=input_shape, mode='zeros')

    def test_really_tiny_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=1)
        self._test_mxnet_model(net, input_shape=input_shape, mode='ones')

    def test_really_tiny_2_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        self._test_mxnet_model(net, input_shape=input_shape, mode='ones')

    def test_tiny_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        self._test_mxnet_model(net, input_shape=input_shape, mode='ones')

    def test_tiny_inner_product_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_inner_product_no_bias(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5, no_bias=True)
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_softmax_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        self._test_mxnet_model(net, input_shape=input_shape, mode='random', label_names=['softmax_label'])

    def test_tiny_relu_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.Activation(net, name='relu1', act_type="relu")
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_sigmoid_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.Activation(net, name='sigmoid1', act_type="sigmoid")
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_tanh_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.Activation(net, name='tanh1', act_type="tanh")
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_really_tiny_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (1 ,1)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='ones')

    def test_tiny_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_asym_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5 ,3)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_asym_conv_random_asym_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 28, 18)
        num_filter = 16
        kernel = (5, 3)
        stride = (1, 1)
        pad = (0, 0)
        dilate = (1, 1)
        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1',
            dilate=dilate)
        net = mx.sym.Activation(net, name='tanh', act_type="tanh")
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_valid_pooling_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (2, 2)
        stride = (2, 2)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        net = mx.symbol.Pooling(
            data=net,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='pool_1',
            pool_type='avg',
            pooling_convention='valid'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_pooling_full_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (2, 2)
        stride = (2, 2)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        net = mx.symbol.Pooling(
            data=net,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='pool_1',
            pool_type='avg',
            pooling_convention='full'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_pooling_full_random_input_with_padding(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 2
        kernel = (2, 2)
        stride = (2, 2)
        pad = (1, 1)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        net = mx.symbol.Pooling(
            data=net,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='pool_1',
            pool_type='avg',
            pooling_convention='full'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_really_tiny_conv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (1, 1)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_really_tiny_conv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (1, 1)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_conv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_conv_random(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_flatten(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        net = mx.sym.Flatten(data=net, name='flatten1')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        self._test_mxnet_model(net, input_shape=input_shape, mode='random', label_names=['softmax_label'])

    def test_transpose(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        net = mx.sym.Variable('data')
        net = mx.sym.transpose(data=net, name='transpose', axes=(0, 1, 2, 3))
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_reshape(self):
        np.random.seed(1988)
        input_shape = (1, 8)
        net = mx.sym.Variable('data')
        net = mx.sym.reshape(data=net, shape=(1, 2, 2, 2))
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_synset_random_input(self):
        np.random.seed(1989)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        mod = _get_mxnet_module(net,
                                data_shapes=[('data', input_shape)],
                                mode='random',
                                label_names=['softmax_label'])

        # Generate some dummy data
        input_data = np.random.uniform(-0.1, 0.1, input_shape)

        Batch = namedtuple('Batch', ['data'])
        mod.forward(Batch([mx.nd.array(input_data)]))

        kwargs = {'input_shape': {'data': input_shape}}
        # Get predictions from coreml
        coreml_model = mxnet_converter.convert(
            model=mod,
            class_labels=['Category1','Category2','Category3','Category4','Category5'],
            mode='classifier',
            **kwargs
        )

        prediction = coreml_model.predict(_mxnet_remove_batch({'data': input_data}))
        self.assertEqual(prediction['classLabel'], 'Category3')

    def test_really_tiny_deconv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (1, 1)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_deconv_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='ones')

    def test_tiny_deconv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_asym_deconv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 3)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_asym_deconv_random_asym_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 28, 18)
        num_filter = 16
        kernel = (5, 3)
        stride = (1, 1)
        pad = (0, 0)
        dilate = (1, 1)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            dilate=dilate,
            name='deconv_1'
        )
        net = mx.sym.Activation(net, name = 'tanh', act_type = "tanh")
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_deconv_pooling_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        net = mx.symbol.Pooling(
            data=net,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='pool_1',
            pool_type='max'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_really_tiny_deconv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (1, 1)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_really_tiny_deconv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (1, 1)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_deconv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (5, 5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_tiny_deconv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            name='deconv_1'
        )
        # Test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_deconv_random(self):
        np.random.seed(1988)
        input_shape = (1, 10, 4, 4)
        num_filter = 3
        kernel = (2, 2)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            name='deconv_1'
        )
        # test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_deconv_random_output_shape(self):
        np.random.seed(1988)
        input_shape = (1, 10, 4, 4)
        num_filter = 3
        kernel = (2, 2)
        stride = (1, 1)
        pad = (0, 0)
        target_shape = (5, 5)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            target_shape=target_shape,
            name='deconv_1'
        )
        # test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_deconv_random_padding(self):
        np.random.seed(1988)
        input_shape = (1, 10, 9, 9)
        num_filter = 3
        kernel = (3, 3)
        stride = (3, 3)
        pad = (2, 2)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
                data=net,
                num_filter=num_filter,
                kernel=kernel,
                stride=stride,
                pad=pad,
                no_bias=False,
                name='deconv_1')
        # test the mxnet model
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_conv_random_padding_odd(self):
        np.random.seed(1988)
        input_shape = (1, 10, 6, 6)
        num_filter = 3
        kernel = (5, 5)
        stride = (1, 1)
        pad = (3, 3)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_conv_random_padding_even(self):
        np.random.seed(1988)
        input_shape = (1, 10, 6, 6)
        num_filter = 3
        kernel = (5, 5)
        stride = (1, 1)
        pad = (2, 2)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_deconv_random_all_inputs(self):
        np.random.seed(1988)
        input_shape = (1, 10, 5, 5)
        num_filter = 3
        kernel = (3, 3)
        stride = (2, 2)
        pad = (1, 1)
        dilate = (1, 1)
        target_shape = (11, 11)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Deconvolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=False,
            target_shape=target_shape,
            dilate=dilate,
            name='deconv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    def test_batch_norm(self):
        np.random.seed(1988)
        input_shape = (1, 1, 2, 3)

        net = mx.sym.Variable('data')
        gamma = mx.sym.Variable('gamma')
        beta = mx.sym.Variable('beta')
        moving_mean = mx.sym.Variable('moving_mean')
        moving_var = mx.sym.Variable('moving_var')
        net = mx.symbol.BatchNorm(
            data=net,
            gamma=gamma,
            beta=beta,
            moving_mean=moving_mean,
            moving_var=moving_var,
            use_global_stats=True,
            name='batch_norm_1')
        self._test_mxnet_model(net, input_shape=input_shape, mode='random', delta=1e-2)

    def test_batch_norm_no_global_stats(self):
        """ This test should throw an exception since converter doesn't support
            conversion of MXNet models that use local batch stats (i.e.
            use_global_stats=False). The reason for this is CoreML doesn't support
            local batch stats.
        """
        np.random.seed(1988)
        input_shape = (1, 1, 2, 3)

        net = mx.sym.Variable('data')
        gamma = mx.sym.Variable('gamma')
        beta = mx.sym.Variable('beta')
        moving_mean = mx.sym.Variable('moving_mean')
        moving_var = mx.sym.Variable('moving_var')
        net = mx.symbol.BatchNorm(
            data=net,
            gamma=gamma,
            beta=beta,
            moving_mean=moving_mean,
            moving_var=moving_var,
            use_global_stats=False,
            name='batch_norm_1')
        self._test_mxnet_model(net, input_shape=input_shape, mode='random', delta=1e-2)

    def test_pre_processing_args(self):
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        net = mx.sym.SoftmaxOutput(net, name='softmax')
        self._test_mxnet_model(net, input_shape=input_shape, mode='random', label_names=['softmax_label'],
                               pre_processing_args={'red_bias':0, 'blue_bias':0, 'green_bias':0, 'image_scale':1})

    def test_different_input_variables(self):
        """
        Verifying the behavior when input variable name is different than the standard name - 'data'.
        """
        np.random.seed(1988)
        input_shape = (1, 10)
        net = mx.sym.Variable('data1')
        net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=5)
        self._test_mxnet_model(net, input_shape=input_shape, mode='zeros', input_name='data1')

    def test_really_tiny_conv_optional_params(self):
        """
        Verifying the behavior of a convolutional layer when stride and pad are not provided.
        """
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (1 ,1)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(
            data=net,
            num_filter=num_filter,
            kernel=kernel,
            name='conv_1'
        )
        self._test_mxnet_model(net, input_shape=input_shape, mode='random')

    # TODO test_concat


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(SingleLayerTest)
    unittest.TextTestRunner(verbosity=2).run(suite)

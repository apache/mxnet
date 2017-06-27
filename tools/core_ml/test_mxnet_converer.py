import unittest 
import mxnet as mx
import numpy as np
import tempfile
import os
import mxnet_converter
import coremltools

#TODO add integration tests for converting sets of layers from data.mxnet.io
#TODO add all new unit tests for layers

def _mxnet_remove_batch(input_data):
    for blob in input_data:
        input_data[blob] = np.reshape(input_data[blob], input_data[blob].shape[1:])
    return input_data

def _get_coreml_model(net, engine, model_path, input_shape,
            input_names = ['data'], output_names = ['output']):
    
    # TODO upgrade this to mxnet mod
    model = mx.model.FeedForward(net, engine, arg_params = engine.arg_dict)
    spec = mxnet_converter.convert(model, **input_shape)
    return coremltools.models.MLModel(spec)

def set_weights(net, engine, mode = 'random'):
    for arg in net.list_arguments():
        if mode == 'random':
            engine.arg_dict[arg][:] = np.random.uniform(-0.1, 0.1, engine.arg_dict[arg].shape)
        elif mode == 'zeros':
            engine.arg_dict[arg][:] = np.zeros(engine.arg_dict[arg].shape)
        elif mode == 'ones':
            engine.arg_dict[arg][:] = np.ones(engine.arg_dict[arg].shape)
    return net

class MXNetSingleLayerTest(unittest.TestCase):
    """
    Unit test class for testing mxnet converter (converts model and generates preds on same data to assert they are the same).
    """
    def _test_mxnet_model(self, net, engine, delta = 1e-3, **input_shape):

        # Generate some dummy data
        input_data = {}
        for ip in input_shape:
            input_data[ip] = engine.arg_dict[ip].asnumpy()
        output_blob = net.list_outputs()[0]

        # Make predictions from mxnet (only works on single output for now)
        mxnet_preds = engine.forward()[0].asnumpy().flatten()

        # Get predictions from coreml
        model_path = os.path.join(tempfile.mkdtemp(), 'mxnet.mlmodel')
        model = _get_coreml_model(net, engine, model_path, input_shape, input_data.keys())
        coreml_preds = model.predict(_mxnet_remove_batch(input_data)).values()[0].flatten()

        # Check prediction accuracy
        self.assertEquals(len(mxnet_preds), len(coreml_preds))
        for i in range(len(mxnet_preds)):
            self.assertAlmostEquals(mxnet_preds[i], coreml_preds[i], delta = delta)

    def test_tiny_inner_product_zero_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        engine = net.simple_bind(ctx=mx.cpu(), data=input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'zeros')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_really_tiny_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 1)
        engine = net.simple_bind(ctx=mx.cpu(), data=input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'ones')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_really_tiny_2_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        engine = net.simple_bind(ctx=mx.cpu(), data=input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'ones')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_inner_product_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        engine = net.simple_bind(ctx=mx.cpu(), data=input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'ones')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_inner_product_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        engine = net.simple_bind(ctx=mx.cpu(), data=input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_softmax_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        net = mx.sym.SoftmaxOutput(net, name = 'softmax')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_relu_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        net = mx.sym.Activation(net, name = 'relu1', act_type = "relu")
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_sigmoid_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        net = mx.sym.Activation(net, name = 'sigmoid1', act_type = "sigmoid")
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_tanh_activation_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 10)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        net = mx.sym.Activation(net, name = 'tanh1', act_type = "tanh")
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_really_tiny_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (1 ,1)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'random')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_conv_ones_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # Define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # Set some random weights
        set_weights(net, engine, mode = 'ones')

        # Test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_asym_conv_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5 ,3)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_asym_conv_random_asym_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 28, 18)
        num_filter = 16
        kernel = (5 ,3)
        stride = (1, 1)
        pad = (0, 0)
        dilate = (1, 1)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1', dilate = dilate)
        net = mx.sym.Activation(net, name = 'tanh', act_type = "tanh")
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_conv_pooling_random_input(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 1
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        net = mx.symbol.Pooling(data = net, kernel=kernel,
                stride = stride, pad = pad, name = 'pool_1', pool_type = 'max')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_really_tiny_conv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (1 ,1)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_really_tiny_conv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (1 ,1)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_conv_random_3d_input(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 1
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_tiny_conv_random_input_multi_filter(self):
        np.random.seed(1988)
        input_shape = (1, 1, 10, 10)
        num_filter = 64
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_conv_random(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_flatten(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        net = mx.sym.Flatten(data = net, name = 'flatten1')
        net = mx.sym.FullyConnected(data = net, name = 'fc1', num_hidden = 5)
        net = mx.sym.SoftmaxOutput(net, name = 'softmax')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

    def test_transpose(self):
        np.random.seed(1988)
        input_shape = (1, 3, 10, 10)
        num_filter = 64
        kernel = (5 ,5)
        stride = (1, 1)
        pad = (0, 0)

        # define a model
        net = mx.sym.Variable('data')
        net = mx.sym.transpose(data = net, name = 'transpose', axes = (0, 1, 2, 3))
        net = mx.symbol.Convolution(data = net, num_filter = num_filter, kernel=kernel,
                stride = stride, pad = pad, name = 'conv_1')
        engine = net.simple_bind(ctx = mx.cpu(), data = input_shape)

        # set some random weights
        set_weights(net, engine, mode = 'random')

        # test the mxnet model
        self._test_mxnet_model(net, engine, data = input_shape)

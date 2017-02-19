import numpy as np
import mxnet as mx
import math
from mxnet.test_utils import *

# Common

def test_lr_wd_mult():
    data = mx.sym.Variable('data')
    bias = mx.sym.Variable('fc1_bias', lr_mult=1.0)
    fc1 = mx.sym.FullyConnected(data=data, bias=bias, name='fc1', num_hidden=10, lr_mult=0)
    fc2 = mx.sym.FullyConnected(data=fc1, name='fc2', num_hidden=10, wd_mult=0.5)

    mod = mx.mod.Module(symbol=fc2, label_names=None, context=default_context())
    mod.bind(data_shapes=[('data', (5,10))])
    mod.init_params(initializer=mx.init.Uniform(1.0))
    mod.init_optimizer(optimizer_params={'learning_rate': 1.0})
    args1, _ = mod.get_params()
    args1 = {k: v.asnumpy() for k, v in args1.items()}
    mod.forward(mx.io.DataBatch(data=[mx.random.uniform(low=-1.0, high=1.0, shape=(5,10))], label=None), is_train=True)
    mod.backward(mod.get_outputs())
    mod.update()
    args2, _ = mod.get_params()
    args2 = {k: v.asnumpy() for k, v in args2.items()}

    assert mod._optimizer.lr_mult == {'fc1_bias': 1.0, 'fc1_weight': 0.0}
    assert mod._optimizer.wd_mult == {'fc2_bias': 0.5, 'fc2_weight': 0.5, 'fc1_bias': 0.0}
    assert mx.test_utils.almost_equal(args1['fc1_weight'], args2['fc1_weight'], 1e-10)
    assert not mx.test_utils.almost_equal(args1['fc1_bias'], args2['fc1_bias'], 1e-1)
    assert not mx.test_utils.almost_equal(args1['fc2_weight'], args2['fc2_weight'], 1e-1)


def compare_optimizer(opt1, opt2, shape):
    w1 = mx.random.uniform(shape=shape, ctx=default_context())
    g1 = mx.random.uniform(shape=shape, ctx=default_context())

    w2 = w1.copyto(default_context())
    g2 = g1.copyto(default_context())

    state1 = opt1.create_state(0, w1)
    state2 = opt2.create_state(0, w2)
    for s1, s2, in zip(state1, state2):
        assert(same(s1.asnumpy(), s2.asnumpy()))

    opt1.update(0, w1, g1, state1)
    opt2.update(0, w2, g2, state2)
    for s1, s2, in zip(state1, state2):
        assert_almost_equal(s1.asnumpy(), s2.asnumpy(), rtol=1e-4, atol=1e-5)
    assert_almost_equal(w1.asnumpy(), w2.asnumpy(), rtol=1e-4, atol=1e-5)

# ADAM

class PyAdam(mx.optimizer.Optimizer):
    """python reference implemenation of adam"""
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 decay_factor=(1 - 1e-8), **kwargs):
        super(PyAdam, self).__init__(learning_rate=learning_rate, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.decay_factor = decay_factor

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
        The weight data

        """
        return (mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype),  # mean
                mx.nd.zeros(weight.shape, weight.context, dtype=weight.dtype))  # variance

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
        An unique integer key used to index the parameters

        weight : NDArray
        weight ndarray

        grad : NDArray
        grad ndarray

        state : NDArray or other objects returned by init_state
        The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        self._update_count(index)

        t = self._index_update_count[index]
        mean, variance = state
        grad *= self.rescale_grad
        if self.clip_gradient is not None:
            mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient, out=grad)

        mean *= self.beta1
        mean += grad * (1. - self.beta1)

        variance *= self.beta2
        variance += (1 - self.beta2) * mx.nd.square(grad, out=grad)

        coef1 = 1. - self.beta1**t
        coef2 = 1. - self.beta2**t
        lr *= math.sqrt(coef2)/coef1

        weight -= lr*mean/(mx.nd.sqrt(variance) + self.epsilon)

        wd = self._get_wd(index)
        if wd > 0.:
            weight[:] -= (lr * wd) * weight


def test_adam():
    mx.random.seed(0)
    opt1 = PyAdam
    opt2 = mx.optimizer.Adam
    shape = (3, 4, 5)
    kwargs = [{},
              {'clip_gradient': 0.5},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14},
              {'rescale_grad': 0.8},
              {'clip_gradient': 0.5, 'wd': 0.07},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14, 'wd': 0.03},
              {'rescale_grad': 0.8, 'wd': 0.05}]
    for kwarg in kwargs:
        compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape)

# RMSProp

class PyRMSProp(mx.optimizer.Optimizer):
    """RMSProp optimizer of Tieleman & Hinton, 2012,

    This code follows the version in  http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45)
    by Alex Graves, 2013.

    Parameters
    ----------
    learning_rate : float, optional
        Step size.
        Default value is set to 0.002.
    gamma1: float, optional
        decay factor of moving average for gradient, gradient^2.
        Default value is set to 0.95.
    gamma2: float, optional
        "momentum" factor.
        Default value if set to 0.9.
    wd : float, optional
        L2 regularization coefficient add to all the weights
    rescale_grad : float, optional
        rescaling factor of gradient.
    clip_gradient : float, optional
        clip gradient in range [-clip_gradient, clip_gradient]
    """
    def __init__(self, learning_rate=0.001, gamma1=0.95, gamma2=0.9,
                 epsilon=1e-8, **kwargs):
        super(PyRMSProp, self).__init__(learning_rate=learning_rate, **kwargs)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.epsilon = epsilon

    def create_state(self, index, weight):
        """Create additional optimizer state: mean, variance

        Parameters
        ----------
        weight : NDArray
            The weight data

        """
        return (mx.nd.zeros(weight.shape, weight.context),  # n
                mx.nd.zeros(weight.shape, weight.context),  # g
                mx.nd.zeros(weight.shape, weight.context))  # delta

    def update(self, index, weight, grad, state):
        """Update the parameters.

        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters

        weight : NDArray
            weight ndarray

        grad : NDArray
            grad ndarray

        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)
        n, g, delta = state
        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)
        n[:] = (1 - self.gamma1) * (grad * grad) + self.gamma1 * n
        g[:] = (1 - self.gamma1) * grad + self.gamma1 * g
        delta[:] = (self.gamma2) * delta - lr * (grad/(mx.nd.sqrt(n - g*g) + self.epsilon) + wd * weight)
        weight[:] += delta

def test_rms():
    mx.random.seed(0)
    opt1 = PyRMSProp
    opt2 = mx.optimizer.RMSProp
    shape = (3, 4, 5)
    kwargs = [{},
              {'clip_gradient': 0.5},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14},
              {'rescale_grad': 0.8},
              {'clip_gradient': 0.5, 'wd': 0.07},
              {'clip_gradient': 0.4, 'rescale_grad': 0.14, 'wd': 0.03},
              {'rescale_grad': 0.8, 'wd': 0.05}]
    for kwarg in kwargs:
        compare_optimizer(opt1(**kwarg), opt2(**kwarg), shape)

if __name__ == '__main__':
    test_adam()
    test_rms()

from .. import symbol
from .. import ndarray
from ..base import numeric_types, string_types

class RNNParams(object):
    def __init__(self):
        self._params = {}

    def get(self, name, **kwargs):
        if name not in self._params:
            self._params[name] = symbol.Variable(name, **kwargs)
        return self._params[name]


class BaseRNNCell(object):
    """Abstract base class for RNN cells"""
    def __call__(self, inputs, states, params, prefix=''):
        """construct symbol"""
        raise NotImplementedError()

    @property
    def state_shape(self):
        """shape(s) of states"""
        raise NotImplementedError()

    @property
    def output_shape(self):
        """shape(s) of output"""
        raise NotImplementedError()

    def begin_state(self, prefix='', init_sym=symbol.zeros, **kwargs):
        """initial state"""
        state_shape = self.state_shape
        if state_shape is None:
            return init_sym(name=prefix+'begin_state', **kwargs)

        assert isinstance(state_shape, (list, tuple))
        if not len(state_shape):
            return []
        if isinstance(state_shape[0], numeric_types):
            return init_sym(name=prefix+'begin_state', shape=state_shape, **kwargs)
        states = []
        for i, shape in enumerate(state_shape):
            states.append(init_sym(name='%sinit_state_%d'%(prefix, i), shape=shape, **kwargs))
        return states

    def _get_activation(self, x, activation):
        if isinstance(activation, string_types):
            return symbol.Activation(x, act_type=activation)
        else:
            return activation(x)

class RNNCell(BaseRNNCell):
    """Simple recurrent neural network cell"""
    def __init__(self, num_hidden, activation='tanh'):
        self._num_hidden = num_hidden
        self._activation = activation

    @property
    def state_shape(self):
        return (0, self._num_hidden)

    @property
    def output_shape(self):
        return (0, self._num_hidden)

    def __call__(self, inputs, states, params, prefix=''):
        W = params.get('%si2h_weight'%prefix)
        B = params.get('%si2h_bias'%prefix)
        U = params.get('%sh2h_weight'%prefix)
        i2h = symbol.FullyConnected(data=inputs, weight=W, bias=B,
                                    num_hidden=self._num_hidden,
                                    name='%si2h'%prefix)
        h2h = symbol.FullyConnected(data=states, weight=U, no_bias=True,
                                    num_hidden=self._num_hidden,
                                    name='%sh2h'%prefix)
        output = self._get_activation(i2h + h2h, self._activation)
        return output, output













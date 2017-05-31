import mxnet.ndarray as _nd
import mxnet.symbol as _sym

from .layer import Layer
from .parameter import Parameter


class _LegacyLayer(Layer):
    _default_initializers = {
        'moving_mean': 'ones',
        'moving_var': 'zeros',
    }

    def __init__(self, operator_name, prefix=None, params=None, **kwargs):
        """ 
        Parameters
        ----------
        prefix : Please refer to the documentation of `mxnet.contrib.nn.layer.Layer`.
        params : Please refer to the documentation of `mxnet.contrib.nn.layer.Layer`.
        init : dict
            `init` maps the name of a(an) (auxiliary) parameter to a initializer.
            If the initializer of a parameter is not specified, it will be initialized
            by the global initializer.
            Only necessary for imperative call.
        shapes : tuple
            Shapes of variables (inputs).
        variables : tuple
            Specify variables. These variables should be provided when you call `forward`.
        """

        self._operator_name = operator_name

        super(_LegacyLayer, self).__init__(prefix, params)

        init_dict = dict(self._default_initializers)
        init_dict.update(kwargs.pop('init', {}))
        shapes = kwargs.pop('shape', tuple())
        self._variables = kwargs.pop('variables', ('data',))

        self._kwargs = dict(kwargs)

        kwargs.update({v: _sym.Variable(v) for v in self._variables})

        # another symbol will be created if symbolic_forward is called
        symbol = getattr(_sym, operator_name)(name=self._prefix, **kwargs)
        self._operator = getattr(_nd, operator_name)

        eliminate_prefix = lambda name: name.replace(self._prefix + '_', '')

        # unspecified shape does not cause error in symbolic call
        shape_dict = dict(zip(self._variables, shapes))

        arg_shapes, _, aux_shapes = symbol.infer_shape_partial(**shape_dict)

        condense = lambda shape: tuple(d for d in shape if d != 0)
        arg_shapes = tuple(map(condense, arg_shapes))
        aux_shapes = tuple(map(condense, aux_shapes))

        params = symbol.list_arguments()
        params = tuple(map(eliminate_prefix, tuple(params)))

        for param, shape in zip(params, arg_shapes):
            if param not in self._variables:
                init = init_dict.get(param)
                self._reg_params[param] = \
                    self.params.get(param, grad_req='write', shape=shape, init=init)

        aux_params = symbol.list_auxiliary_states()
        aux_params = tuple(map(eliminate_prefix, tuple(aux_params)))

        for aux_param, shape in zip(aux_params, aux_shapes):
            if aux_param not in self._variables:
                init = init_dict.get(aux_param) 
                self._reg_params[aux_param] = \
                    self.params.get(aux_param, grad_req='null', shape=shape, init=init)

    def _alias(self):
        return self._operator_name.lower()

    def ndarray_forward(self, *args, **kwargs):
        """ Imperative call.

        Parameters
        ----------
        args : `NDArray` inputs. The order of `args` should be identical to that of `variable`,
            which is specified by calling `__init__`.
        kwargs : Please do not provide any key-word argument.
            `kwargs` is only for implementation purpose.
        """
        kwargs.update(dict(zip(self._variables, args)))
        kwargs.update(self._kwargs)
        return self._operator(**kwargs)

    def symbolic_forward(self, *args, **kwargs):
        """ Symbolic call.

        Parameters
        ----------
        args : `Symbol` inputs. The order of `args` should be identical to that of `variable`,
            which is specified by calling `__init__`.
        kwargs : Please do not provide any key-word argument.
            `kwargs` is only for implementation purpose.
        """
        kwargs.update(dict(zip(self._variables, args)))
        kwargs.update(self._kwargs)
        return getattr(_sym, self._operator_name)(name=self._prefix, **kwargs)


# pylint: disable=locally-disabled, invalid-name
Activation = lambda **kwargs: _LegacyLayer('Activation', **kwargs)
BatchNorm = lambda **kwargs: _LegacyLayer('BatchNorm', **kwargs)
Convolution = lambda **kwargs: _LegacyLayer('Convolution', **kwargs)
Deconvolution = lambda **kwargs: _LegacyLayer('Deconvolution', **kwargs)
Dropout = lambda **kwargs: _LegacyLayer('Dropout', **kwargs)
Embedding = lambda **kwargs: _LegacyLayer('Embedding', **kwargs)
FullyConnected = lambda **kwargs: _LegacyLayer('FullyConnected', **kwargs)
LeakyReLU = lambda **kwargs: _LegacyLayer('LeakyReLU', **kwargs)
Pooling = lambda **kwargs: _LegacyLayer('Pooling', **kwargs)
RNN = lambda **kwargs: _LegacyLayer('RNN', **kwargs)

# pylint: disable=locally-disabled, invalid-name
ReLU = lambda: _LegacyLayer('Activation', act_type='relu')
Sigmoid = lambda: _LegacyLayer('Activation', act_type='sigmoid')
Tanh = lambda: _LegacyLayer('Activation', act_type='tanh')

# pylint: disable=locally-disabled, invalid-name
RNNReLU = lambda **kwargs: _LegacyLayer('RNN', mode='rnn_relu', **kwargs)
RNNTanh = lambda **kwargs: _LegacyLayer('RNN', mode='rnn_tanh', **kwargs)
GRU = lambda **kwargs: _LegacyLayer('RNN', mode='gru', **kwargs)
LSTM = lambda **kwargs: _LegacyLayer('RNN', mode='lstm', **kwargs)

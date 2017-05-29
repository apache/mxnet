import mxnet.ndarray as _nd
import mxnet.symbol as _sym

from .layer import Layer
from .parameter import Parameter


class _LegacyLayer(Layer):
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

        super(_LegacyLayer, self).__init__(prefix, params)

        init = kwargs.pop('init', {})
        shapes = kwargs.pop('shape', tuple())
        self._variables = kwargs.pop('variables', ('data',))

        self._kwargs = dict(kwargs)

        kwargs.update({v: _sym.Variable(v) for v in self._variables})

        self._operator_name = operator_name

        # another symbol will be created if symbolic_forward is called
        symbol = getattr(_sym, operator_name)(name=self._prefix, **kwargs)
        self._operator = getattr(_nd, operator_name)

        eliminate_prefix = lambda name: name.replace(self._prefix + '_', '')

        # unspecified shape does not cause error in symbolic call
        arg_shapes, _, aux_shapes = symbol.infer_shape(*shapes)

        params = set(symbol.list_arguments())
        params = tuple(map(eliminate_prefix, tuple(params)))
        params = params.difference(set(self._variables))

        for param, shape in zip(params, arg_shapes):
            if param not in variables:
                init = init_dict.get(param)
                self._reg_params[param] = \
                    self.params.get(param, grad_req='write', shape=shape, init=init)

        aux_params = symbol.list_auxiliary_states()
        aux_params = tuple(map(eliminate_prefix, tuple(aux_params)))
        aux_params = aux_params.difference(set(self._variables))

        for aux_param, shape in zip(aux_params, aux_param_shapes):
            if aux_param not in variables:
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
        kwargs.update(dict(zip(self._variables, args))
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
        kwargs.update(dict(zip(self._variables, args))
        kwargs.update(self._kwargs)
        return getattr(_sym, self._operator_name)(name=self._prefix, **kwargs)


# pylint: disable=locally-disabled, invalid-name
Activation = lambda **kwargs: _Operatorized('Activation', **kwargs)
BatchNorm = lambda **kwargs: _Operatorized('BatchNorm', **kwargs)
Convolution = lambda **kwargs: _Operatorized('Convolution', **kwargs)
Deconvolution = lambda **kwargs: _Operatorized('Deconvolution', **kwargs)
Dropout = lambda **kwargs: _Operatorized('Dropout', **kwargs)
Embedding = lambda **kwargs: _Operatorized('Embedding', **kwargs)
FullyConnected = lambda **kwargs: _Operatorized('FullyConnected', **kwargs)
LeakyReLU = lambda **kwargs: _Operatorized('LeakyReLU', **kwargs)
Pooling = lambda **kwargs: _Operatorized('Pooling', **kwargs)
RNN = lambda **kwargs: _Operatorized('RNN', **kwargs)

# pylint: disable=locally-disabled, invalid-name
ReLU = lambda: _Operatorized('Activation', act_type='relu')
Sigmoid = lambda: _Operatorized('Activation', act_type='sigmoid')
Tanh = lambda: _Operatorized('Activation', act_type='tanh')

# pylint: disable=locally-disabled, invalid-name
RNNReLU = lambda **kwargs: _Operatorized('RNN', mode='rnn_relu', **kwargs)
RNNTanh = lambda **kwargs: _Operatorized('RNN', mode='rnn_tanh', **kwargs)
GRU = lambda **kwargs: _Operatorized('RNN', mode='gru', **kwargs)
LSTM = lambda **kwargs: _Operatorized('RNN', mode='lstm', **kwargs)

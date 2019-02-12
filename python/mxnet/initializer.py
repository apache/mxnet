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

"""Weight initializer."""
from __future__ import absolute_import, print_function

import re
import logging
import warnings
import json
from math import sqrt
import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
from . import registry
from . import ndarray

# inherit str for backward compatibility
class InitDesc(str):
    """
    Descriptor for the initialization pattern.

    Parameters
    ----------
    name : str
        Name of variable.
    attrs : dict of str to str
        Attributes of this variable taken from ``Symbol.attr_dict``.
    global_init : Initializer
        Global initializer to fallback to.
    """
    def __new__(cls, name, attrs=None, global_init=None):
        ret = super(InitDesc, cls).__new__(cls, name)
        ret.attrs = attrs or {}
        ret.global_init = global_init
        return ret


class Initializer(object):
    """The base class of an initializer."""
    def __init__(self, **kwargs):
        self._kwargs = kwargs
        self._verbose = False
        self._print_func = None

    def set_verbosity(self, verbose=False, print_func=None):
        """Switch on/off verbose mode

        Parameters
        ----------
        verbose : bool
            switch on/off verbose mode
        print_func : function
            A function that computes statistics of initialized arrays.
            Takes an `NDArray` and returns an `str`. Defaults to mean
            absolute value str((abs(x)/size(x)).asscalar()).
        """
        self._verbose = verbose
        if print_func is None:
            def asum_stat(x):
                """returns |x|/size(x), async execution."""
                return str((ndarray.norm(x)/sqrt(x.size)).asscalar())
            print_func = asum_stat
        self._print_func = print_func
        return self

    def _verbose_print(self, desc, init, arr):
        """Internal verbose print function

        Parameters
        ----------
        desc : InitDesc or str
            name of the array
        init : str
            initializer pattern
        arr : NDArray
            initialized array
        """
        if self._verbose and self._print_func:
            logging.info('Initialized %s as %s: %s', desc, init, self._print_func(arr))

    def dumps(self):
        """Saves the initializer to string

        Returns
        -------
        str
            JSON formatted string that describes the initializer.

        Examples
        --------
        >>> # Create initializer and retrieve its parameters
        ...
        >>> init = mx.init.Normal(0.5)
        >>> init.dumps()
        '["normal", {"sigma": 0.5}]'
        >>> init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        >>> init.dumps()
        '["xavier", {"rnd_type": "uniform", "magnitude": 2.34, "factor_type": "in"}]'
        """
        return json.dumps([self.__class__.__name__.lower(), self._kwargs])

    def __call__(self, desc, arr):
        """Initialize an array

        Parameters
        ----------
        desc : InitDesc
            Initialization pattern descriptor.

        arr : NDArray
            The array to be initialized.
        """
        if not isinstance(desc, InitDesc):
            self._legacy_init(desc, arr)
            return

        if desc.global_init is None:
            desc.global_init = self
        init = desc.attrs.get('__init__', "")

        if init:
            # when calling Variable initializer
            create(init)._init_weight(desc, arr)
            self._verbose_print(desc, init, arr)
        else:
            # register nnvm::FSetInputVariableAttrs in the backend for new patterns
            # don't add new cases here.
            if desc.endswith('weight'):
                self._init_weight(desc, arr)
                self._verbose_print(desc, 'weight', arr)
            elif desc.endswith('bias'):
                self._init_bias(desc, arr)
                self._verbose_print(desc, 'bias', arr)
            elif desc.endswith('gamma'):
                self._init_gamma(desc, arr)
                self._verbose_print(desc, 'gamma', arr)
            elif desc.endswith('beta'):
                self._init_beta(desc, arr)
                self._verbose_print(desc, 'beta', arr)
            elif desc.endswith('min'):
                self._init_zero(desc, arr)
                self._verbose_print(desc, 'min', arr)
            elif desc.endswith('max'):
                self._init_one(desc, arr)
                self._verbose_print(desc, 'max', arr)
            elif desc.endswith('weight_quantize'):
                self._init_quantized_weight(desc, arr)
                self._verbose_print(desc, 'weight_quantize', arr)
            elif desc.endswith('bias_quantize'):
                self._init_quantized_bias(desc, arr)
                self._verbose_print(desc, 'bias_quantize', arr)
            else:
                self._init_default(desc, arr)

    def _legacy_init(self, name, arr):
        """Legacy initialization method.

        Parameters
        ----------
        name : str
            Name of corresponding NDArray.

        arr : NDArray
            NDArray to be initialized.
        """
        warnings.warn(
            "\033[91mCalling initializer with init(str, NDArray) has been deprecated." \
            "please use init(mx.init.InitDesc(...), NDArray) instead.\033[0m",
            DeprecationWarning, stacklevel=3)
        if not isinstance(name, string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, NDArray):
            raise TypeError('arr must be NDArray')
        if name.startswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.startswith('stn_loc') and name.endswith('weight'):
            self._init_zero(name, arr)
        elif name.startswith('stn_loc') and name.endswith('bias'):
            self._init_loc_bias(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_one(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        elif name.endswith('min'):
            self._init_zero(name, arr)
        elif name.endswith('max'):
            self._init_one(name, arr)
        else:
            self._init_default(name, arr)

    def _init_bilinear(self, _, arr):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i // shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        arr[:] = weight.reshape(shape)

    def _init_loc_bias(self, _, arr):
        shape = arr.shape
        assert(shape[0] == 6)
        arr[:] = np.array([1.0, 0, 0, 0, 1.0, 0])

    def _init_zero(self, _, arr):
        arr[:] = 0.0

    def _init_one(self, _, arr):
        arr[:] = 1.0

    def _init_bias(self, _, arr):
        arr[:] = 0.0

    def _init_quantized_bias(self, _, arr):
        arr[:] = 0

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_weight(self, name, arr):
        """Abstract method to Initialize weight."""
        raise NotImplementedError("Must override it")

    def _init_quantized_weight(self, _, arr):
        _arr = random.randint(-127, 127, dtype='int32').asnumpy()
        arr[:] = np.int8(_arr)

    def _init_default(self, name, _):
        raise ValueError(
            'Unknown initialization pattern for %s. ' \
            'Default initialization is now limited to '\
            '"weight", "bias", "gamma" (1.0), and "beta" (0.0).' \
            'Please use mx.sym.Variable(init=mx.init.*) to set initialization pattern' % name)


# pylint: disable=invalid-name
_register = registry.get_register_func(Initializer, 'initializer')
alias = registry.get_alias_func(Initializer, 'initializer')
create = registry.get_create_func(Initializer, 'initializer')
# pylint: enable=invalid-name

def register(klass):
    """Registers a custom initializer.

    Custom initializers can be created by extending `mx.init.Initializer` and implementing the
    required functions like `_init_weight` and `_init_bias`. The created initializer must be
    registered using `mx.init.register` before it can be called by name.

    Parameters
    ----------
    klass : class
        A subclass of `mx.init.Initializer` that needs to be registered as a custom initializer.

    Example
    -------
    >>> # Create and register a custom initializer that
    ... # initializes weights to 0.1 and biases to 1.
    ...
    >>> @mx.init.register
    ... @alias('myinit')
    ... class CustomInit(mx.init.Initializer):
    ...   def __init__(self):
    ...     super(CustomInit, self).__init__()
    ...   def _init_weight(self, _, arr):
    ...     arr[:] = 0.1
    ...   def _init_bias(self, _, arr):
    ...     arr[:] = 1
    ...
    >>> # Module is an instance of 'mxnet.module.Module'
    ...
    >>> module.init_params("custominit")
    >>> # module.init_params("myinit")
    >>> # module.init_params(CustomInit())
    """
    return _register(klass)


class Load(object):
    """Initializes variables by loading data from file or dict.

    **Note** Load will drop ``arg:`` or ``aux:`` from name and
    initialize the variables that match with the prefix dropped.

    Parameters
    ----------
    param: str or dict of str->`NDArray`
        Parameter file or dict mapping name to NDArray.
    default_init: Initializer
        Default initializer when name is not found in `param`.
    verbose: bool
        Flag for enabling logging of source when initializing.

    """
    def __init__(self, param, default_init=None, verbose=False):
        if isinstance(param, str):
            param = load(param)
        assert isinstance(param, dict)
        self.param = {}
        for name, arr in param.items():
            if name.startswith('arg:') or name.startswith('aux:'):
                self.param[name[4:]] = arr
            else:
                self.param[name] = arr
        self.default_init = default_init
        self.verbose = verbose

    def __call__(self, name, arr):
        if name in self.param:
            assert arr.shape == self.param[name].shape, \
                'Parameter %s cannot be initialized from loading. '%name + \
                'Shape mismatch, target %s vs loaded %s'%(str(arr.shape),
                                                          self.param[name].shape)
            arr[:] = self.param[name]
            if self.verbose:
                logging.info('Initialized %s by loading', name)
        else:
            assert self.default_init is not None, \
                "Cannot Initialize %s. Not found in loaded param "%name + \
                "and no default Initializer is provided."
            self.default_init(name, arr)
            if self.verbose:
                logging.info('Initialized %s by default', name)


class Mixed(object):
    """Initialize parameters using multiple initializers.

    Parameters
    ----------
    patterns: list of str
        List of regular expressions matching parameter names.
    initializers: list of Initializer
        List of initializers corresponding to `patterns`.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize biases to zero
    ... # and every other parameter to random values with uniform distribution.
    ...
    >>> init = mx.initializer.Mixed(['bias', '.*'], [mx.init.Zero(), mx.init.Uniform(0.1)])
    >>> module.init_params(init)
    >>>
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected1_weight
    [[ 0.0097627   0.01856892  0.04303787]]
    fullyconnected1_bias
    [ 0.]

    """
    def __init__(self, patterns, initializers):
        assert len(patterns) == len(initializers)
        self.map = list(zip([re.compile(p) for p in patterns], initializers))

    def __call__(self, name, arr):
        for prog, init in self.map:
            if prog.match(name):
                init(name, arr)
                return
        raise ValueError('Parameter name %s did not match any pattern. Consider' +
                         'add a ".*" pattern at the and with default Initializer.')

@register
@alias("zeros")
class Zero(Initializer):
    """Initializes weights to zero.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize weights to zero.
    ...
    >>> init = mx.initializer.Zero()
    >>> module.init_params(init)
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected0_weight
    [[ 0.  0.  0.]]
    """
    def __init__(self):
        super(Zero, self).__init__()

    def _init_weight(self, _, arr):
        arr[:] = 0

@register
@alias("ones")
class One(Initializer):
    """Initializes weights to one.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize weights to one.
    ...
    >>> init = mx.initializer.One()
    >>> module.init_params(init)
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected0_weight
    [[ 1.  1.  1.]]
    """
    def __init__(self):
        super(One, self).__init__()

    def _init_weight(self, _, arr):
        arr[:] = 1

@register
class Constant(Initializer):
    """Initializes the weights to a given value.
    The value passed in can be a scalar or a NDarray that matches the shape
    of the parameter to be set.

    Parameters
    ----------
    value : float, NDArray
        Value to set.
    """
    def __init__(self, value):
        super(Constant, self).__init__(value=value)
        self.value = value

    def _init_weight(self, _, arr):
        arr[:] = self.value

@register
class Uniform(Initializer):
    """Initializes weights with random values uniformly sampled from a given range.

    Parameters
    ----------
    scale : float, optional
        The bound on the range of the generated random values.
        Values are generated from the range [-`scale`, `scale`].
        Default scale is 0.07.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize weights
    >>> # to random values uniformly sampled between -0.1 and 0.1.
    ...
    >>> init = mx.init.Uniform(0.1)
    >>> module.init_params(init)
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected0_weight
    [[ 0.01360891 -0.02144304  0.08511933]]
    """
    def __init__(self, scale=0.07):
        super(Uniform, self).__init__(scale=scale)
        self.scale = scale

    def _init_weight(self, _, arr):
        random.uniform(-self.scale, self.scale, out=arr)

@register
class Normal(Initializer):
    """Initializes weights with random values sampled from a normal distribution
    with a mean of zero and standard deviation of `sigma`.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the normal distribution.
        Default standard deviation is 0.01.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize weights
    >>> # to random values sampled from a normal distribution.
    ...
    >>> init = mx.init.Normal(0.5)
    >>> module.init_params(init)
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected0_weight
    [[-0.3214761  -0.12660924  0.53789419]]
    """
    def __init__(self, sigma=0.01):
        super(Normal, self).__init__(sigma=sigma)
        self.sigma = sigma

    def _init_weight(self, _, arr):
        random.normal(0, self.sigma, out=arr)

@register
class Orthogonal(Initializer):
    """Initialize weight as orthogonal matrix.

    This initializer implements *Exact solutions to the nonlinear dynamics of
    learning in deep linear neural networks*, available at
    https://arxiv.org/abs/1312.6120.

    Parameters
    ----------
    scale : float optional
        Scaling factor of weight.

    rand_type: string optional
        Use "uniform" or "normal" random number to initialize weight.

    """
    def __init__(self, scale=1.414, rand_type="uniform"):
        super(Orthogonal, self).__init__(scale=scale, rand_type=rand_type)
        self.scale = scale
        self.rand_type = rand_type

    def _init_weight(self, _, arr):
        nout = arr.shape[0]
        nin = np.prod(arr.shape[1:])
        if self.rand_type == "uniform":
            tmp = random.uniform(-1.0, 1.0, shape=(nout, nin)).asnumpy()
        elif self.rand_type == "normal":
            tmp = random.normal(0.0, 1.0, shape=(nout, nin)).asnumpy()
        u, _, v = np.linalg.svd(tmp, full_matrices=False) # pylint: disable=invalid-name
        if u.shape == tmp.shape:
            res = u
        else:
            res = v
        res = self.scale * res.reshape(arr.shape)
        arr[:] = res

@register
class Xavier(Initializer):
    """Returns an initializer performing "Xavier" initialization for weights.

    This initializer is designed to keep the scale of gradients roughly the same
    in all layers.

    By default, `rnd_type` is ``'uniform'`` and `factor_type` is ``'avg'``,
    the initializer fills the weights with random numbers in the range
    of :math:`[-c, c]`, where :math:`c = \\sqrt{\\frac{3.}{0.5 * (n_{in} + n_{out})}}`.
    :math:`n_{in}` is the number of neurons feeding into weights, and :math:`n_{out}` is
    the number of neurons the result is fed to.

    If `rnd_type` is ``'uniform'`` and `factor_type` is ``'in'``,
    the :math:`c = \\sqrt{\\frac{3.}{n_{in}}}`.
    Similarly when `factor_type` is ``'out'``, the :math:`c = \\sqrt{\\frac{3.}{n_{out}}}`.

    If `rnd_type` is ``'gaussian'`` and `factor_type` is ``'avg'``,
    the initializer fills the weights with numbers from normal distribution with
    a standard deviation of :math:`\\sqrt{\\frac{3.}{0.5 * (n_{in} + n_{out})}}`.

    Parameters
    ----------
    rnd_type: str, optional
        Random generator type, can be ``'gaussian'`` or ``'uniform'``.

    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    magnitude: float, optional
        Scale of random number.
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        super(Xavier, self).__init__(rnd_type=rnd_type, factor_type=factor_type,
                                     magnitude=magnitude)
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)


    def _init_weight(self, name, arr):
        shape = arr.shape
        hw_scale = 1.
        if len(shape) < 2:
            raise ValueError('Xavier initializer cannot be applied to vector {0}. It requires at'
                             ' least 2D.'.format(name))
        if len(shape) > 2:
            hw_scale = np.prod(shape[2:])
        fan_in, fan_out = shape[1] * hw_scale, shape[0] * hw_scale
        factor = 1.
        if self.factor_type == "avg":
            factor = (fan_in + fan_out) / 2.0
        elif self.factor_type == "in":
            factor = fan_in
        elif self.factor_type == "out":
            factor = fan_out
        else:
            raise ValueError("Incorrect factor type")
        scale = np.sqrt(self.magnitude / factor)
        if self.rnd_type == "uniform":
            random.uniform(-scale, scale, out=arr)
        elif self.rnd_type == "gaussian":
            random.normal(0, scale, out=arr)
        else:
            raise ValueError("Unknown random type")

@register
class MSRAPrelu(Xavier):
    """Initialize the weight according to a MSRA paper.

    This initializer implements *Delving Deep into Rectifiers: Surpassing
    Human-Level Performance on ImageNet Classification*, available at
    https://arxiv.org/abs/1502.01852.

    This initializer is proposed for initialization related to ReLu activation,
    it maked some changes on top of Xavier method.

    Parameters
    ----------
    factor_type: str, optional
        Can be ``'avg'``, ``'in'``, or ``'out'``.

    slope: float, optional
        initial slope of any PReLU (or similar) nonlinearities.
    """
    def __init__(self, factor_type="avg", slope=0.25):
        magnitude = 2. / (1 + slope ** 2)
        super(MSRAPrelu, self).__init__("gaussian", factor_type, magnitude)
        self._kwargs = {'factor_type': factor_type, 'slope': slope}

@register
class Bilinear(Initializer):
    """Initialize weight for upsampling layers."""
    def __init__(self):
        super(Bilinear, self).__init__()

    def _init_weight(self, _, arr):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i // shape[3]) % shape[2]
            weight[i] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        arr[:] = weight.reshape(shape)


@register
class LSTMBias(Initializer):
    """Initialize all biases of an LSTMCell to 0.0 except for
    the forget gate whose bias is set to custom value.

    Parameters
    ----------
    forget_bias: float, default 1.0
        bias for the forget gate. Jozefowicz et al. 2015 recommends
        setting this to 1.0.
    """
    def __init__(self, forget_bias=1.0):
        super(LSTMBias, self).__init__(forget_bias=forget_bias)
        self.forget_bias = forget_bias

    def _init_weight(self, name, arr):
        arr[:] = 0.0
        # in the case of LSTMCell the forget gate is the second
        # gate of the 4 LSTM gates, we modify the according values.
        num_hidden = int(arr.shape[0] / 4)
        arr[num_hidden:2*num_hidden] = self.forget_bias


@register
class FusedRNN(Initializer):
    """Initialize parameters for fused rnn layers.

    Parameters
    ----------
    init : Initializer
        initializer applied to unpacked weights. Fall back to global
        initializer if None.
    num_hidden : int
        should be the same with arguments passed to FusedRNNCell.
    num_layers : int
        should be the same with arguments passed to FusedRNNCell.
    mode : str
        should be the same with arguments passed to FusedRNNCell.
    bidirectional : bool
        should be the same with arguments passed to FusedRNNCell.
    forget_bias : float
        should be the same with arguments passed to FusedRNNCell.
    """
    def __init__(self, init, num_hidden, num_layers, mode, bidirectional=False, forget_bias=1.0):
        if isinstance(init, string_types):
            klass, kwargs = json.loads(init)
            init = registry._REGISTRY[klass.lower()](**kwargs)
        super(FusedRNN, self).__init__(init=init.dumps() if init is not None else None,
                                       num_hidden=num_hidden, num_layers=num_layers, mode=mode,
                                       bidirectional=bidirectional, forget_bias=forget_bias)
        self._init = init
        self._num_hidden = num_hidden
        self._num_layers = num_layers
        self._mode = mode
        self._bidirectional = bidirectional
        self._forget_bias = forget_bias

    def _init_weight(self, desc, arr): # pylint: disable=arguments-differ
        from .rnn import rnn_cell
        cell = rnn_cell.FusedRNNCell(self._num_hidden, self._num_layers,
                                     self._mode, self._bidirectional,
                                     forget_bias=self._forget_bias, prefix='')
        args = cell.unpack_weights({'parameters': arr})
        for name in args:
            arg_desc = InitDesc(name, global_init=desc.global_init)
            # for lstm bias, we use a custom initializer
            # which adds a bias to the forget gate
            if self._mode == 'lstm' and name.endswith("_f_bias"):
                args[name][:] = self._forget_bias
            elif self._init is None:
                desc.global_init(arg_desc, args[name])
            else:
                self._init(arg_desc, args[name])

        arr[:] = cell.pack_weights(args)['parameters']

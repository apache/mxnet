# coding: utf-8
# pylint: disable=too-many-branches
"""Initialization helper for mxnet"""
from __future__ import absolute_import

import numpy as np
from .base import string_types
from .ndarray import NDArray, load
from . import random
import logging
import re

class Initializer(object):
    """Base class for Initializer."""

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
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
        else:
            self._init_default(name, arr)
    # pylint: disable=no-self-use, missing-docstring, invalid-name
    def _init_bilinear(self, _, arr):
        weight = np.zeros(np.prod(arr.shape), dtype='float32')
        shape = arr.shape
        f = np.ceil(shape[3] / 2.)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(np.prod(shape)):
            x = i % shape[3]
            y = (i / shape[3]) % shape[2]
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

    def _init_gamma(self, _, arr):
        arr[:] = 1.0

    def _init_beta(self, _, arr):
        arr[:] = 0.0

    def _init_weight(self, name, arr):
        """Abstruct method to Initialize weight"""
        raise NotImplementedError("Must override it")

    def _init_default(self, name, _):
        raise ValueError('Unknown initialization pattern for %s' % name)
    # pylint: enable=no-self-use, missing-docstring, invalid-name


class Load(object):
    """Initialize by loading pretrained param from file or dict

    Parameters
    ----------
    param: str or dict of str->NDArray
        param file or dict mapping name to NDArray.
    default_init: Initializer
        default initializer when name is not found in param.
    verbose: bool
        log source when initializing.
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
    """Initialize with mixed Initializer

    Parameters
    ----------
    patterns: list of str
        list of regular expression patterns to match parameter names.
    initializers: list of Initializer
        list of Initializer corrosponding to patterns
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


class Uniform(Initializer):
    """Initialize the weight with uniform [-scale, scale]

    Parameters
    ----------
    scale : float, optional
        The scale of uniform distribution
    """
    def __init__(self, scale=0.07):
        self.scale = scale

    def _init_weight(self, _, arr):
        random.uniform(-self.scale, self.scale, out=arr)


class Normal(Initializer):
    """Initialize the weight with normal(0, sigma)

    Parameters
    ----------
    sigma : float, optional
        Standard deviation for gaussian distribution.
    """
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def _init_weight(self, _, arr):
        random.normal(0, self.sigma, out=arr)


class Orthogonal(Initializer):
    """Intialize weight as Orthogonal matrix

    Parameters
    ----------
    scale : float optional
        scaling factor of weight

    rand_type: string optional
        use "uniform" or "normal" random number to initialize weight

    Reference
    ---------
    Exact solutions to the nonlinear dynamics of learning in deep linear neural networks
    arXiv preprint arXiv:1312.6120 (2013).
    """
    def __init__(self, scale=1.414, rand_type="uniform"):
        self.scale = scale
        self.rand_type = rand_type

    # pylint: disable=invalid-name
    def _init_weight(self, _, arr):
        nout = arr.shape[0]
        nin = np.prod(arr.shape[1:])
        if self.rand_type == "uniform":
            tmp = np.random.uniform(-1.0, 1.0, (nout, nin))
        elif self.rand_type == "normal":
            tmp = np.random.normal(0.0, 1.0, (nout, nin))
        u, _, v = np.linalg.svd(tmp, full_matrices=False)
        if u.shape == tmp.shape:
            q = u
        else:
            q = v
        q = self.scale * q.reshape(arr.shape)
        arr[:] = q


class Xavier(Initializer):
    """Initialize the weight with Xavier or similar initialization scheme.

    Parameters
    ----------
    rnd_type: str, optional
        Use ```gaussian``` or ```uniform``` to init

    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    magnitude: float, optional
        scale of random number range
    """
    def __init__(self, rnd_type="uniform", factor_type="avg", magnitude=3):
        self.rnd_type = rnd_type
        self.factor_type = factor_type
        self.magnitude = float(magnitude)


    def _init_weight(self, _, arr):
        shape = arr.shape
        hw_scale = 1.
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

class MSRAPrelu(Xavier):
    """Initialize the weight with initialization scheme from
        Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification.

    Parameters
    ----------
    factor_type: str, optional
        Use ```avg```, ```in```, or ```out``` to init

    slope: float, optional
        initial slope of any PReLU (or similar) nonlinearities.
    """
    def __init__(self, factor_type="avg", slope=0.25):
        magnitude = 2. / (1 + slope ** 2)
        super(MSRAPrelu, self).__init__("gaussian", factor_type, magnitude)

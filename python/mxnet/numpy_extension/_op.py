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

"""Namespace for registering numpy_extension ops for imperative programming."""

from ..ndarray import numpy_extension as _mx_nd_npx
from ..util import set_module


__all__ = ['softmax', 'log_softmax', 'masked_softmax', 'masked_log_softmax',
           'activation', 'batch_norm', 'fully_connected']


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    r"""Applies the softmax function.

    The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.

    .. math::
       softmax(\mathbf{z/t})_j = \frac{e^{z_j/t}}{\sum_{k=1}^K e^{z_k/t}}

    for :math:`j = 1, ..., K`

    t is the temperature parameter in softmax function. By default, t equals 1.0

    Parameters
    ----------
    data : NDArray
        The input array.
    length : NDArray
        The length array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax
    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to
        the same as input's dtype if not defined (dtype=None).
    use_length : boolean or None, optional, default=0
        Whether to use the length input as a mask over the data input.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> data = np.ones((2, 3))
    >>> npx.softmax(data, axis=0)
    array([[0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]])
    >>> npx.softmax(data, axis=1)
    array([[0.33333334, 0.33333334, 0.33333334],
        [0.33333334, 0.33333334, 0.33333334]])
    """
    return _mx_nd_npx.softmax(data, length, axis=axis, temperature=temperature,
                              use_length=use_length, dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def log_softmax(data, length=None, axis=-1, temperature=None, use_length=False, dtype=None):
    r"""Computes the log softmax of the input.
    This is equivalent to computing softmax followed by log.

    Parameters
    ----------
    data : NDArray
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax
    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to
        the same as input's dtype if not defined (dtype=None).
    use_length : boolean or None, optional, default=0
        Whether to use the length input as a mask over the data input.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    >>> data = np.array([1, 2, .1])
    >>> npx.log_softmax(data)
    array([-1.4170278, -0.4170278, -2.3170278])
    >>> data = np.array([[1, 2, .1],[.1, 2, 1]])
    >>> npx.log_softmax(data, axis=0)
    array([[-0.34115386, -0.6931472 , -1.2411538 ],
        [-1.2411538 , -0.6931472 , -0.34115386]])
    """
    return _mx_nd_npx.log_softmax(data, length, axis=axis, temperature=temperature,
                                  use_length=use_length, dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def masked_softmax(data, mask, axis=-1, temperature=1.0, dtype=None):
    r"""Applies the softmax function masking elements according to the mask provided

    Parameters
    ----------
    data : NDArray
        The input array.
    mask : NDArray
        Mask to apply.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax
    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to
        the same as input's dtype if not defined (dtype=None).
    normalize : boolean or None, optional, default=1
        Whether to normalize input data x: x = x - max(x)

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    >>> data = np.arange(5)
    >>> mask = np.array([1, 0, 1, 0, 1])
    >>> npx.masked_softmax(data, mask)
    array([0.01587624, 0.        , 0.11731042, 0.        , 0.8668133 ])
    >>> data = np.arange(10).reshape((2, 5))
    >>> npx.masked_softmax(data, mask, axis=0)
    array([[0.00669285, 0.        , 0.00669285, 0.        , 0.00669285],
        [0.9933072 , 0.        , 0.9933072 , 0.        , 0.9933072 ]])
    """
    return _mx_nd_npx.masked_softmax(data, mask, axis=axis, temperature=temperature,
                                     dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def masked_log_softmax(data, mask, axis=-1, temperature=1.0, dtype=None):
    r"""Computes the masked log softmax of the input.
    This is equivalent to computing masked softmax followed by log.

    Parameters
    ----------
    data : NDArray
        The input array.
    mask : NDArray
        Mask to apply.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    temperature : double or None, optional, default=None
        Temperature parameter in softmax
    dtype : {None, 'float16', 'float32', 'float64'},optional, default='None'
        DType of the output in case this can't be inferred. Defaults to
        the same as input's dtype if not defined (dtype=None).
    normalize : boolean or None, optional, default=1
        Whether to normalize input data x: x = x - max(x)

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Examples
    --------
    >>> data = np.arange(5)
    >>> mask = np.array([1, 0, 1, 0, 1])
    >>> npx.masked_log_softmax(data, mask)
    array([-4.1429286 ,        -inf, -2.1429286 ,        -inf, -0.14292854])
    >>> data = np.arange(10).reshape((2, 5))
    >>> npx.masked_log_softmax(data, mask, axis=0)
    array([[-5.0067153 ,        -inf, -5.0067153 ,        -inf, -5.0067153 ],
       [-0.00671535,        -inf, -0.00671535,        -inf, -0.00671535]])
    """
    return _mx_nd_npx.masked_log_softmax(data, mask, axis=axis, temperature=temperature,
                                         dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def activation(data, act_type='relu', name=None):
    r"""Applies an activation function element-wise to the input.

    The following activation functions are supported:

    - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
    - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
    - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
    - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
    - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`

    Parameters
    ----------
    data : NDArray
        The input array.
    act_type : {'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required
        Activation function to be applied.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.activation(data, act_type=act_type, name=name)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def batch_norm(x, gamma, beta, running_mean, running_var, name=None, eps=1e-3, momentum=0.9,
               fix_gamma=True, use_global_stats=False, output_mean_var=False, axis=1,
               cudnn_off=False, min_calib_range=None, max_calib_range=None):
    r"""Batch normalization.

    Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis:

    .. math::

      data\_mean[i] = mean(data[:,i,:,...]) \\
      data\_var[i] = var(data[:,i,:,...])

    Then compute the normalized output, which has the same shape as input, as following:

    .. math::

      out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]

    Both *mean* and *var* returns a scalar by treating the input as a vector.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these
    two outputs are blocked.

    Besides the inputs and the outputs, this operator accepts two auxiliary
    states, ``moving_mean`` and ``moving_var``, which are *k*-length
    vectors. They are global statistics for the whole dataset, which are updated
    by::

      moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
      moving_var = moving_var * momentum + data_var * (1 - momentum)

    If ``use_global_stats`` is set to be true, then ``moving_mean`` and
    ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
    the output. It is often used during inference.

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
    axis to be the last item in the input shape.

    Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
    then set ``gamma`` to 1 and its gradient to 0.

    .. Note::
      When ``fix_gamma`` is set to True, no sparse support is provided. If ``fix_gamma is`` set to False,
      the sparse tensors will fallback.

    Parameters
    ----------
    data : NDArray
        Input data to batch normalization
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    moving_mean : NDArray
        running mean of input
    moving_var : NDArray
        running variance of input
    eps : double, optional, default=0.0010000000474974513
        Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON
        defined in cudnn.h when using cudnn (usually 1e-5)
    momentum : float, optional, default=0.899999976
        Momentum for moving average
    fix_gamma : boolean, optional, default=1
        Fix gamma while training
    use_global_stats : boolean, optional, default=0
        Whether use global moving statistics instead of local batch-norm.
        This will force change batch-norm into a scale shift operator.
    output_mean_var : boolean, optional, default=0
        Output the mean and inverse std
    axis : int, optional, default='1'
        Specify which shape axis the channel is specified
    cudnn_off : boolean, optional, default=0
        Do not select CUDNN operator, if available
    min_calib_range : float or None, optional, default=None
        The minimum scalar value in the form of float32 obtained through calibration.
        If present, it will be used to by quantized batch norm op to calculate primitive scale.
        Note: this calib_range is to calib bn output.
    max_calib_range : float or None, optional, default=None
        The maximum scalar value in the form of float32 obtained through calibration.
        If present, it will be used to by quantized batch norm op to calculate primitive scale.
        Note: this calib_range is to calib bn output.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.batch_norm(x, gamma, beta, running_mean, running_var,eps=eps,
                                 momentum=momentum, fix_gamma=fix_gamma,
                                 use_global_stats=use_global_stats, name=name,
                                 output_mean_var=output_mean_var, axis=axis, cudnn_off=cudnn_off,
                                 min_calib_range=min_calib_range, max_calib_range=max_calib_range)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def fully_connected(x, weight, name=None, bias=None, num_hidden=None,
                    no_bias=True, flatten=True):
    r"""Applies a linear transformation: :math:`Y = XW^T + b`.

    If ``flatten`` is set to be true, then the shapes are:

    - **data**: `(batch_size, x1, x2, ..., xn)`
    - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
    - **bias**: `(num_hidden,)`
    - **out**: `(batch_size, num_hidden)`

    If ``flatten`` is set to be false, then the shapes are:

    - **data**: `(x1, x2, ..., xn, input_dim)`
    - **weight**: `(num_hidden, input_dim)`
    - **bias**: `(num_hidden,)`
    - **out**: `(x1, x2, ..., xn, num_hidden)`

    The learnable parameters include both ``weight`` and ``bias``.

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    .. Note::

        The sparse support for FullyConnected is limited to forward evaluation with `row_sparse`
        weight and bias, where the length of `weight.indices` and `bias.indices` must be equal
        to `num_hidden`. This could be useful for model inference with `row_sparse` weights
        trained with importance sampling or noise contrastive estimation.

        To compute linear transformation with 'csr' sparse data, sparse.dot is recommended instead
        of sparse.FullyConnected.

    Parameters
    ----------
    data : NDArray
        Input data.
    weight : NDArray
        Weight matrix.
    bias : NDArray
        Bias parameter.
    num_hidden : int, required
        Number of hidden nodes of the output.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    flatten : boolean, optional, default=1
        Whether to collapse all but the first axis of the input data tensor.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.fully_connected(x, weight, bias=bias, num_hidden=num_hidden,
                                      no_bias=no_bias, flatten=flatten, name=name)

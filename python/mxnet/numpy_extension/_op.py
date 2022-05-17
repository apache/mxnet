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
           'activation', 'batch_norm', 'fully_connected', 'pick', 'convolution',
           'deconvolution', 'pooling', 'dropout', 'one_hot', 'rnn', 'embedding',
           'topk', 'layer_norm', 'leaky_relu', 'batch_dot', 'broadcast_like',
           'arange_like', 'group_norm']


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
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    length : NDArray
        The length array.
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
    return _mx_nd_npx.softmax(data, axis=axis, length=length, temperature=temperature,
                              use_length=use_length, dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def log_softmax(data, axis=-1, length=None, temperature=None, use_length=False, dtype=None):
    r"""Computes the log softmax of the input.
    This is equivalent to computing softmax followed by log.

    Parameters
    ----------
    data : NDArray
        The input array.
    axis : int, optional, default='-1'
        The axis along which to compute softmax.
    length : NDArray
        The length array.
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
    return _mx_nd_npx.log_softmax(data, axis=axis, length=length, temperature=temperature,
                                  use_length=use_length, dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def masked_softmax(data, mask, axis=-1, temperature=1.0, normalize=True):
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
                                     normalize=normalize)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def masked_log_softmax(data, mask, axis=-1, temperature=1.0, normalize=True):
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
                                         normalize=normalize)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def activation(data, act_type='relu', **kwargs):
    r"""Applies an activation function element-wise to the input.

    The following activation functions are supported:

    - `log_sigmoid`: :math:`y = log(\frac{1}{1 + exp(-x)})`
    - `mish`: :math:`y = x * tanh(log(1 + exp(x)))`
    - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
    - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
    - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
    - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
    - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`

    Parameters
    ----------
    data : NDArray
        The input array.
    act_type : {'log_sigmoid', 'mish', 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh'}, required
        Activation function to be applied.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.activation(data, act_type=act_type)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def batch_norm(x, gamma, beta, running_mean, running_var, eps=1e-3, momentum=0.9,
               fix_gamma=True, use_global_stats=False, output_mean_var=False, axis=1,
               cudnn_off=False, min_calib_range=None, max_calib_range=None, **kwargs):
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
    return _mx_nd_npx.batch_norm(x, gamma, beta, running_mean, running_var, eps=eps,
                                 momentum=momentum, fix_gamma=fix_gamma,
                                 use_global_stats=use_global_stats,
                                 output_mean_var=output_mean_var, axis=axis, cudnn_off=cudnn_off,
                                 min_calib_range=min_calib_range, max_calib_range=max_calib_range)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def fully_connected(x, weight, bias=None, num_hidden=None,
                    no_bias=True, flatten=True, **kwargs):
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
    return _mx_nd_npx.fully_connected(x, weight, bias, num_hidden=num_hidden,
                                      no_bias=no_bias, flatten=flatten)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def pick(data, index, axis=-1, mode='clip', keepdims=False):
    r"""Picks elements from an input array according to the input indices along the given axis.

    Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
    an output array of shape ``(i0,)`` with::

      output[i] = input[i, indices[i]]

    By default, if any index mentioned is too large, it is replaced by the index that addresses
    the last element along an axis (the `clip` mode).

    This function supports n-dimensional input and (n-1)-dimensional indices arrays.

    Parameters
    ----------
    data : NDArray
        The input array
    index : NDArray
        The index array
    axis : int or None, optional, default='-1'
        int or None. The axis to picking the elements.
        Negative values means indexing from right to left.
        If is `None`, the elements in the index w.r.t the flattened input will be picked.
    keepdims : boolean, optional, default=0
        If true, the axis where we pick the elements is
        left in the result as dimension with size one.
    mode : {'clip', 'wrap'},optional, default='clip'
        Specify how out-of-bound indices behave. Default is "clip".
        "clip" means clip to the range. So, if all indices mentioned are too large,
        they are replaced by the index that addresses the last element along an axis.
        "wrap" means to wrap around.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> x = np.array([[1., 2.],[3., 4.],[5., 6.]])

    picks elements with specified indices along axis 0

    >>> npx.pick(x, np.array([0, 1]), 0)
    array([1., 4.])

    picks elements with specified indices along axis 1

    >>> npx.pick(x, np.array([0, 1, 0]), 1)
    array([1., 4., 5.])

    picks elements with specified indices along axis 1 using 'wrap' mode
    to place indicies that would normally be out of bounds

    >>> npx.pick(x, np.array([2, -1, -2]), 1, mode='wrap')
    array([1., 4., 5.])

    picks elements with specified indices along axis 1 and dims are maintained

    >>> npx.pick(x, np.array([[1.], [0.], [2.]]), 1, keepdims=True)
    array([[2.],
           [3.],
           [6.]])
    """
    return _mx_nd_npx.pick(data, index, axis, mode, keepdims)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def convolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                pad=None, num_filter=1, num_group=1, workspace=1024, no_bias=False,
                cudnn_tune=None, cudnn_off=False, layout=None):
    r"""Compute *N*-D convolution on *(N+2)*-D input.

    In the 2-D convolution, given input data with shape *(batch_size,
    channel, height, width)*, the output is computed by

    .. math::

       out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
       weight[i,j,:,:]

    where :math:`\star` is the 2-D cross-correlation operator.

    For general 2-D convolution, the shapes are

    - **data**: *(batch_size, channel, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_height, out_width)*.

    Define::

      f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

    then we have::

      out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
      out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

    If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

    The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
    width)*. We can choose other layouts such as *NWC*.

    If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
    evenly into *g* parts along the channel axis, and also evenly split ``weight``
    along the first dimension. Next compute the convolution on the *i*-th part of
    the data with the *i*-th weight part. The output is obtained by concatenating all
    the *g* results.

    1-D convolution does not have *height* dimension but only *width* in space.

    - **data**: *(batch_size, channel, width)*
    - **weight**: *(num_filter, channel, kernel[0])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_width)*.

    3-D convolution adds an additional *depth* dimension besides *height* and
    *width*. The shapes are

    - **data**: *(batch_size, channel, depth, height, width)*
    - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
    - **bias**: *(num_filter,)*
    - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.

    Both ``weight`` and ``bias`` are learnable parameters.

    There are other options to tune the performance.

    - **cudnn_tune**: enable this option leads to higher startup time but may give
      faster speed. Options are

      - **off**: no tuning
      - **limited_workspace**:run test and pick the fastest algorithm that doesn't
        exceed workspace limit.
      - **fastest**: pick the fastest algorithm and ignore workspace limit.
      - **None** (default): the behavior is determined by environment variable
        ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
        (default), 2 for fastest.

    - **workspace**: A large number leads to more (GPU) memory usage but may improve
      the performance.

    Parameters
    ----------
    data : NDArray
        Input data to the ConvolutionOp.
    weight : NDArray
        Weight matrix.
    bias : NDArray
        Bias parameter.
    kernel : Shape(tuple), required
        Convolution kernel size: (w,), (h, w) or (d, h, w)
    stride : Shape(tuple), optional, default=[]
        Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
    num_filter : int (non-negative), required
        Convolution filter(channel) number
    num_group : int (non-negative), optional, default=1
        Number of group partitions.
    workspace : long (non-negative), optional, default=1024
        Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages.
        When CUDNN is not used, it determines the effective batch size of the convolution kernel.
        When CUDNN is used, it controls the maximum temporary storage used for tuning the best
        CUDNN kernel when `limited_workspace` strategy is used.
    no_bias : boolean, optional, default=0
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algo by running performance test.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
        NHWC and NDHWC are only supported on GPU.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.convolution(data=data, weight=weight, bias=bias, kernel=kernel,
                                  stride=stride, dilate=dilate, pad=pad, num_filter=num_filter,
                                  num_group=num_group, workspace=workspace, no_bias=no_bias,
                                  cudnn_tune=cudnn_tune, cudnn_off=cudnn_off, layout=layout)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def deconvolution(data=None, weight=None, bias=None, kernel=None, stride=None, dilate=None,
                  pad=None, adj=None, target_shape=None, num_filter=1, num_group=1,
                  workspace=1024, no_bias=False, cudnn_tune=None,
                  cudnn_off=False, layout=None):
    r"""Computes 1D, 2D or 3D transposed convolution (aka fractionally strided convolution) of
    the input tensor. This operation can be seen as the gradient of Convolution operation
    with respect to its input. Convolution usually reduces the size of the input.
    Transposed convolution works the other way, going from a smaller input
    to a larger output while preserving the connectivity pattern.

    Parameters
    ----------
    data : NDArray
        Input tensor to the deconvolution operation.
    weight : NDArray
        Weights representing the kernel.
    bias : NDArray
        Bias added to the result after the deconvolution operation.
    kernel : Shape(tuple), required
        Deconvolution kernel size: (w,), (h, w) or (d, h, w).
        This is same as the kernel size used for the corresponding convolution
    stride : Shape(tuple), optional, default=[]
        The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w).
        Defaults to 1 for each dimension.
    dilate : Shape(tuple), optional, default=[]
        Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w).
        Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        The amount of implicit zero padding added during convolution for each dimension of
        the input: (w,), (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice.
        If `target_shape` is set, `pad` will be ignored and a padding that will generate
        the target shape will be used. Defaults to no padding.
    adj : Shape(tuple), optional, default=[]
        Adjustment for output shape: (w,), (h, w) or (d, h, w).
        If `target_shape` is set, `adj` will be ignored and computed accordingly.
    target_shape : Shape(tuple), optional, default=[]
        Shape of the output tensor: (w,), (h, w) or (d, h, w).
    num_filter : int (non-negative), required
        Number of output filters.
    num_group : int (non-negative), optional, default=1
        Number of groups partition.
    workspace : long (non-negative), optional, default=512
        Maximum temporary workspace allowed (MB) in deconvolution. This parameter has two usages.
        When CUDNN is not used, it determines the effective batch size of the deconvolution kernel.
        When CUDNN is used, it controls the maximum temporary storage used for tuning
        the best CUDNN kernel when `limited_workspace` strategy is used.
    no_bias : boolean, optional, default=1
        Whether to disable bias parameter.
    cudnn_tune : {None, 'fastest', 'limited_workspace', 'off'},optional, default='None'
        Whether to pick convolution algorithm by running performance test.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn for this layer.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC'},optional, default='None'
        Set layout for input, output and weight. Empty for
        default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.
        NHWC and NDHWC are only supported on GPU.

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.deconvolution(data=data, weight=weight, bias=bias, kernel=kernel,
                                    stride=stride, dilate=dilate, pad=pad, adj=adj,
                                    target_shape=target_shape, num_filter=num_filter,
                                    num_group=num_group, workspace=workspace, no_bias=no_bias,
                                    cudnn_tune=cudnn_tune, cudnn_off=cudnn_off, layout=layout)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def pooling(data=None, kernel=None, stride=None, pad=None, pool_type="max",
            pooling_convention="valid", global_pool=False, cudnn_off=False,
            p_value=None, count_include_pad=None, layout=None, **kwargs):
    r"""Performs pooling on the input.

    The shapes for 1-D pooling are

    - **data** and **out**: *(batch_size, channel, width)* (NCW layout) or
      *(batch_size, width, channel)* (NWC layout),

    The shapes for 2-D pooling are

    - **data** and **out**: *(batch_size, channel, height, width)* (NCHW layout) or
      *(batch_size, height, width, channel)* (NHWC layout),

        out_height = f(height, kernel[0], pad[0], stride[0])
        out_width = f(width, kernel[1], pad[1], stride[1])

    The definition of *f* depends on ``pooling_convention``, which has two options:

    - **valid** (default)::

        f(x, k, p, s) = floor((x+2*p-k)/s)+1

    - **full**, which is compatible with Caffe::

        f(x, k, p, s) = ceil((x+2*p-k)/s)+1

    When ``global_pool`` is set to be true, then global pooling is performed. It will reset
    ``kernel=(height, width)`` and set the appropiate padding to 0.

    Three pooling options are supported by ``pool_type``:

    - **avg**: average pooling
    - **max**: max pooling
    - **sum**: sum pooling
    - **lp**: Lp pooling

    For 3-D pooling, an additional *depth* dimension is added before
    *height*. Namely the input data and output will have shape *(batch_size, channel, depth,
    height, width)* (NCDHW layout) or *(batch_size, depth, height, width, channel)* (NDHWC layout).

    Notes on Lp pooling:

    Lp pooling was first introduced by this paper: https://arxiv.org/pdf/1204.3968.pdf.
    L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.
    We can see that Lp pooling stands between those two, in practice the most common value for p is 2.

    For each window ``X``, the mathematical expression for Lp pooling is:

    :math:`f(X) = \sqrt[p]{\sum_{x}^{X} x^p}`

    Parameters
    ----------
    data : NDArray
        Input data to the pooling operator.
    kernel : Shape(tuple), optional, default=[]
        Pooling kernel size: (y, x) or (d, y, x)
    pool_type : {'avg', 'lp', 'max', 'sum'},optional, default='max'
        Pooling type to be applied.
    global_pool : boolean, optional, default=0
        Ignore kernel size, do global pooling based on current input feature map.
    cudnn_off : boolean, optional, default=0
        Turn off cudnn pooling and use MXNet pooling operator.
    pooling_convention : {'full', 'same', 'valid'},optional, default='valid'
        Pooling convention to be applied.
    stride : Shape(tuple), optional, default=[]
        Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.
    pad : Shape(tuple), optional, default=[]
        Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
    p_value : int or None, optional, default='None'
        Value of p for Lp pooling, can be 1 or 2, required for Lp Pooling.
    count_include_pad : boolean or None, optional, default=None
        Only used for AvgPool, specify whether to count padding elements for averagecalculation.
        For example, with a 5*5 kernel on a 3*3 corner of a image,the sum of the 9 valid elements will
        be divided by 25 if this is set to true,or it will be divided by 9 if this is set to false.
        Defaults to true.
    layout : {None, 'NCDHW', 'NCHW', 'NCW', 'NDHWC', 'NHWC', 'NWC'},optional, default='None'
        Set layout for input and output. Empty for
        default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.pooling(data=data, kernel=kernel, stride=stride, pad=pad,
                              pool_type=pool_type, pooling_convention=pooling_convention,
                              global_pool=global_pool, cudnn_off=cudnn_off, p_value=p_value,
                              count_include_pad=count_include_pad, layout=layout)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def dropout(data, p=0.5, mode="training", axes=None, cudnn_off=False, **kwargs):
    r"""Applies dropout operation to input array.

    - During training, each element of the input is set to zero with probability p.
      The whole array is rescaled by :math:`1/(1-p)` to keep the expected
      sum of the input unchanged.

    - During testing, this operator does not change the input if mode is 'training'.
      If mode is 'always', the same computaion as during training will be applied.

    Parameters
    ----------
    data : NDArray
        Input array to which dropout will be applied.
    p : float, optional, default=0.5
        Fraction of the input that gets dropped out during training time.
    mode : {'always', 'training'},optional, default='training'
        Whether to only turn on dropout during training or to also turn on for inference.
    axes : Shape(tuple), optional, default=[]
        Axes for variational dropout kernel.
    cudnn_off : boolean or None, optional, default=0
        Whether to turn off cudnn in dropout operator. This option is ignored if axes is specified.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.dropout(data=data, p=p, mode=mode, axes=axes, cudnn_off=cudnn_off)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def one_hot(data, depth=None, on_value=1.0, off_value=0.0, dtype="float32"):
    r"""Returns a one-hot array.

    The locations represented by `indices` take value `on_value`, while all
    other locations take value `off_value`.

    `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
    in an output array of shape ``(i0, i1, d)`` with::

      output[i,j,:] = off_value
      output[i,j,indices[i,j]] = on_value

    Parameters
    ----------
    indices : NDArray
        array of locations where to set on_value
    depth : long, required
        Depth of the one hot dimension.
    on_value : double, optional, default=1
        The value assigned to the locations represented by indices.
    off_value : double, optional, default=0
        The value assigned to the locations not represented by indices.
    dtype : {'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},
            optional, default='float32'
        DType of the output

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> data = np.array([1,0,2,0])
    >>> npx.one_hot(data, 3)
    array([[0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.]], dtype=float64)
    >>> npx.one_hot(data, 3, on_value=8, off_value=1, dtype='int32')
    array([[1, 8, 1],
           [8, 1, 1],
           [1, 1, 8],
           [8, 1, 1]], dtype=int32)
    >>> data = np.array([[1,0],[1,0],[2,0]])
    >>> npx.one_hot(data, 3)
    array([[[0., 1., 0.],
            [1., 0., 0.]],
           [[0., 1., 0.],
            [1., 0., 0.]],
           [[0., 0., 1.],
            [1., 0., 0.]]], dtype=float64)
    """
    return _mx_nd_npx.one_hot(data=data, depth=depth, on_value=on_value, off_value=off_value,
                              dtype=dtype)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def rnn(data=None, parameters=None, state=None, state_cell=None, sequence_length=None,
        mode=None, state_size=None, num_layers=None, bidirectional=False,
        state_outputs=False, p=0.0, use_sequence_length=False, projection_size=None,
        lstm_state_clip_min=None, lstm_state_clip_max=None, lstm_state_clip_nan=None):
    r"""Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are
    implemented, with both multi-layer and bidirectional support.

    When the input data is of type float32 and the environment variables MXNET_CUDA_ALLOW_TENSOR_CORE
    and MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION are set to 1, this operator will try to use
    pseudo-float16 precision (float32 math with float16 I/O) precision in order to use
    Tensor Cores on suitable NVIDIA GPUs. This can sometimes give significant speedups.

    **Vanilla RNN**

    Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported:
    ReLU and Tanh.

    With ReLU activation function:

    .. math::
        h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    With Tanh activtion function:

    .. math::
        h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})

    Reference paper: Finding structure in time - Elman, 1988.
    https://axon.cs.byu.edu/~martinez/classes/678/Papers/Elman_time.pdf

    **LSTM**

    Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf

    .. math::
      \begin{array}{ll}
                i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\
                f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\
                o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\
                c_t = f_t * c_{(t-1)} + i_t * g_t \\
                h_t = o_t * \tanh(c_t)
                \end{array}

    With the projection size being set, LSTM could use the projection feature to reduce the parameters
    size and give some speedups without significant damage to the accuracy.

    Long Short-Term Memory Based Recurrent Neural Network Architectures for Large Vocabulary Speech
    Recognition - Sak et al. 2014. https://arxiv.org/abs/1402.1128

    .. math::
      \begin{array}{ll}
                i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{ri} r_{(t-1)} + b_{ri}) \\
                f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{rf} r_{(t-1)} + b_{rf}) \\
                g_t = \tanh(W_{ig} x_t + b_{ig} + W_{rc} r_{(t-1)} + b_{rg}) \\
                o_t = \mathrm{sigmoid}(W_{io} x_t + b_{o} + W_{ro} r_{(t-1)} + b_{ro}) \\
                c_t = f_t * c_{(t-1)} + i_t * g_t \\
                h_t = o_t * \tanh(c_t)
                r_t = W_{hr} h_t
                \end{array}

    **GRU**

    Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078

    The definition of GRU here is slightly different from paper but compatible with CUDNN.

    .. math::
      \begin{array}{ll}
                r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
                z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
                n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
                h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\
                \end{array}

    Parameters
    ----------
    data : NDArray
        Input data to RNN
    parameters : NDArray
        Vector of all RNN trainable parameters concatenated
    state : NDArray
        initial hidden state of the RNN
    state_cell : NDArray
        initial cell state for LSTM networks (only for LSTM)
    sequence_length : NDArray
        Vector of valid sequence lengths for each element in batch.
        (Only used if use_sequence_length kwarg is True)
    state_size : int (non-negative), required
        size of the state for each layer
    num_layers : int (non-negative), required
        number of stacked layers
    bidirectional : boolean, optional, default=0
        whether to use bidirectional recurrent layers
    mode : {'gru', 'lstm', 'rnn_relu', 'rnn_tanh'}, required
        the type of RNN to compute
    p : float, optional, default=0
        drop rate of the dropout on the outputs of each RNN layer, except the last layer.
    state_outputs : boolean, optional, default=0
        Whether to have the states as symbol outputs.
    projection_size : int or None, optional, default='None'
        size of project size
    lstm_state_clip_min : double or None, optional, default=None
        Minimum clip value of LSTM states. This option must be used together with lstm_state_clip_max.
    lstm_state_clip_max : double or None, optional, default=None
        Maximum clip value of LSTM states. This option must be used together with lstm_state_clip_min.
    lstm_state_clip_nan : boolean, optional, default=0
        Whether to stop NaN from propagating in state by clipping it to min/max.
        If clipping range is not specified, this option is ignored.
    use_sequence_length : boolean, optional, default=0
        If set to true, this layer takes in an extra input parameter `sequence_length`
        to specify variable length sequence

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.rnn(data=data, parameters=parameters, state=state, state_cell=state_cell,
                          sequence_length=sequence_length, mode=mode, state_size=state_size,
                          num_layers=num_layers, bidirectional=bidirectional,
                          state_outputs=state_outputs, p=p, use_sequence_length=use_sequence_length,
                          projection_size=projection_size, lstm_state_clip_min=lstm_state_clip_min,
                          lstm_state_clip_max=lstm_state_clip_max,
                          lstm_state_clip_nan=lstm_state_clip_nan)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def embedding(data, weight, input_dim=None, output_dim=None, dtype="float32", sparse_grad=False,
              **kwargs):
    r"""Maps integer indices to vector representations (embeddings).

    This operator maps words to real-valued vectors in a high-dimensional space,
    called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
    For example, it has been noted that in the learned embedding spaces, similar words tend
    to be close to each other and dissimilar words far apart.

    For an input array of shape (d1, ..., dK),
    the shape of an output array is (d1, ..., dK, output_dim).
    All the input values should be integers in the range [0, input_dim).

    If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
    (ip0, op0).

    When "sparse_grad" is False, if any index mentioned is too large, it is replaced by the index that
    addresses the last vector in an embedding matrix.
    When "sparse_grad" is True, an error will be raised if invalid indices are found.

    The storage type of weight can be either row_sparse or default.

    .. Note::

        If "sparse_grad" is set to True, the storage type of gradient w.r.t weights will be
        "row_sparse". Only a subset of optimizers support sparse gradients, including SGD, AdaGrad
        and Adam. Note that by default lazy updates is turned on, which may perform differently
        from standard updates. For more details, please check the Optimization API at:
        https://mxnet.apache.org/versions/master/api/python/docs/api/optimizer/index.html

    Parameters
    ----------
    data : NDArray
        The input array to the embedding operator.
    weight : NDArray
        The embedding weight matrix.
    input_dim : long, required
        Vocabulary size of the input indices.
    output_dim : long, required
        Dimension of the embedding vectors.
    dtype : {'bfloat16', 'float16', 'float32', 'float64', 'int32', 'int64', 'int8', 'uint8'},
            optional, default='float32'
        Data type of weight.
    sparse_grad : boolean, optional, default=0
        Compute row sparse gradient in the backward calculation.
        If set to True, the grad's storage type is row_sparse.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> input_dim = 4
    >>> output_dim = 5

    Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)

    >>> y = np.arange(input_dim * output_dim).reshape(input_dim, output_dim)
    >>> y
    array([[ 0.,  1.,  2.,  3.,  4.],
           [ 5.,  6.,  7.,  8.,  9.],
           [10., 11., 12., 13., 14.],
           [15., 16., 17., 18., 19.]])

    Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]

    >>> x = np.array([[1., 3.], [0., 2.]])
    >>> x
    array([[1., 3.],
           [0., 2.]])

    Mapped input x to its vector representation y.

    >>> npx.embedding(x, y, input_dim, output_dim)
    array([[[ 5.,  6.,  7.,  8.,  9.],
            [15., 16., 17., 18., 19.]],

           [[ 0.,  1.,  2.,  3.,  4.],
            [10., 11., 12., 13., 14.]]])
    """
    return _mx_nd_npx.embedding(data=data, weight=weight, input_dim=input_dim, output_dim=output_dim,
                                dtype=dtype, sparse_grad=sparse_grad)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def topk(data, axis=-1, k=1, ret_typ="indices", is_ascend=False, dtype="float32"):
    r"""Returns the indices of the top *k* elements in an input array along the given
     axis (by default).
     If ret_type is set to 'value' returns the value of top *k* elements (instead of indices).
     In case of ret_type = 'both', both value and index would be returned.
     The returned elements will be sorted.

    Parameters
    ----------
    data : NDArray
        The input array
    axis : int or None, optional, default='-1'
        Axis along which to choose the top k indices.
        If not given, the flattened array is used. Default is -1.
    k : int, optional, default='1'
        Number of top elements to select, should be always smaller than or equal to
        the element number in the given axis. A global sort is performed if set k < 1.
    ret_typ : {'both', 'indices', 'mask', 'value'},optional, default='indices'
        The return type.
     "value" means to return the top k values,
     "indices" means to return the indices of the top k values,
     "mask" means to return a mask array containing 0 and 1. 1 means the top k values.
     "both" means to return a list of both values and indices of top k elements.
    is_ascend : boolean, optional, default=0
        Whether to choose k largest or k smallest elements.
        Top K largest elements will be chosen if set to false.
    dtype : {'float16', 'float32', 'float64', 'int32', 'int64', 'uint8'},
            optional, default='float32'
        DType of the output indices when ret_typ is "indices" or "both".
        An error will be raised if the selected data type cannot precisely represent the indices.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> x = np.array([[0.3, 0.2, 0.4], [0.1, 0.3, 0.2]])

    returns an index of the largest element on last axis

    >>> npx.topk(x)
    array([[2.],
           [1.]])

    returns the value of top-2 largest elements on last axis

    >>> npx.topk(x, ret_typ='value', k=2)
    array([[0.4, 0.3],
           [0.3, 0.2]])

    returns the value of top-2 smallest elements on last axis

    >>> npx.topk(x, ret_typ='value', k=2, is_ascend=1)
    array([[0.2, 0.3],
           [0.1, 0.2]])

    returns the value of top-2 largest elements on axis 0

    >>> npx.topk(x, axis=0, ret_typ='value', k=2)
    array([[0.3, 0.3, 0.4],
           [0.1, 0.2, 0.2]])

    flattens and then returns list of both values and indices

    >>> npx.topk(x, ret_typ='both', k=2)
    [array([[0.4, 0.3], [0.3, 0.2]]),
     array([[2., 0.], [1., 2.]])]
    """
    return _mx_nd_npx.topk(data=data, axis=axis, k=k, ret_typ=ret_typ, is_ascend=is_ascend, dtype=dtype)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def layer_norm(data=None, gamma=None, beta=None, axis=None, eps=None, output_mean_var=None):
    r"""Layer normalization.

    Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
    well as offset ``beta``.

    Assume the input has more than one dimension and we normalize along axis 1.
    We first compute the mean and variance along this axis and then
    compute the normalized output, which has the same shape as input, as following:

    .. math::

      out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

    Both ``gamma`` and ``beta`` are learnable parameters.

    Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.

    Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
    have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
    ``data_std``. Note that no gradient will be passed through these two outputs.

    The parameter ``axis`` specifies which axis of the input shape denotes
    the 'channel' (separately normalized groups).  The default is -1, which sets the channel
    axis to be the last item in the input shape.

    Parameters
    ----------
    data : NDArray
        Input data to layer normalization
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    axis : int, optional, default='-1'
        The axis to perform layer normalization.
        Usually, this should be be axis of the channel dimension.
        Negative values means indexing from right to left.
    eps : float, optional, default=9.99999975e-06
        An `epsilon` parameter to prevent division by 0.
    output_mean_var : boolean, optional, default=0
        Output the mean and std calculated along the given axis.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.layer_norm(data=data, gamma=gamma, beta=beta, axis=axis, eps=eps,
                                 output_mean_var=output_mean_var)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def leaky_relu(data=None, gamma=None, act_type="leaky", slope=0.25, lower_bound=0.125,
               upper_bound=0.334, **kwargs):
    r"""Applies Leaky rectified linear unit activation element-wise to the input.

    Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
    when the input is negative and has a slope of one when input is positive.

    The following modified ReLU Activation functions are supported:

    - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
    - *gelu*: Gaussian Error Linear Unit. `y = 0.5 * x * (1 + erf(x / sqrt(2)))`
    - *selu*: Scaled Exponential Linear Unit. `y = lambda * (x > 0 ? x : alpha * (exp(x) - 1))` where
      *lambda = 1.0507009873554804934193349852946* and *alpha = 1.6732632423543772848170429916717*.
    - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
    - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.
    - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from
      *[lower_bound, upper_bound)* for training, while fixed to be
      *(lower_bound+upper_bound)/2* for inference.

    Parameters
    ----------
    data : NDArray
        Input data to activation function.
    gamma : NDArray
        Input data to activation function.
    act_type : {'elu', 'gelu', 'leaky', 'prelu', 'rrelu', 'selu'},optional, default='leaky'
        Activation function to be applied.
    slope : float, optional, default=0.25
        Init slope for the activation. (For leaky and elu only)
    lower_bound : float, optional, default=0.125
        Lower bound of random slope. (For rrelu only)
    upper_bound : float, optional, default=0.333999991
        Upper bound of random slope. (For rrelu only)

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.leaky_relu(data=data, gamma=gamma, act_type=act_type, slope=slope,
                                 lower_bound=lower_bound, upper_bound=upper_bound)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def batch_dot(a, b, transpose_a=False, transpose_b=False, forward_stype="default"):
    r"""Batchwise dot product.

    ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
    ``y`` are data in batch, namely N-D (N >= 3) arrays in shape of `(B0, ..., B_i, :, :)`.

    For example, given ``x`` with shape `(B_0, ..., B_i, N, M)` and ``y`` with shape
    `(B_0, ..., B_i, M, K)`, the result array will have shape `(B_0, ..., B_i, N, K)`,
    which is computed by::

       batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ..., b_i, :, :])

    Parameters
    ----------
    lhs : NDArray
        The first input
    rhs : NDArray
        The second input
    transpose_a : boolean, optional, default=0
        If true then transpose the first input before dot.
    transpose_b : boolean, optional, default=0
        If true then transpose the second input before dot.
    forward_stype : {None, 'csr', 'default', 'row_sparse'},optional, default='None'
        The desired storage type of the forward output given by user,
        if thecombination of input storage types and this hint does not matchany implemented ones,
        the dot operator will perform fallback operationand still produce
        an output of the desired storage type.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.batch_dot(a=a, b=b, transpose_a=transpose_a,
                                transpose_b=transpose_b, forward_stype=forward_stype)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def broadcast_like(lhs, rhs, lhs_axes=None, rhs_axes=None):
    r"""Broadcasts lhs to have the same shape as rhs.

    Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
    with arrays of different shapes efficiently without creating multiple copies of arrays.
    Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
    for more explanation.

    Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
    `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.

    Parameters
    ----------
    lhs : NDArray
        First input.
    rhs : NDArray
        Second input.
    lhs_axes : Shape or None, optional, default=None
        Axes to perform broadcast on in the first input array
    rhs_axes : Shape or None, optional, default=None
        Axes to copy from the second input array

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    example
    -------
    >>> a = np.array([[1,2,3]])
    >>> b = np.array([[5,6,7],[7,8,9]])
    >>> npx.broadcast_like(a, b)
    array([[1., 2., 3.],
           [1., 2., 3.]])
    >>> a = np.array([9])
    >>> b = np.array([1,2,3,4,5])
    >>> npx.broadcast_like(a, b, lhs_axes=(0,), rhs_axes=(-1,))
    array([9., 9., 9., 9., 9.])
    """
    return _mx_nd_npx.broadcast_like(lhs=lhs, rhs=rhs, lhs_axes=lhs_axes, rhs_axes=rhs_axes)


# pylint: disable=too-many-arguments, unused-argument
@set_module('mxnet.numpy_extension')
def arange_like(data, start=0.0, step=1.0, repeat=1, ctx=None, axis=None):
    r"""Return an array with evenly spaced values. If axis is not given, the output will
    have the same shape as the input array. Otherwise, the output will be a 1-D array with size of
    the specified axis in input shape.

    Parameters
    ----------
    data : NDArray
        The input
    start : double, optional, default=0
        Start of interval. The interval includes this value. The default start value is 0.
    step : double, optional, default=1
        Spacing between values.
    repeat : int, optional, default='1'
        The repeating time of all elements.
        E.g repeat=3, the element a will be repeated three times --> a, a, a.
    ctx : string, optional, default=''
        Context of output, in format [cpu|gpu|cpu_pinned](n).Only used for imperative calls.
    axis : int or None, optional, default='None'
        Arange elements according to the size of a certain axis of input array.
        The negative numbers are interpreted counting from the backward.
        If not provided, will arange elements according to the input shape.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.

    Example
    -------
    >>> x = np.random.uniform(0, 1, size=(3,4))
    >>> x
    array([[0.5488135 , 0.5928446 , 0.71518934, 0.84426576],
           [0.60276335, 0.8579456 , 0.5448832 , 0.8472517 ],
           [0.4236548 , 0.6235637 , 0.6458941 , 0.3843817 ]])
    >>> npx.arange_like(x, start=0)
    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])
    >>> npx.arange_like(x, start=0, axis=-1)
    array([0., 1., 2., 3.])
    """
    return _mx_nd_npx.arange_like(data=data, start=start, step=step, repeat=repeat,
                                  ctx=ctx, axis=axis)


# pylint: disable=too-many-arguments
@set_module('mxnet.numpy_extension')
def group_norm(data, gamma, beta, num_groups=1, eps=1e-3, output_mean_var=False):
    r"""Group normalization.

    The input channels are separated into ``num_groups`` groups,
    each containing ``num_channels / num_groups`` channels.
    The mean and standard-deviation are calculated separately over the each group.

    .. math::

      data = data.reshape((N, num_groups, C // num_groups, ...))
      out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta

    Both ``gamma`` and ``beta`` are learnable parameters.



    Defined in ../src/operator/nn/group_norm.cc:L78

    Parameters
    ----------
    data : NDArray
        Input data
    gamma : NDArray
        gamma array
    beta : NDArray
        beta array
    num_groups : int, optional, default='1'
        Total number of groups.
    eps : float, optional, default=9.99999975e-06
        An `epsilon` parameter to prevent division by 0.
    output_mean_var : boolean, optional, default=0
        Output the mean and std calculated along the given axis.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return _mx_nd_npx.group_norm(data=data, gamma=gamma, beta=beta, num_groups=num_groups,
                                 eps=eps, output_mean_var=output_mean_var)

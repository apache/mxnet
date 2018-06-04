/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

// scalastyle:off
package org.apache.mxnet
abstract class NDArrayBase {
  /**
  * Applies an activation function element-wise to the input.
  * 
  * The following activation functions are supported:
  * 
  * - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
  * - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
  * - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
  * - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
  * - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`
  * 
  * 
  * 
  * Defined in src/operator/nn/activation.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def Activation(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies an activation function element-wise to the input.
  * 
  * The following activation functions are supported:
  * 
  * - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
  * - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
  * - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
  * - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`
  * - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`
  * 
  * 
  * 
  * Defined in src/operator/nn/activation.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def Activation(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batch normalization.
  * 
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis:
  * 
  * .. math::
  * 
  *   data\_mean[i] = mean(data[:,i,:,...]) \\
  *   data\_var[i] = var(data[:,i,:,...])
  * 
  * Then compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]
  * 
  * Both *mean* and *var* returns a scalar by treating the input as a vector.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these 
  * two outputs are blocked.
  * 
  * Besides the inputs and the outputs, this operator accepts two auxiliary
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length
  * vectors. They are global statistics for the whole dataset, which are updated
  * by::
  * 
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)
  * 
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
  * the output. It is often used during inference.
  * 
  * The parameter ``axis`` specifies which axis of the input shape denotes
  * the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
  * axis to be the last item in the input shape.
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
  * then set ``gamma`` to 1 and its gradient to 0.
  * 
  * 
  * 
  * Defined in src/operator/nn/batch_norm.cc:L575
  * @return org.apache.mxnet.NDArray
  */
def BatchNorm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batch normalization.
  * 
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis:
  * 
  * .. math::
  * 
  *   data\_mean[i] = mean(data[:,i,:,...]) \\
  *   data\_var[i] = var(data[:,i,:,...])
  * 
  * Then compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]
  * 
  * Both *mean* and *var* returns a scalar by treating the input as a vector.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these 
  * two outputs are blocked.
  * 
  * Besides the inputs and the outputs, this operator accepts two auxiliary
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length
  * vectors. They are global statistics for the whole dataset, which are updated
  * by::
  * 
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)
  * 
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
  * the output. It is often used during inference.
  * 
  * The parameter ``axis`` specifies which axis of the input shape denotes
  * the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel
  * axis to be the last item in the input shape.
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
  * then set ``gamma`` to 1 and its gradient to 0.
  * 
  * 
  * 
  * Defined in src/operator/nn/batch_norm.cc:L575
  * @return org.apache.mxnet.NDArray
  */
def BatchNorm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batch normalization.
  * 
  * This operator is DEPRECATED. Perform BatchNorm on the input.
  * 
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis:
  * 
  * .. math::
  * 
  *   data\_mean[i] = mean(data[:,i,:,...]) \\
  *   data\_var[i] = var(data[:,i,:,...])
  * 
  * Then compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]
  * 
  * Both *mean* and *var* returns a scalar by treating the input as a vector.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * ``data_var`` as well, which are needed for the backward pass.
  * 
  * Besides the inputs and the outputs, this operator accepts two auxiliary
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length
  * vectors. They are global statistics for the whole dataset, which are updated
  * by::
  * 
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)
  * 
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
  * the output. It is often used during inference.
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
  * then set ``gamma`` to 1 and its gradient to 0.
  * 
  * 
  * 
  * Defined in src/operator/batch_norm_v1.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def BatchNorm_v1(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batch normalization.
  * 
  * This operator is DEPRECATED. Perform BatchNorm on the input.
  * 
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis:
  * 
  * .. math::
  * 
  *   data\_mean[i] = mean(data[:,i,:,...]) \\
  *   data\_var[i] = var(data[:,i,:,...])
  * 
  * Then compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]
  * 
  * Both *mean* and *var* returns a scalar by treating the input as a vector.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * ``data_var`` as well, which are needed for the backward pass.
  * 
  * Besides the inputs and the outputs, this operator accepts two auxiliary
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length
  * vectors. They are global statistics for the whole dataset, which are updated
  * by::
  * 
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)
  * 
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute
  * the output. It is often used during inference.
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,
  * then set ``gamma`` to 1 and its gradient to 0.
  * 
  * 
  * 
  * Defined in src/operator/batch_norm_v1.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def BatchNorm_v1(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies bilinear sampling to input feature map.
  * 
  * Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,
  * except that the operator has the backward pass.
  * 
  * Given :math:`data` and :math:`grid`, then the output is computed by
  * 
  * .. math::
  *   x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
  *   y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
  *   output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})
  * 
  * :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.
  * The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).
  * 
  * The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].
  * 
  * BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.
  * GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
  * If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.
  * 
  * Example 1::
  * 
  *   ## Zoom out data two times
  *   data = array([[[[1, 4, 3, 6],
  *                   [1, 8, 8, 9],
  *                   [0, 4, 1, 5],
  *                   [1, 0, 1, 3]]]])
  * 
  *   affine_matrix = array([[2, 0, 0],
  *                          [0, 2, 0]])
  * 
  *   affine_matrix = reshape(affine_matrix, shape=(1, 6))
  * 
  *   grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))
  * 
  *   out = BilinearSampler(data, grid)
  * 
  *   out
  *   [[[[ 0,   0,     0,   0],
  *      [ 0,   3.5,   6.5, 0],
  *      [ 0,   1.25,  2.5, 0],
  *      [ 0,   0,     0,   0]]]
  * 
  * 
  * Example 2::
  * 
  *   ## shift data horizontally by -1 pixel
  * 
  *   data = array([[[[1, 4, 3, 6],
  *                   [1, 8, 8, 9],
  *                   [0, 4, 1, 5],
  *                   [1, 0, 1, 3]]]])
  * 
  *   warp_maxtrix = array([[[[1, 1, 1, 1],
  *                           [1, 1, 1, 1],
  *                           [1, 1, 1, 1],
  *                           [1, 1, 1, 1]],
  *                          [[0, 0, 0, 0],
  *                           [0, 0, 0, 0],
  *                           [0, 0, 0, 0],
  *                           [0, 0, 0, 0]]]])
  * 
  *   grid = GridGenerator(data=warp_matrix, transform_type='warp')
  *   out = BilinearSampler(data, grid)
  * 
  *   out
  *   [[[[ 4,  3,  6,  0],
  *      [ 8,  8,  9,  0],
  *      [ 4,  1,  5,  0],
  *      [ 0,  1,  3,  0]]]
  * 
  * 
  * Defined in src/operator/bilinear_sampler.cc:L245
  * @return org.apache.mxnet.NDArray
  */
def BilinearSampler(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies bilinear sampling to input feature map.
  * 
  * Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,
  * except that the operator has the backward pass.
  * 
  * Given :math:`data` and :math:`grid`, then the output is computed by
  * 
  * .. math::
  *   x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\
  *   y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\
  *   output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})
  * 
  * :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.
  * The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).
  * 
  * The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].
  * 
  * BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.
  * GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.
  * If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.
  * 
  * Example 1::
  * 
  *   ## Zoom out data two times
  *   data = array([[[[1, 4, 3, 6],
  *                   [1, 8, 8, 9],
  *                   [0, 4, 1, 5],
  *                   [1, 0, 1, 3]]]])
  * 
  *   affine_matrix = array([[2, 0, 0],
  *                          [0, 2, 0]])
  * 
  *   affine_matrix = reshape(affine_matrix, shape=(1, 6))
  * 
  *   grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))
  * 
  *   out = BilinearSampler(data, grid)
  * 
  *   out
  *   [[[[ 0,   0,     0,   0],
  *      [ 0,   3.5,   6.5, 0],
  *      [ 0,   1.25,  2.5, 0],
  *      [ 0,   0,     0,   0]]]
  * 
  * 
  * Example 2::
  * 
  *   ## shift data horizontally by -1 pixel
  * 
  *   data = array([[[[1, 4, 3, 6],
  *                   [1, 8, 8, 9],
  *                   [0, 4, 1, 5],
  *                   [1, 0, 1, 3]]]])
  * 
  *   warp_maxtrix = array([[[[1, 1, 1, 1],
  *                           [1, 1, 1, 1],
  *                           [1, 1, 1, 1],
  *                           [1, 1, 1, 1]],
  *                          [[0, 0, 0, 0],
  *                           [0, 0, 0, 0],
  *                           [0, 0, 0, 0],
  *                           [0, 0, 0, 0]]]])
  * 
  *   grid = GridGenerator(data=warp_matrix, transform_type='warp')
  *   out = BilinearSampler(data, grid)
  * 
  *   out
  *   [[[[ 4,  3,  6,  0],
  *      [ 8,  8,  9,  0],
  *      [ 4,  1,  5,  0],
  *      [ 0,  1,  3,  0]]]
  * 
  * 
  * Defined in src/operator/bilinear_sampler.cc:L245
  * @return org.apache.mxnet.NDArray
  */
def BilinearSampler(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Stops gradient computation.
  * 
  * Stops the accumulated gradient of the inputs from flowing through this operator
  * in the backward direction. In other words, this operator prevents the contribution
  * of its inputs to be taken into account for computing gradients.
  * 
  * Example::
  * 
  *   v1 = [1, 2]
  *   v2 = [0, 1]
  *   a = Variable('a')
  *   b = Variable('b')
  *   b_stop_grad = stop_gradient(3 * b)
  *   loss = MakeLoss(b_stop_grad + a)
  * 
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  *   executor.forward(is_train=True, a=v1, b=v2)
  *   executor.outputs
  *   [ 1.  5.]
  * 
  *   executor.backward()
  *   executor.grad_arrays
  *   [ 0.  0.]
  *   [ 1.  1.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L241
  * @return org.apache.mxnet.NDArray
  */
def BlockGrad(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Stops gradient computation.
  * 
  * Stops the accumulated gradient of the inputs from flowing through this operator
  * in the backward direction. In other words, this operator prevents the contribution
  * of its inputs to be taken into account for computing gradients.
  * 
  * Example::
  * 
  *   v1 = [1, 2]
  *   v2 = [0, 1]
  *   a = Variable('a')
  *   b = Variable('b')
  *   b_stop_grad = stop_gradient(3 * b)
  *   loss = MakeLoss(b_stop_grad + a)
  * 
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  *   executor.forward(is_train=True, a=v1, b=v2)
  *   executor.outputs
  *   [ 1.  5.]
  * 
  *   executor.backward()
  *   executor.grad_arrays
  *   [ 0.  0.]
  *   [ 1.  1.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L241
  * @return org.apache.mxnet.NDArray
  */
def BlockGrad(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts all elements of the input to a new type.
  * 
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
  * 
  * Example::
  * 
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L385
  * @return org.apache.mxnet.NDArray
  */
def Cast(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts all elements of the input to a new type.
  * 
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
  * 
  * Example::
  * 
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L385
  * @return org.apache.mxnet.NDArray
  */
def Cast(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Joins input arrays along a given axis.
  * 
  * .. note:: `Concat` is deprecated. Use `concat` instead.
  * 
  * The dimensions of the input arrays should be the same except the axis along
  * which they will be concatenated.
  * The dimension of the output array along the concatenated axis will be equal
  * to the sum of the corresponding dimensions of the input arrays.
  * 
  * Example::
  * 
  *    x = [[1,1],[2,2]]
  *    y = [[3,3],[4,4],[5,5]]
  *    z = [[6,6], [7,7],[8,8]]
  * 
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],
  *                           [ 2.,  2.],
  *                           [ 3.,  3.],
  *                           [ 4.,  4.],
  *                           [ 5.,  5.],
  *                           [ 6.,  6.],
  *                           [ 7.,  7.],
  *                           [ 8.,  8.]]
  * 
  *    Note that you cannot concat x,y,z along dimension 1 since dimension
  *    0 is not the same for all the input arrays.
  * 
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
  *                          [ 4.,  4.,  7.,  7.],
  *                          [ 5.,  5.,  8.,  8.]]
  * 
  * 
  * 
  * Defined in src/operator/nn/concat.cc:L235
  * @return org.apache.mxnet.NDArray
  */
def Concat(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Joins input arrays along a given axis.
  * 
  * .. note:: `Concat` is deprecated. Use `concat` instead.
  * 
  * The dimensions of the input arrays should be the same except the axis along
  * which they will be concatenated.
  * The dimension of the output array along the concatenated axis will be equal
  * to the sum of the corresponding dimensions of the input arrays.
  * 
  * Example::
  * 
  *    x = [[1,1],[2,2]]
  *    y = [[3,3],[4,4],[5,5]]
  *    z = [[6,6], [7,7],[8,8]]
  * 
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],
  *                           [ 2.,  2.],
  *                           [ 3.,  3.],
  *                           [ 4.,  4.],
  *                           [ 5.,  5.],
  *                           [ 6.,  6.],
  *                           [ 7.,  7.],
  *                           [ 8.,  8.]]
  * 
  *    Note that you cannot concat x,y,z along dimension 1 since dimension
  *    0 is not the same for all the input arrays.
  * 
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
  *                          [ 4.,  4.,  7.,  7.],
  *                          [ 5.,  5.,  8.,  8.]]
  * 
  * 
  * 
  * Defined in src/operator/nn/concat.cc:L235
  * @return org.apache.mxnet.NDArray
  */
def Concat(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Compute *N*-D convolution on *(N+2)*-D input.
  * 
  * In the 2-D convolution, given input data with shape *(batch_size,
  * channel, height, width)*, the output is computed by
  * 
  * .. math::
  * 
  *    out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
  *    weight[i,j,:,:]
  * 
  * where :math:`\star` is the 2-D cross-correlation operator.
  * 
  * For general 2-D convolution, the shapes are
  * 
  * - **data**: *(batch_size, channel, height, width)*
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_height, out_width)*.
  * 
  * Define::
  * 
  *   f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
  * 
  * then we have::
  * 
  *   out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  *   out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
  * 
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
  * 
  * The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
  * width)*. We can choose other layouts such as *NHWC*.
  * 
  * If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
  * evenly into *g* parts along the channel axis, and also evenly split ``weight``
  * along the first dimension. Next compute the convolution on the *i*-th part of
  * the data with the *i*-th weight part. The output is obtained by concatenating all
  * the *g* results.
  * 
  * 1-D convolution does not have *height* dimension but only *width* in space.
  * 
  * - **data**: *(batch_size, channel, width)*
  * - **weight**: *(num_filter, channel, kernel[0])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_width)*.
  * 
  * 3-D convolution adds an additional *depth* dimension besides *height* and
  * *width*. The shapes are
  * 
  * - **data**: *(batch_size, channel, depth, height, width)*
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
  * 
  * Both ``weight`` and ``bias`` are learnable parameters.
  * 
  * There are other options to tune the performance.
  * 
  * - **cudnn_tune**: enable this option leads to higher startup time but may give
  *   faster speed. Options are
  * 
  *   - **off**: no tuning
  *   - **limited_workspace**:run test and pick the fastest algorithm that doesn't
  *     exceed workspace limit.
  *   - **fastest**: pick the fastest algorithm and ignore workspace limit.
  *   - **None** (default): the behavior is determined by environment variable
  *     ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
  *     (default), 2 for fastest.
  * 
  * - **workspace**: A large number leads to more (GPU) memory usage but may improve
  *   the performance.
  * 
  * 
  * 
  * Defined in src/operator/nn/convolution.cc:L456
  * @return org.apache.mxnet.NDArray
  */
def Convolution(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Compute *N*-D convolution on *(N+2)*-D input.
  * 
  * In the 2-D convolution, given input data with shape *(batch_size,
  * channel, height, width)*, the output is computed by
  * 
  * .. math::
  * 
  *    out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star
  *    weight[i,j,:,:]
  * 
  * where :math:`\star` is the 2-D cross-correlation operator.
  * 
  * For general 2-D convolution, the shapes are
  * 
  * - **data**: *(batch_size, channel, height, width)*
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_height, out_width)*.
  * 
  * Define::
  * 
  *   f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1
  * 
  * then we have::
  * 
  *   out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  *   out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])
  * 
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
  * 
  * The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,
  * width)*. We can choose other layouts such as *NHWC*.
  * 
  * If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
  * evenly into *g* parts along the channel axis, and also evenly split ``weight``
  * along the first dimension. Next compute the convolution on the *i*-th part of
  * the data with the *i*-th weight part. The output is obtained by concatenating all
  * the *g* results.
  * 
  * 1-D convolution does not have *height* dimension but only *width* in space.
  * 
  * - **data**: *(batch_size, channel, width)*
  * - **weight**: *(num_filter, channel, kernel[0])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_width)*.
  * 
  * 3-D convolution adds an additional *depth* dimension besides *height* and
  * *width*. The shapes are
  * 
  * - **data**: *(batch_size, channel, depth, height, width)*
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*
  * - **bias**: *(num_filter,)*
  * - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.
  * 
  * Both ``weight`` and ``bias`` are learnable parameters.
  * 
  * There are other options to tune the performance.
  * 
  * - **cudnn_tune**: enable this option leads to higher startup time but may give
  *   faster speed. Options are
  * 
  *   - **off**: no tuning
  *   - **limited_workspace**:run test and pick the fastest algorithm that doesn't
  *     exceed workspace limit.
  *   - **fastest**: pick the fastest algorithm and ignore workspace limit.
  *   - **None** (default): the behavior is determined by environment variable
  *     ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace
  *     (default), 2 for fastest.
  * 
  * - **workspace**: A large number leads to more (GPU) memory usage but may improve
  *   the performance.
  * 
  * 
  * 
  * Defined in src/operator/nn/convolution.cc:L456
  * @return org.apache.mxnet.NDArray
  */
def Convolution(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * This operator is DEPRECATED. Apply convolution to input then add a bias.
  * @return org.apache.mxnet.NDArray
  */
def Convolution_v1(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * This operator is DEPRECATED. Apply convolution to input then add a bias.
  * @return org.apache.mxnet.NDArray
  */
def Convolution_v1(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies correlation to inputs.
  * 
  * The correlation layer performs multiplicative patch comparisons between two feature maps.
  * 
  * Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,
  * the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.
  * 
  * For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and
  * :math:`x_{2}` in the second map is then defined as:
  * 
  * .. math::
  * 
  *    c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>
  * 
  * for a square patch of size :math:`K:=2k+1`.
  * 
  * Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other
  * data. For this reason, it has no training weights.
  * 
  * Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.
  * 
  * Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
  * by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
  * centered around :math:`x_{1}`.
  * 
  * The final output is defined by the following expression:
  * 
  * .. math::
  *   out[n, q, i, j] = c(x_{i, j}, x_{q})
  * 
  * where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.
  * 
  * 
  * Defined in src/operator/correlation.cc:L198
  * @return org.apache.mxnet.NDArray
  */
def Correlation(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies correlation to inputs.
  * 
  * The correlation layer performs multiplicative patch comparisons between two feature maps.
  * 
  * Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,
  * the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.
  * 
  * For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and
  * :math:`x_{2}` in the second map is then defined as:
  * 
  * .. math::
  * 
  *    c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)>
  * 
  * for a square patch of size :math:`K:=2k+1`.
  * 
  * Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other
  * data. For this reason, it has no training weights.
  * 
  * Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.
  * 
  * Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,
  * by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood
  * centered around :math:`x_{1}`.
  * 
  * The final output is defined by the following expression:
  * 
  * .. math::
  *   out[n, q, i, j] = c(x_{i, j}, x_{q})
  * 
  * where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.
  * 
  * 
  * Defined in src/operator/correlation.cc:L198
  * @return org.apache.mxnet.NDArray
  */
def Correlation(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * 
  * 
  * .. note:: `Crop` is deprecated. Use `slice` instead.
  * 
  * Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
  * with width and height of the second input symbol, i.e., with one input, we need h_w to
  * specify the crop height and width, otherwise the second input symbol's size will be used
  * 
  * 
  * Defined in src/operator/crop.cc:L50
  * @return org.apache.mxnet.NDArray
  */
def Crop(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * 
  * 
  * .. note:: `Crop` is deprecated. Use `slice` instead.
  * 
  * Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
  * with width and height of the second input symbol, i.e., with one input, we need h_w to
  * specify the crop height and width, otherwise the second input symbol's size will be used
  * 
  * 
  * Defined in src/operator/crop.cc:L50
  * @return org.apache.mxnet.NDArray
  */
def Crop(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Apply a custom operator implemented in a frontend language (like Python).
  * 
  * Custom operators should override required methods like `forward` and `backward`.
  * The custom operator must be registered before it can be used.
  * Please check the tutorial here: http://mxnet.io/faq/new_op.html.
  * 
  * 
  * 
  * Defined in src/operator/custom/custom.cc:L369
  * @return org.apache.mxnet.NDArray
  */
def Custom(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Apply a custom operator implemented in a frontend language (like Python).
  * 
  * Custom operators should override required methods like `forward` and `backward`.
  * The custom operator must be registered before it can be used.
  * Please check the tutorial here: http://mxnet.io/faq/new_op.html.
  * 
  * 
  * 
  * Defined in src/operator/custom/custom.cc:L369
  * @return org.apache.mxnet.NDArray
  */
def Custom(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.
  * @return org.apache.mxnet.NDArray
  */
def Deconvolution(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.
  * @return org.apache.mxnet.NDArray
  */
def Deconvolution(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies dropout operation to input array.
  * 
  * - During training, each element of the input is set to zero with probability p.
  *   The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  *   sum of the input unchanged.
  * 
  * - During testing, this operator does not change the input if mode is 'training'.
  *   If mode is 'always', the same computaion as during training will be applied.
  * 
  * Example::
  * 
  *   random.seed(998)
  *   input_array = array([[3., 0.5,  -0.5,  2., 7.],
  *                       [2., -0.4,   7.,  3., 0.2]])
  *   a = symbol.Variable('a')
  *   dropout = symbol.Dropout(a, p = 0.2)
  *   executor = dropout.simple_bind(a = input_array.shape)
  * 
  *   ## If training
  *   executor.forward(is_train = True, a = input_array)
  *   executor.outputs
  *   [[ 3.75   0.625 -0.     2.5    8.75 ]
  *    [ 2.5   -0.5    8.75   3.75   0.   ]]
  * 
  *   ## If testing
  *   executor.forward(is_train = False, a = input_array)
  *   executor.outputs
  *   [[ 3.     0.5   -0.5    2.     7.   ]
  *    [ 2.    -0.4    7.     3.     0.2  ]]
  * 
  * 
  * Defined in src/operator/nn/dropout.cc:L76
  * @return org.apache.mxnet.NDArray
  */
def Dropout(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies dropout operation to input array.
  * 
  * - During training, each element of the input is set to zero with probability p.
  *   The whole array is rescaled by :math:`1/(1-p)` to keep the expected
  *   sum of the input unchanged.
  * 
  * - During testing, this operator does not change the input if mode is 'training'.
  *   If mode is 'always', the same computaion as during training will be applied.
  * 
  * Example::
  * 
  *   random.seed(998)
  *   input_array = array([[3., 0.5,  -0.5,  2., 7.],
  *                       [2., -0.4,   7.,  3., 0.2]])
  *   a = symbol.Variable('a')
  *   dropout = symbol.Dropout(a, p = 0.2)
  *   executor = dropout.simple_bind(a = input_array.shape)
  * 
  *   ## If training
  *   executor.forward(is_train = True, a = input_array)
  *   executor.outputs
  *   [[ 3.75   0.625 -0.     2.5    8.75 ]
  *    [ 2.5   -0.5    8.75   3.75   0.   ]]
  * 
  *   ## If testing
  *   executor.forward(is_train = False, a = input_array)
  *   executor.outputs
  *   [[ 3.     0.5   -0.5    2.     7.   ]
  *    [ 2.    -0.4    7.     3.     0.2  ]]
  * 
  * 
  * Defined in src/operator/nn/dropout.cc:L76
  * @return org.apache.mxnet.NDArray
  */
def Dropout(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds all input arguments element-wise.
  * 
  * .. math::
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
  * 
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
  * 
  * The storage type of ``add_n`` output depends on storage types of inputs
  * 
  * - add_n(row_sparse, row_sparse, ..) = row_sparse
  * - otherwise, ``add_n`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_sum.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def ElementWiseSum(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds all input arguments element-wise.
  * 
  * .. math::
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
  * 
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
  * 
  * The storage type of ``add_n`` output depends on storage types of inputs
  * 
  * - add_n(row_sparse, row_sparse, ..) = row_sparse
  * - otherwise, ``add_n`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_sum.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def ElementWiseSum(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Maps integer indices to vector representations (embeddings).
  * 
  * This operator maps words to real-valued vectors in a high-dimensional space,
  * called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
  * For example, it has been noted that in the learned embedding spaces, similar words tend
  * to be close to each other and dissimilar words far apart.
  * 
  * For an input array of shape (d1, ..., dK),
  * the shape of an output array is (d1, ..., dK, output_dim).
  * All the input values should be integers in the range [0, input_dim).
  * 
  * If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
  * (ip0, op0).
  * 
  * By default, if any index mentioned is too large, it is replaced by the index that addresses
  * the last vector in an embedding matrix.
  * 
  * Examples::
  * 
  *   input_dim = 4
  *   output_dim = 5
  * 
  *   // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  *   y = [[  0.,   1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.,   9.],
  *        [ 10.,  11.,  12.,  13.,  14.],
  *        [ 15.,  16.,  17.,  18.,  19.]]
  * 
  *   // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  *   x = [[ 1.,  3.],
  *        [ 0.,  2.]]
  * 
  *   // Mapped input x to its vector representation y.
  *   Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
  *                             [ 15.,  16.,  17.,  18.,  19.]],
  * 
  *                            [[  0.,   1.,   2.,   3.,   4.],
  *                             [ 10.,  11.,  12.,  13.,  14.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L227
  * @return org.apache.mxnet.NDArray
  */
def Embedding(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Maps integer indices to vector representations (embeddings).
  * 
  * This operator maps words to real-valued vectors in a high-dimensional space,
  * called word embeddings. These embeddings can capture semantic and syntactic properties of the words.
  * For example, it has been noted that in the learned embedding spaces, similar words tend
  * to be close to each other and dissimilar words far apart.
  * 
  * For an input array of shape (d1, ..., dK),
  * the shape of an output array is (d1, ..., dK, output_dim).
  * All the input values should be integers in the range [0, input_dim).
  * 
  * If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be
  * (ip0, op0).
  * 
  * By default, if any index mentioned is too large, it is replaced by the index that addresses
  * the last vector in an embedding matrix.
  * 
  * Examples::
  * 
  *   input_dim = 4
  *   output_dim = 5
  * 
  *   // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)
  *   y = [[  0.,   1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.,   9.],
  *        [ 10.,  11.,  12.,  13.,  14.],
  *        [ 15.,  16.,  17.,  18.,  19.]]
  * 
  *   // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]
  *   x = [[ 1.,  3.],
  *        [ 0.,  2.]]
  * 
  *   // Mapped input x to its vector representation y.
  *   Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],
  *                             [ 15.,  16.,  17.,  18.,  19.]],
  * 
  *                            [[  0.,   1.,   2.,   3.,   4.],
  *                             [ 10.,  11.,  12.,  13.,  14.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L227
  * @return org.apache.mxnet.NDArray
  */
def Embedding(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.
  * 
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.
  * 
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
  * the input array into an output array of shape ``(d1, d2*...*dk)``.
  * 
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,
  * which behaves similar to mxnet.ndarray.reshape((-1,)).
  * 
  * Example::
  * 
  *     x = [[
  *         [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ],
  *     [    [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ]],
  * 
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L257
  * @return org.apache.mxnet.NDArray
  */
def Flatten(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.
  * 
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.
  * 
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
  * the input array into an output array of shape ``(d1, d2*...*dk)``.
  * 
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,
  * which behaves similar to mxnet.ndarray.reshape((-1,)).
  * 
  * Example::
  * 
  *     x = [[
  *         [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ],
  *     [    [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ]],
  * 
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L257
  * @return org.apache.mxnet.NDArray
  */
def Flatten(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a linear transformation: :math:`Y = XW^T + b`.
  * 
  * If ``flatten`` is set to be true, then the shapes are:
  * 
  * - **data**: `(batch_size, x1, x2, ..., xn)`
  * - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
  * - **bias**: `(num_hidden,)`
  * - **out**: `(batch_size, num_hidden)`
  * 
  * If ``flatten`` is set to be false, then the shapes are:
  * 
  * - **data**: `(x1, x2, ..., xn, input_dim)`
  * - **weight**: `(num_hidden, input_dim)`
  * - **bias**: `(num_hidden,)`
  * - **out**: `(x1, x2, ..., xn, num_hidden)`
  * 
  * The learnable parameters include both ``weight`` and ``bias``.
  * 
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
  * 
  * Note that the operator also supports forward computation with `row_sparse` weight and bias,
  * where the length of `weight.indices` and `bias.indices` must be equal to `num_hidden`.
  * This could be used for model inference with `row_sparse` weights trained with `SparseEmbedding`.
  * 
  * 
  * 
  * Defined in src/operator/nn/fully_connected.cc:L254
  * @return org.apache.mxnet.NDArray
  */
def FullyConnected(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a linear transformation: :math:`Y = XW^T + b`.
  * 
  * If ``flatten`` is set to be true, then the shapes are:
  * 
  * - **data**: `(batch_size, x1, x2, ..., xn)`
  * - **weight**: `(num_hidden, x1 * x2 * ... * xn)`
  * - **bias**: `(num_hidden,)`
  * - **out**: `(batch_size, num_hidden)`
  * 
  * If ``flatten`` is set to be false, then the shapes are:
  * 
  * - **data**: `(x1, x2, ..., xn, input_dim)`
  * - **weight**: `(num_hidden, input_dim)`
  * - **bias**: `(num_hidden,)`
  * - **out**: `(x1, x2, ..., xn, num_hidden)`
  * 
  * The learnable parameters include both ``weight`` and ``bias``.
  * 
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.
  * 
  * Note that the operator also supports forward computation with `row_sparse` weight and bias,
  * where the length of `weight.indices` and `bias.indices` must be equal to `num_hidden`.
  * This could be used for model inference with `row_sparse` weights trained with `SparseEmbedding`.
  * 
  * 
  * 
  * Defined in src/operator/nn/fully_connected.cc:L254
  * @return org.apache.mxnet.NDArray
  */
def FullyConnected(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Generates 2D sampling grid for bilinear sampling.
  * @return org.apache.mxnet.NDArray
  */
def GridGenerator(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Generates 2D sampling grid for bilinear sampling.
  * @return org.apache.mxnet.NDArray
  */
def GridGenerator(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Apply a sparse regularization to the output a sigmoid activation function.
  * @return org.apache.mxnet.NDArray
  */
def IdentityAttachKLSparseReg(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Apply a sparse regularization to the output a sigmoid activation function.
  * @return org.apache.mxnet.NDArray
  */
def IdentityAttachKLSparseReg(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies instance normalization to the n-dimensional input array.
  * 
  * This operator takes an n-dimensional input array where (n>2) and normalizes
  * the input using the following formula:
  * 
  * .. math::
  * 
  *   out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
  * 
  * This layer is similar to batch normalization layer (`BatchNorm`)
  * with two differences: first, the normalization is
  * carried out per example (instance), not over a batch. Second, the
  * same normalization is applied both at test and train time. This
  * operation is also known as `contrast normalization`.
  * 
  * If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
  * `gamma` and `beta` parameters must be vectors of shape [channel].
  * 
  * This implementation is based on paper:
  * 
  * .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
  *    D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
  * 
  * Examples::
  * 
  *   // Input of shape (2,1,2)
  *   x = [[[ 1.1,  2.2]],
  *        [[ 3.3,  4.4]]]
  * 
  *   // gamma parameter of length 1
  *   gamma = [1.5]
  * 
  *   // beta parameter of length 1
  *   beta = [0.5]
  * 
  *   // Instance normalization is calculated with the above formula
  *   InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
  *                                 [[-0.99752653,  1.99752724]]]
  * 
  * 
  * 
  * Defined in src/operator/instance_norm.cc:L95
  * @return org.apache.mxnet.NDArray
  */
def InstanceNorm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies instance normalization to the n-dimensional input array.
  * 
  * This operator takes an n-dimensional input array where (n>2) and normalizes
  * the input using the following formula:
  * 
  * .. math::
  * 
  *   out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta
  * 
  * This layer is similar to batch normalization layer (`BatchNorm`)
  * with two differences: first, the normalization is
  * carried out per example (instance), not over a batch. Second, the
  * same normalization is applied both at test and train time. This
  * operation is also known as `contrast normalization`.
  * 
  * If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],
  * `gamma` and `beta` parameters must be vectors of shape [channel].
  * 
  * This implementation is based on paper:
  * 
  * .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,
  *    D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).
  * 
  * Examples::
  * 
  *   // Input of shape (2,1,2)
  *   x = [[[ 1.1,  2.2]],
  *        [[ 3.3,  4.4]]]
  * 
  *   // gamma parameter of length 1
  *   gamma = [1.5]
  * 
  *   // beta parameter of length 1
  *   beta = [0.5]
  * 
  *   // Instance normalization is calculated with the above formula
  *   InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],
  *                                 [[-0.99752653,  1.99752724]]]
  * 
  * 
  * 
  * Defined in src/operator/instance_norm.cc:L95
  * @return org.apache.mxnet.NDArray
  */
def InstanceNorm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Normalize the input array using the L2 norm.
  * 
  * For 1-D NDArray, it computes::
  * 
  *   out = data / sqrt(sum(data ** 2) + eps)
  * 
  * For N-D NDArray, if the input array has shape (N, N, ..., N),
  * 
  * with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
  * array by its L2 norm.::
  * 
  *   for i in 0...N
  *     out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
  * 
  * with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::
  * 
  *   for i in 0...N
  *     out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
  * 
  * with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
  * in the array by its L2 norm.::
  * 
  *   for dim in 2...N
  *     for i in 0...N
  *       out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
  *           -dim-
  * 
  * Example::
  * 
  *   x = [[[1,2],
  *         [3,4]],
  *        [[2,2],
  *         [5,6]]]
  * 
  *   L2Normalization(x, mode='instance')
  *   =[[[ 0.18257418  0.36514837]
  *      [ 0.54772252  0.73029673]]
  *     [[ 0.24077171  0.24077171]
  *      [ 0.60192931  0.72231513]]]
  * 
  *   L2Normalization(x, mode='channel')
  *   =[[[ 0.31622776  0.44721359]
  *      [ 0.94868326  0.89442718]]
  *     [[ 0.37139067  0.31622776]
  *      [ 0.92847669  0.94868326]]]
  * 
  *   L2Normalization(x, mode='spatial')
  *   =[[[ 0.44721359  0.89442718]
  *      [ 0.60000002  0.80000001]]
  *     [[ 0.70710677  0.70710677]
  *      [ 0.6401844   0.76822126]]]
  * 
  * 
  * 
  * Defined in src/operator/l2_normalization.cc:L98
  * @return org.apache.mxnet.NDArray
  */
def L2Normalization(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Normalize the input array using the L2 norm.
  * 
  * For 1-D NDArray, it computes::
  * 
  *   out = data / sqrt(sum(data ** 2) + eps)
  * 
  * For N-D NDArray, if the input array has shape (N, N, ..., N),
  * 
  * with ``mode`` = ``instance``, it normalizes each instance in the multidimensional
  * array by its L2 norm.::
  * 
  *   for i in 0...N
  *     out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)
  * 
  * with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::
  * 
  *   for i in 0...N
  *     out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)
  * 
  * with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position
  * in the array by its L2 norm.::
  * 
  *   for dim in 2...N
  *     for i in 0...N
  *       out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)
  *           -dim-
  * 
  * Example::
  * 
  *   x = [[[1,2],
  *         [3,4]],
  *        [[2,2],
  *         [5,6]]]
  * 
  *   L2Normalization(x, mode='instance')
  *   =[[[ 0.18257418  0.36514837]
  *      [ 0.54772252  0.73029673]]
  *     [[ 0.24077171  0.24077171]
  *      [ 0.60192931  0.72231513]]]
  * 
  *   L2Normalization(x, mode='channel')
  *   =[[[ 0.31622776  0.44721359]
  *      [ 0.94868326  0.89442718]]
  *     [[ 0.37139067  0.31622776]
  *      [ 0.92847669  0.94868326]]]
  * 
  *   L2Normalization(x, mode='spatial')
  *   =[[[ 0.44721359  0.89442718]
  *      [ 0.60000002  0.80000001]]
  *     [[ 0.70710677  0.70710677]
  *      [ 0.6401844   0.76822126]]]
  * 
  * 
  * 
  * Defined in src/operator/l2_normalization.cc:L98
  * @return org.apache.mxnet.NDArray
  */
def L2Normalization(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies local response normalization to the input.
  * 
  * The local response normalization layer performs "lateral inhibition" by normalizing
  * over local input regions.
  * 
  * If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
  * :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
  * activity :math:`b_{x,y}^{i}` is given by the expression:
  * 
  * .. math::
  *    b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}
  * 
  * where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
  * number of kernels in the layer.
  * 
  * 
  * 
  * Defined in src/operator/nn/lrn.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def LRN(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies local response normalization to the input.
  * 
  * The local response normalization layer performs "lateral inhibition" by normalizing
  * over local input regions.
  * 
  * If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position
  * :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized
  * activity :math:`b_{x,y}^{i}` is given by the expression:
  * 
  * .. math::
  *    b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \alpha \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}
  * 
  * where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total
  * number of kernels in the layer.
  * 
  * 
  * 
  * Defined in src/operator/nn/lrn.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def LRN(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Layer normalization.
  * 
  * Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis and then 
  * compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters.
  * 
  * Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * ``data_std``. Note that no gradient will be passed through these two outputs.
  * 
  * The parameter ``axis`` specifies which axis of the input shape denotes
  * the 'channel' (separately normalized groups).  The default is -1, which sets the channel
  * axis to be the last item in the input shape.
  * 
  * 
  * 
  * Defined in src/operator/nn/layer_norm.cc:L94
  * @return org.apache.mxnet.NDArray
  */
def LayerNorm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Layer normalization.
  * 
  * Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as
  * well as offset ``beta``.
  * 
  * Assume the input has more than one dimension and we normalize along axis 1.
  * We first compute the mean and variance along this axis and then 
  * compute the normalized output, which has the same shape as input, as following:
  * 
  * .. math::
  * 
  *   out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta
  * 
  * Both ``gamma`` and ``beta`` are learnable parameters.
  * 
  * Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.
  * 
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and
  * ``data_std``. Note that no gradient will be passed through these two outputs.
  * 
  * The parameter ``axis`` specifies which axis of the input shape denotes
  * the 'channel' (separately normalized groups).  The default is -1, which sets the channel
  * axis to be the last item in the input shape.
  * 
  * 
  * 
  * Defined in src/operator/nn/layer_norm.cc:L94
  * @return org.apache.mxnet.NDArray
  */
def LayerNorm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies Leaky rectified linear unit activation element-wise to the input.
  * 
  * Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
  * when the input is negative and has a slope of one when input is positive.
  * 
  * The following modified ReLU Activation functions are supported:
  * 
  * - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
  * - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
  * - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.
  * - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from
  *   *[lower_bound, upper_bound)* for training, while fixed to be
  *   *(lower_bound+upper_bound)/2* for inference.
  * 
  * 
  * 
  * Defined in src/operator/leaky_relu.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def LeakyReLU(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies Leaky rectified linear unit activation element-wise to the input.
  * 
  * Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`
  * when the input is negative and has a slope of one when input is positive.
  * 
  * The following modified ReLU Activation functions are supported:
  * 
  * - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`
  * - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`
  * - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.
  * - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from
  *   *[lower_bound, upper_bound)* for training, while fixed to be
  *   *(lower_bound+upper_bound)/2* for inference.
  * 
  * 
  * 
  * Defined in src/operator/leaky_relu.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def LeakyReLU(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes and optimizes for squared loss during backward propagation.
  * Just outputs ``data`` during forward propagation.
  * 
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
  * then the squared loss estimated over :math:`n` samples is defined as
  * 
  * :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`
  * 
  * .. note::
  *    Use the LinearRegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - LinearRegressionOutput(default, default) = default
  * - LinearRegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def LinearRegressionOutput(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes and optimizes for squared loss during backward propagation.
  * Just outputs ``data`` during forward propagation.
  * 
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
  * then the squared loss estimated over :math:`n` samples is defined as
  * 
  * :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`
  * 
  * .. note::
  *    Use the LinearRegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - LinearRegressionOutput(default, default) = default
  * - LinearRegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def LinearRegressionOutput(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a logistic function to the input.
  * 
  * The logistic function, also known as the sigmoid function, is computed as
  * :math:`\frac{1}{1+exp(-\textbf{x})}`.
  * 
  * Commonly, the sigmoid is used to squash the real-valued output of a linear model
  * :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.
  * It is suitable for binary classification or probability prediction tasks.
  * 
  * .. note::
  *    Use the LogisticRegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - LogisticRegressionOutput(default, default) = default
  * - LogisticRegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L148
  * @return org.apache.mxnet.NDArray
  */
def LogisticRegressionOutput(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a logistic function to the input.
  * 
  * The logistic function, also known as the sigmoid function, is computed as
  * :math:`\frac{1}{1+exp(-\textbf{x})}`.
  * 
  * Commonly, the sigmoid is used to squash the real-valued output of a linear model
  * :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.
  * It is suitable for binary classification or probability prediction tasks.
  * 
  * .. note::
  *    Use the LogisticRegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - LogisticRegressionOutput(default, default) = default
  * - LogisticRegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L148
  * @return org.apache.mxnet.NDArray
  */
def LogisticRegressionOutput(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes mean absolute error of the input.
  * 
  * MAE is a risk metric corresponding to the expected value of the absolute error.
  * 
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
  * then the mean absolute error (MAE) estimated over :math:`n` samples is defined as
  * 
  * :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`
  * 
  * .. note::
  *    Use the MAERegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - MAERegressionOutput(default, default) = default
  * - MAERegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L120
  * @return org.apache.mxnet.NDArray
  */
def MAERegressionOutput(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes mean absolute error of the input.
  * 
  * MAE is a risk metric corresponding to the expected value of the absolute error.
  * 
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
  * then the mean absolute error (MAE) estimated over :math:`n` samples is defined as
  * 
  * :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`
  * 
  * .. note::
  *    Use the MAERegressionOutput as the final output layer of a net.
  * 
  * The storage type of ``label`` can be ``default`` or ``csr``
  * 
  * - MAERegressionOutput(default, default) = default
  * - MAERegressionOutput(default, csr) = default
  * 
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.
  * 
  * 
  * 
  * Defined in src/operator/regression_output.cc:L120
  * @return org.apache.mxnet.NDArray
  */
def MAERegressionOutput(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Make your own loss function in network construction.
  * 
  * This operator accepts a customized loss function symbol as a terminal loss and
  * the symbol should be an operator with no backward dependency.
  * The output of this function is the gradient of loss with respect to the input data.
  * 
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::
  * 
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  *   loss = MakeLoss(cross_entropy)
  * 
  * We will need to use ``MakeLoss`` when we are creating our own loss function or we want to
  * combine multiple loss functions. Also we may want to stop some variables' gradients
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
  * 
  * In addition, we can give a scale to the loss by setting ``grad_scale``,
  * so that the gradient of the loss will be rescaled in the backpropagation.
  * 
  * .. note:: This operator should be used as a Symbol instead of NDArray.
  * 
  * 
  * 
  * Defined in src/operator/make_loss.cc:L71
  * @return org.apache.mxnet.NDArray
  */
def MakeLoss(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Make your own loss function in network construction.
  * 
  * This operator accepts a customized loss function symbol as a terminal loss and
  * the symbol should be an operator with no backward dependency.
  * The output of this function is the gradient of loss with respect to the input data.
  * 
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::
  * 
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  *   loss = MakeLoss(cross_entropy)
  * 
  * We will need to use ``MakeLoss`` when we are creating our own loss function or we want to
  * combine multiple loss functions. Also we may want to stop some variables' gradients
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
  * 
  * In addition, we can give a scale to the loss by setting ``grad_scale``,
  * so that the gradient of the loss will be rescaled in the backpropagation.
  * 
  * .. note:: This operator should be used as a Symbol instead of NDArray.
  * 
  * 
  * 
  * Defined in src/operator/make_loss.cc:L71
  * @return org.apache.mxnet.NDArray
  */
def MakeLoss(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pads an input array with a constant or edge values of the array.
  * 
  * .. note:: `Pad` is deprecated. Use `pad` instead.
  * 
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
  * 
  * This operation pads an input array with either a `constant_value` or edge values
  * along each axis of the input array. The amount of padding is specified by `pad_width`.
  * 
  * `pad_width` is a tuple of integer padding widths for each axis of the format
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
  * where ``N`` is the number of dimensions of the array.
  * 
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
  * to add before and after the elements of the array along dimension ``N``.
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
  * ``after_2`` must be 0.
  * 
  * Example::
  * 
  *    x = [[[[  1.   2.   3.]
  *           [  4.   5.   6.]]
  * 
  *          [[  7.   8.   9.]
  *           [ 10.  11.  12.]]]
  * 
  * 
  *         [[[ 11.  12.  13.]
  *           [ 14.  15.  16.]]
  * 
  *          [[ 17.  18.  19.]
  *           [ 20.  21.  22.]]]]
  * 
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  1.   1.   2.   3.   3.]
  *             [  1.   1.   2.   3.   3.]
  *             [  4.   4.   5.   6.   6.]
  *             [  4.   4.   5.   6.   6.]]
  * 
  *            [[  7.   7.   8.   9.   9.]
  *             [  7.   7.   8.   9.   9.]
  *             [ 10.  10.  11.  12.  12.]
  *             [ 10.  10.  11.  12.  12.]]]
  * 
  * 
  *           [[[ 11.  11.  12.  13.  13.]
  *             [ 11.  11.  12.  13.  13.]
  *             [ 14.  14.  15.  16.  16.]
  *             [ 14.  14.  15.  16.  16.]]
  * 
  *            [[ 17.  17.  18.  19.  19.]
  *             [ 17.  17.  18.  19.  19.]
  *             [ 20.  20.  21.  22.  22.]
  *             [ 20.  20.  21.  22.  22.]]]]
  * 
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  0.   0.   0.   0.   0.]
  *             [  0.   1.   2.   3.   0.]
  *             [  0.   4.   5.   6.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.   7.   8.   9.   0.]
  *             [  0.  10.  11.  12.   0.]
  *             [  0.   0.   0.   0.   0.]]]
  * 
  * 
  *           [[[  0.   0.   0.   0.   0.]
  *             [  0.  11.  12.  13.   0.]
  *             [  0.  14.  15.  16.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.  17.  18.  19.   0.]
  *             [  0.  20.  21.  22.   0.]
  *             [  0.   0.   0.   0.   0.]]]]
  * 
  * 
  * 
  * 
  * Defined in src/operator/pad.cc:L766
  * @return org.apache.mxnet.NDArray
  */
def Pad(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pads an input array with a constant or edge values of the array.
  * 
  * .. note:: `Pad` is deprecated. Use `pad` instead.
  * 
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
  * 
  * This operation pads an input array with either a `constant_value` or edge values
  * along each axis of the input array. The amount of padding is specified by `pad_width`.
  * 
  * `pad_width` is a tuple of integer padding widths for each axis of the format
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
  * where ``N`` is the number of dimensions of the array.
  * 
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
  * to add before and after the elements of the array along dimension ``N``.
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
  * ``after_2`` must be 0.
  * 
  * Example::
  * 
  *    x = [[[[  1.   2.   3.]
  *           [  4.   5.   6.]]
  * 
  *          [[  7.   8.   9.]
  *           [ 10.  11.  12.]]]
  * 
  * 
  *         [[[ 11.  12.  13.]
  *           [ 14.  15.  16.]]
  * 
  *          [[ 17.  18.  19.]
  *           [ 20.  21.  22.]]]]
  * 
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  1.   1.   2.   3.   3.]
  *             [  1.   1.   2.   3.   3.]
  *             [  4.   4.   5.   6.   6.]
  *             [  4.   4.   5.   6.   6.]]
  * 
  *            [[  7.   7.   8.   9.   9.]
  *             [  7.   7.   8.   9.   9.]
  *             [ 10.  10.  11.  12.  12.]
  *             [ 10.  10.  11.  12.  12.]]]
  * 
  * 
  *           [[[ 11.  11.  12.  13.  13.]
  *             [ 11.  11.  12.  13.  13.]
  *             [ 14.  14.  15.  16.  16.]
  *             [ 14.  14.  15.  16.  16.]]
  * 
  *            [[ 17.  17.  18.  19.  19.]
  *             [ 17.  17.  18.  19.  19.]
  *             [ 20.  20.  21.  22.  22.]
  *             [ 20.  20.  21.  22.  22.]]]]
  * 
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  0.   0.   0.   0.   0.]
  *             [  0.   1.   2.   3.   0.]
  *             [  0.   4.   5.   6.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.   7.   8.   9.   0.]
  *             [  0.  10.  11.  12.   0.]
  *             [  0.   0.   0.   0.   0.]]]
  * 
  * 
  *           [[[  0.   0.   0.   0.   0.]
  *             [  0.  11.  12.  13.   0.]
  *             [  0.  14.  15.  16.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.  17.  18.  19.   0.]
  *             [  0.  20.  21.  22.   0.]
  *             [  0.   0.   0.   0.   0.]]]]
  * 
  * 
  * 
  * 
  * Defined in src/operator/pad.cc:L766
  * @return org.apache.mxnet.NDArray
  */
def Pad(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pooling operator for input and output data type of int8.
  * The input and output data comes with min and max thresholds for quantizing
  * the float32 data into int8.
  * 
  * .. Note::
  *     This operator only supports forward propogation. DO NOT use it in training.
  *     This operator only supports `pool_type` of `avg` or `max`.
  * 
  * Defined in src/operator/quantization/quantized_pooling.cc:L127
  * @return org.apache.mxnet.NDArray
  */
def Pooling(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pooling operator for input and output data type of int8.
  * The input and output data comes with min and max thresholds for quantizing
  * the float32 data into int8.
  * 
  * .. Note::
  *     This operator only supports forward propogation. DO NOT use it in training.
  *     This operator only supports `pool_type` of `avg` or `max`.
  * 
  * Defined in src/operator/quantization/quantized_pooling.cc:L127
  * @return org.apache.mxnet.NDArray
  */
def Pooling(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * This operator is DEPRECATED.
  * Perform pooling on the input.
  * 
  * The shapes for 2-D pooling is
  * 
  * - **data**: *(batch_size, channel, height, width)*
  * - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
  * 
  *     out_height = f(height, kernel[0], pad[0], stride[0])
  *     out_width = f(width, kernel[1], pad[1], stride[1])
  * 
  * The definition of *f* depends on ``pooling_convention``, which has two options:
  * 
  * - **valid** (default)::
  * 
  *     f(x, k, p, s) = floor((x+2*p-k)/s)+1
  * 
  * - **full**, which is compatible with Caffe::
  * 
  *     f(x, k, p, s) = ceil((x+2*p-k)/s)+1
  * 
  * But ``global_pool`` is set to be true, then do a global pooling, namely reset
  * ``kernel=(height, width)``.
  * 
  * Three pooling options are supported by ``pool_type``:
  * 
  * - **avg**: average pooling
  * - **max**: max pooling
  * - **sum**: sum pooling
  * 
  * 1-D pooling is special case of 2-D pooling with *weight=1* and
  * *kernel[1]=1*.
  * 
  * For 3-D pooling, an additional *depth* dimension is added before
  * *height*. Namely the input data will have shape *(batch_size, channel, depth,
  * height, width)*.
  * 
  * 
  * 
  * Defined in src/operator/pooling_v1.cc:L104
  * @return org.apache.mxnet.NDArray
  */
def Pooling_v1(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * This operator is DEPRECATED.
  * Perform pooling on the input.
  * 
  * The shapes for 2-D pooling is
  * 
  * - **data**: *(batch_size, channel, height, width)*
  * - **out**: *(batch_size, num_filter, out_height, out_width)*, with::
  * 
  *     out_height = f(height, kernel[0], pad[0], stride[0])
  *     out_width = f(width, kernel[1], pad[1], stride[1])
  * 
  * The definition of *f* depends on ``pooling_convention``, which has two options:
  * 
  * - **valid** (default)::
  * 
  *     f(x, k, p, s) = floor((x+2*p-k)/s)+1
  * 
  * - **full**, which is compatible with Caffe::
  * 
  *     f(x, k, p, s) = ceil((x+2*p-k)/s)+1
  * 
  * But ``global_pool`` is set to be true, then do a global pooling, namely reset
  * ``kernel=(height, width)``.
  * 
  * Three pooling options are supported by ``pool_type``:
  * 
  * - **avg**: average pooling
  * - **max**: max pooling
  * - **sum**: sum pooling
  * 
  * 1-D pooling is special case of 2-D pooling with *weight=1* and
  * *kernel[1]=1*.
  * 
  * For 3-D pooling, an additional *depth* dimension is added before
  * *height*. Namely the input data will have shape *(batch_size, channel, depth,
  * height, width)*.
  * 
  * 
  * 
  * Defined in src/operator/pooling_v1.cc:L104
  * @return org.apache.mxnet.NDArray
  */
def Pooling_v1(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a recurrent layer to input.
  * @return org.apache.mxnet.NDArray
  */
def RNN(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a recurrent layer to input.
  * @return org.apache.mxnet.NDArray
  */
def RNN(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs region of interest(ROI) pooling on the input array.
  * 
  * ROI pooling is a variant of a max pooling layer, in which the output size is fixed and
  * region of interest is a parameter. Its purpose is to perform max pooling on the inputs
  * of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net
  * layer mostly used in training a `Fast R-CNN` network for object detection.
  * 
  * This operator takes a 4D feature map as an input array and region proposals as `rois`,
  * then it pools over sub-regions of input and produces a fixed-sized output array
  * regardless of the ROI size.
  * 
  * To crop the feature map accordingly, you can resize the bounding box coordinates
  * by changing the parameters `rois` and `spatial_scale`.
  * 
  * The cropped feature maps are pooled by standard max pooling operation to a fixed size output
  * indicated by a `pooled_size` parameter. batch_size will change to the number of region
  * bounding boxes after `ROIPooling`.
  * 
  * The size of each region of interest doesn't have to be perfectly divisible by
  * the number of pooling sections(`pooled_size`).
  * 
  * Example::
  * 
  *   x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
  *          [  6.,   7.,   8.,   9.,  10.,  11.],
  *          [ 12.,  13.,  14.,  15.,  16.,  17.],
  *          [ 18.,  19.,  20.,  21.,  22.,  23.],
  *          [ 24.,  25.,  26.,  27.,  28.,  29.],
  *          [ 30.,  31.,  32.,  33.,  34.,  35.],
  *          [ 36.,  37.,  38.,  39.,  40.,  41.],
  *          [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
  * 
  *   // region of interest i.e. bounding box coordinates.
  *   y = [[0,0,0,4,4]]
  * 
  *   // returns array of shape (2,2) according to the given roi with max pooling.
  *   ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
  *                                     [ 26.,  28.]]]]
  * 
  *   // region of interest is changed due to the change in `spacial_scale` parameter.
  *   ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
  *                                     [ 19.,  21.]]]]
  * 
  * 
  * 
  * Defined in src/operator/roi_pooling.cc:L295
  * @return org.apache.mxnet.NDArray
  */
def ROIPooling(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs region of interest(ROI) pooling on the input array.
  * 
  * ROI pooling is a variant of a max pooling layer, in which the output size is fixed and
  * region of interest is a parameter. Its purpose is to perform max pooling on the inputs
  * of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net
  * layer mostly used in training a `Fast R-CNN` network for object detection.
  * 
  * This operator takes a 4D feature map as an input array and region proposals as `rois`,
  * then it pools over sub-regions of input and produces a fixed-sized output array
  * regardless of the ROI size.
  * 
  * To crop the feature map accordingly, you can resize the bounding box coordinates
  * by changing the parameters `rois` and `spatial_scale`.
  * 
  * The cropped feature maps are pooled by standard max pooling operation to a fixed size output
  * indicated by a `pooled_size` parameter. batch_size will change to the number of region
  * bounding boxes after `ROIPooling`.
  * 
  * The size of each region of interest doesn't have to be perfectly divisible by
  * the number of pooling sections(`pooled_size`).
  * 
  * Example::
  * 
  *   x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],
  *          [  6.,   7.,   8.,   9.,  10.,  11.],
  *          [ 12.,  13.,  14.,  15.,  16.,  17.],
  *          [ 18.,  19.,  20.,  21.,  22.,  23.],
  *          [ 24.,  25.,  26.,  27.,  28.,  29.],
  *          [ 30.,  31.,  32.,  33.,  34.,  35.],
  *          [ 36.,  37.,  38.,  39.,  40.,  41.],
  *          [ 42.,  43.,  44.,  45.,  46.,  47.]]]]
  * 
  *   // region of interest i.e. bounding box coordinates.
  *   y = [[0,0,0,4,4]]
  * 
  *   // returns array of shape (2,2) according to the given roi with max pooling.
  *   ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],
  *                                     [ 26.,  28.]]]]
  * 
  *   // region of interest is changed due to the change in `spacial_scale` parameter.
  *   ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],
  *                                     [ 19.,  21.]]]]
  * 
  * 
  * 
  * Defined in src/operator/roi_pooling.cc:L295
  * @return org.apache.mxnet.NDArray
  */
def ROIPooling(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshapes the input array.
  * 
  * .. note:: ``Reshape`` is deprecated, use ``reshape``
  * 
  * Given an array and a shape, this function returns a copy of the array in the new shape.
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.
  * 
  * Example::
  * 
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
  * 
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:
  * 
  * - ``0``  copy this dimension from the input to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
  * 
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  *   keeping the size of the new array same as that of the input array.
  *   At most one dimension of shape can be -1.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)
  * 
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
  * 
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
  * 
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
  * 
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.
  * 
  *   Example::
  * 
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
  *   - with reverse=1, output shape will be (50,4).
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L167
  * @return org.apache.mxnet.NDArray
  */
def Reshape(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshapes the input array.
  * 
  * .. note:: ``Reshape`` is deprecated, use ``reshape``
  * 
  * Given an array and a shape, this function returns a copy of the array in the new shape.
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.
  * 
  * Example::
  * 
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
  * 
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:
  * 
  * - ``0``  copy this dimension from the input to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
  * 
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  *   keeping the size of the new array same as that of the input array.
  *   At most one dimension of shape can be -1.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)
  * 
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
  * 
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
  * 
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
  * 
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.
  * 
  *   Example::
  * 
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
  *   - with reverse=1, output shape will be (50,4).
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L167
  * @return org.apache.mxnet.NDArray
  */
def Reshape(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes support vector machine based transformation of the input.
  * 
  * This tutorial demonstrates using SVM as output layer for classification instead of softmax:
  * https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
  * @return org.apache.mxnet.NDArray
  */
def SVMOutput(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes support vector machine based transformation of the input.
  * 
  * This tutorial demonstrates using SVM as output layer for classification instead of softmax:
  * https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
  * @return org.apache.mxnet.NDArray
  */
def SVMOutput(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes the last element of a sequence.
  * 
  * This function takes an n-dimensional input array of the form
  * [max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array
  * of the form [batch_size, other_feature_dims].
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be
  * an input array of positive ints of dimension [batch_size]. To use this parameter,
  * set `use_sequence_length` to `True`, otherwise each example in the batch is assumed
  * to have the max sequence length.
  * 
  * .. note:: Alternatively, you can also use `take` operator.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.],
  *          [  7.,   8.,   9.]],
  * 
  *         [[ 10.,   11.,   12.],
  *          [ 13.,   14.,   15.],
  *          [ 16.,   17.,   18.]],
  * 
  *         [[  19.,   20.,   21.],
  *          [  22.,   23.,   24.],
  *          [  25.,   26.,   27.]]]
  * 
  *    // returns last sequence when sequence_length parameter is not used
  *    SequenceLast(x) = [[  19.,   20.,   21.],
  *                       [  22.,   23.,   24.],
  *                       [  25.,   26.,   27.]]
  * 
  *    // sequence_length is used
  *    SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
  *             [[  1.,   2.,   3.],
  *              [  4.,   5.,   6.],
  *              [  7.,   8.,   9.]]
  * 
  *    // sequence_length is used
  *    SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
  *             [[  1.,    2.,   3.],
  *              [  13.,  14.,  15.],
  *              [  25.,  26.,  27.]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_last.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def SequenceLast(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes the last element of a sequence.
  * 
  * This function takes an n-dimensional input array of the form
  * [max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array
  * of the form [batch_size, other_feature_dims].
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be
  * an input array of positive ints of dimension [batch_size]. To use this parameter,
  * set `use_sequence_length` to `True`, otherwise each example in the batch is assumed
  * to have the max sequence length.
  * 
  * .. note:: Alternatively, you can also use `take` operator.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.],
  *          [  7.,   8.,   9.]],
  * 
  *         [[ 10.,   11.,   12.],
  *          [ 13.,   14.,   15.],
  *          [ 16.,   17.,   18.]],
  * 
  *         [[  19.,   20.,   21.],
  *          [  22.,   23.,   24.],
  *          [  25.,   26.,   27.]]]
  * 
  *    // returns last sequence when sequence_length parameter is not used
  *    SequenceLast(x) = [[  19.,   20.,   21.],
  *                       [  22.,   23.,   24.],
  *                       [  25.,   26.,   27.]]
  * 
  *    // sequence_length is used
  *    SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =
  *             [[  1.,   2.,   3.],
  *              [  4.,   5.,   6.],
  *              [  7.,   8.,   9.]]
  * 
  *    // sequence_length is used
  *    SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =
  *             [[  1.,    2.,   3.],
  *              [  13.,  14.,  15.],
  *              [  25.,  26.,  27.]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_last.cc:L92
  * @return org.apache.mxnet.NDArray
  */
def SequenceLast(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Sets all elements outside the sequence to a constant value.
  * 
  * This function takes an n-dimensional input array of the form
  * [max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`
  * should be an input array of positive ints of dimension [batch_size].
  * To use this parameter, set `use_sequence_length` to `True`,
  * otherwise each example in the batch is assumed to have the max sequence length and
  * this operator works as the `identity` operator.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.]],
  * 
  *         [[  7.,   8.,   9.],
  *          [ 10.,  11.,  12.]],
  * 
  *         [[ 13.,  14.,   15.],
  *          [ 16.,  17.,   18.]]]
  * 
  *    // Batch 1
  *    B1 = [[  1.,   2.,   3.],
  *          [  7.,   8.,   9.],
  *          [ 13.,  14.,  15.]]
  * 
  *    // Batch 2
  *    B2 = [[  4.,   5.,   6.],
  *          [ 10.,  11.,  12.],
  *          [ 16.,  17.,  18.]]
  * 
  *    // works as identity operator when sequence_length parameter is not used
  *    SequenceMask(x) = [[[  1.,   2.,   3.],
  *                        [  4.,   5.,   6.]],
  * 
  *                       [[  7.,   8.,   9.],
  *                        [ 10.,  11.,  12.]],
  * 
  *                       [[ 13.,  14.,   15.],
  *                        [ 16.,  17.,   18.]]]
  * 
  *    // sequence_length [1,1] means 1 of each batch will be kept
  *    // and other rows are masked with default mask value = 0
  *    SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
  *                 [[[  1.,   2.,   3.],
  *                   [  4.,   5.,   6.]],
  * 
  *                  [[  0.,   0.,   0.],
  *                   [  0.,   0.,   0.]],
  * 
  *                  [[  0.,   0.,   0.],
  *                   [  0.,   0.,   0.]]]
  * 
  *    // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
  *    // and other rows are masked with value = 1
  *    SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
  *                 [[[  1.,   2.,   3.],
  *                   [  4.,   5.,   6.]],
  * 
  *                  [[  7.,   8.,   9.],
  *                   [  10.,  11.,  12.]],
  * 
  *                  [[   1.,   1.,   1.],
  *                   [  16.,  17.,  18.]]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_mask.cc:L114
  * @return org.apache.mxnet.NDArray
  */
def SequenceMask(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Sets all elements outside the sequence to a constant value.
  * 
  * This function takes an n-dimensional input array of the form
  * [max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`
  * should be an input array of positive ints of dimension [batch_size].
  * To use this parameter, set `use_sequence_length` to `True`,
  * otherwise each example in the batch is assumed to have the max sequence length and
  * this operator works as the `identity` operator.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.]],
  * 
  *         [[  7.,   8.,   9.],
  *          [ 10.,  11.,  12.]],
  * 
  *         [[ 13.,  14.,   15.],
  *          [ 16.,  17.,   18.]]]
  * 
  *    // Batch 1
  *    B1 = [[  1.,   2.,   3.],
  *          [  7.,   8.,   9.],
  *          [ 13.,  14.,  15.]]
  * 
  *    // Batch 2
  *    B2 = [[  4.,   5.,   6.],
  *          [ 10.,  11.,  12.],
  *          [ 16.,  17.,  18.]]
  * 
  *    // works as identity operator when sequence_length parameter is not used
  *    SequenceMask(x) = [[[  1.,   2.,   3.],
  *                        [  4.,   5.,   6.]],
  * 
  *                       [[  7.,   8.,   9.],
  *                        [ 10.,  11.,  12.]],
  * 
  *                       [[ 13.,  14.,   15.],
  *                        [ 16.,  17.,   18.]]]
  * 
  *    // sequence_length [1,1] means 1 of each batch will be kept
  *    // and other rows are masked with default mask value = 0
  *    SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =
  *                 [[[  1.,   2.,   3.],
  *                   [  4.,   5.,   6.]],
  * 
  *                  [[  0.,   0.,   0.],
  *                   [  0.,   0.,   0.]],
  * 
  *                  [[  0.,   0.,   0.],
  *                   [  0.,   0.,   0.]]]
  * 
  *    // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept
  *    // and other rows are masked with value = 1
  *    SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =
  *                 [[[  1.,   2.,   3.],
  *                   [  4.,   5.,   6.]],
  * 
  *                  [[  7.,   8.,   9.],
  *                   [  10.,  11.,  12.]],
  * 
  *                  [[   1.,   1.,   1.],
  *                   [  16.,  17.,  18.]]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_mask.cc:L114
  * @return org.apache.mxnet.NDArray
  */
def SequenceMask(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the elements of each sequence.
  * 
  * This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]
  * and returns an array of the same shape.
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences.
  * `sequence_length` should be an input array of positive ints of dimension [batch_size].
  * To use this parameter, set `use_sequence_length` to `True`,
  * otherwise each example in the batch is assumed to have the max sequence length.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.]],
  * 
  *         [[  7.,   8.,   9.],
  *          [ 10.,  11.,  12.]],
  * 
  *         [[ 13.,  14.,   15.],
  *          [ 16.,  17.,   18.]]]
  * 
  *    // Batch 1
  *    B1 = [[  1.,   2.,   3.],
  *          [  7.,   8.,   9.],
  *          [ 13.,  14.,  15.]]
  * 
  *    // Batch 2
  *    B2 = [[  4.,   5.,   6.],
  *          [ 10.,  11.,  12.],
  *          [ 16.,  17.,  18.]]
  * 
  *    // returns reverse sequence when sequence_length parameter is not used
  *    SequenceReverse(x) = [[[ 13.,  14.,   15.],
  *                           [ 16.,  17.,   18.]],
  * 
  *                          [[  7.,   8.,   9.],
  *                           [ 10.,  11.,  12.]],
  * 
  *                          [[  1.,   2.,   3.],
  *                           [  4.,   5.,   6.]]]
  * 
  *    // sequence_length [2,2] means 2 rows of
  *    // both batch B1 and B2 will be reversed.
  *    SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =
  *                      [[[  7.,   8.,   9.],
  *                        [ 10.,  11.,  12.]],
  * 
  *                       [[  1.,   2.,   3.],
  *                        [  4.,   5.,   6.]],
  * 
  *                       [[ 13.,  14.,   15.],
  *                        [ 16.,  17.,   18.]]]
  * 
  *    // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
  *    // will be reversed.
  *    SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =
  *                     [[[  7.,   8.,   9.],
  *                       [ 16.,  17.,  18.]],
  * 
  *                      [[  1.,   2.,   3.],
  *                       [ 10.,  11.,  12.]],
  * 
  *                      [[ 13.,  14,   15.],
  *                       [  4.,   5.,   6.]]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_reverse.cc:L113
  * @return org.apache.mxnet.NDArray
  */
def SequenceReverse(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the elements of each sequence.
  * 
  * This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]
  * and returns an array of the same shape.
  * 
  * Parameter `sequence_length` is used to handle variable-length sequences.
  * `sequence_length` should be an input array of positive ints of dimension [batch_size].
  * To use this parameter, set `use_sequence_length` to `True`,
  * otherwise each example in the batch is assumed to have the max sequence length.
  * 
  * Example::
  * 
  *    x = [[[  1.,   2.,   3.],
  *          [  4.,   5.,   6.]],
  * 
  *         [[  7.,   8.,   9.],
  *          [ 10.,  11.,  12.]],
  * 
  *         [[ 13.,  14.,   15.],
  *          [ 16.,  17.,   18.]]]
  * 
  *    // Batch 1
  *    B1 = [[  1.,   2.,   3.],
  *          [  7.,   8.,   9.],
  *          [ 13.,  14.,  15.]]
  * 
  *    // Batch 2
  *    B2 = [[  4.,   5.,   6.],
  *          [ 10.,  11.,  12.],
  *          [ 16.,  17.,  18.]]
  * 
  *    // returns reverse sequence when sequence_length parameter is not used
  *    SequenceReverse(x) = [[[ 13.,  14.,   15.],
  *                           [ 16.,  17.,   18.]],
  * 
  *                          [[  7.,   8.,   9.],
  *                           [ 10.,  11.,  12.]],
  * 
  *                          [[  1.,   2.,   3.],
  *                           [  4.,   5.,   6.]]]
  * 
  *    // sequence_length [2,2] means 2 rows of
  *    // both batch B1 and B2 will be reversed.
  *    SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =
  *                      [[[  7.,   8.,   9.],
  *                        [ 10.,  11.,  12.]],
  * 
  *                       [[  1.,   2.,   3.],
  *                        [  4.,   5.,   6.]],
  * 
  *                       [[ 13.,  14.,   15.],
  *                        [ 16.,  17.,   18.]]]
  * 
  *    // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3
  *    // will be reversed.
  *    SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =
  *                     [[[  7.,   8.,   9.],
  *                       [ 16.,  17.,  18.]],
  * 
  *                      [[  1.,   2.,   3.],
  *                       [ 10.,  11.,  12.]],
  * 
  *                      [[ 13.,  14,   15.],
  *                       [  4.,   5.,   6.]]]
  * 
  * 
  * 
  * Defined in src/operator/sequence_reverse.cc:L113
  * @return org.apache.mxnet.NDArray
  */
def SequenceReverse(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Splits an array along a particular axis into multiple sub-arrays.
  * 
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
  * 
  * **Note** that `num_outputs` should evenly divide the length of the axis
  * along which to split the array.
  * 
  * Example::
  * 
  *    x  = [[[ 1.]
  *           [ 2.]]
  *          [[ 3.]
  *           [ 4.]]
  *          [[ 5.]
  *           [ 6.]]]
  *    x.shape = (3, 2, 1)
  * 
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
  *    y = [[[ 1.]]
  *         [[ 3.]]
  *         [[ 5.]]]
  * 
  *        [[[ 2.]]
  *         [[ 4.]]
  *         [[ 6.]]]
  * 
  *    y[0].shape = (3, 1, 1)
  * 
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
  *    z = [[[ 1.]
  *          [ 2.]]]
  * 
  *        [[[ 3.]
  *          [ 4.]]]
  * 
  *        [[[ 5.]
  *          [ 6.]]]
  * 
  *    z[0].shape = (1, 2, 1)
  * 
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
  * along the `axis` which it is split.
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.
  * 
  * Example::
  * 
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
  *    z = [[ 1.]
  *         [ 2.]]
  * 
  *        [[ 3.]
  *         [ 4.]]
  * 
  *        [[ 5.]
  *         [ 6.]]
  *    z[0].shape = (2 ,1 )
  * 
  * 
  * 
  * Defined in src/operator/slice_channel.cc:L107
  * @return org.apache.mxnet.NDArray
  */
def SliceChannel(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Splits an array along a particular axis into multiple sub-arrays.
  * 
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
  * 
  * **Note** that `num_outputs` should evenly divide the length of the axis
  * along which to split the array.
  * 
  * Example::
  * 
  *    x  = [[[ 1.]
  *           [ 2.]]
  *          [[ 3.]
  *           [ 4.]]
  *          [[ 5.]
  *           [ 6.]]]
  *    x.shape = (3, 2, 1)
  * 
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
  *    y = [[[ 1.]]
  *         [[ 3.]]
  *         [[ 5.]]]
  * 
  *        [[[ 2.]]
  *         [[ 4.]]
  *         [[ 6.]]]
  * 
  *    y[0].shape = (3, 1, 1)
  * 
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
  *    z = [[[ 1.]
  *          [ 2.]]]
  * 
  *        [[[ 3.]
  *          [ 4.]]]
  * 
  *        [[[ 5.]
  *          [ 6.]]]
  * 
  *    z[0].shape = (1, 2, 1)
  * 
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
  * along the `axis` which it is split.
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.
  * 
  * Example::
  * 
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
  *    z = [[ 1.]
  *         [ 2.]]
  * 
  *        [[ 3.]
  *         [ 4.]]
  * 
  *        [[ 5.]
  *         [ 6.]]
  *    z[0].shape = (2 ,1 )
  * 
  * 
  * 
  * Defined in src/operator/slice_channel.cc:L107
  * @return org.apache.mxnet.NDArray
  */
def SliceChannel(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Please use `SoftmaxOutput`.
  * 
  * .. note::
  * 
  *   This operator has been renamed to `SoftmaxOutput`, which
  *   computes the gradient of cross-entropy loss w.r.t softmax output.
  *   To just compute softmax output, use the `softmax` operator.
  * 
  * 
  * 
  * Defined in src/operator/softmax_output.cc:L138
  * @return org.apache.mxnet.NDArray
  */
def Softmax(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Please use `SoftmaxOutput`.
  * 
  * .. note::
  * 
  *   This operator has been renamed to `SoftmaxOutput`, which
  *   computes the gradient of cross-entropy loss w.r.t softmax output.
  *   To just compute softmax output, use the `softmax` operator.
  * 
  * 
  * 
  * Defined in src/operator/softmax_output.cc:L138
  * @return org.apache.mxnet.NDArray
  */
def Softmax(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies softmax activation to input. This is intended for internal layers.
  * 
  * .. note::
  * 
  *   This operator has been deprecated, please use `softmax`.
  * 
  * If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.
  * This is the default mode.
  * 
  * If `mode` = ``channel``, this operator will compute a k-class softmax at each position
  * of each instance, where `k` = ``num_channel``. This mode can only be used when the input array
  * has at least 3 dimensions.
  * This can be used for `fully convolutional network`, `image segmentation`, etc.
  * 
  * Example::
  * 
  *   >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
  *   >>>                            [2., -.4, 7.,   3., 0.2]])
  *   >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
  *   >>> print softmax_act.asnumpy()
  *   [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]
  *    [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]
  * 
  * 
  * 
  * Defined in src/operator/nn/softmax_activation.cc:L59
  * @return org.apache.mxnet.NDArray
  */
def SoftmaxActivation(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies softmax activation to input. This is intended for internal layers.
  * 
  * .. note::
  * 
  *   This operator has been deprecated, please use `softmax`.
  * 
  * If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.
  * This is the default mode.
  * 
  * If `mode` = ``channel``, this operator will compute a k-class softmax at each position
  * of each instance, where `k` = ``num_channel``. This mode can only be used when the input array
  * has at least 3 dimensions.
  * This can be used for `fully convolutional network`, `image segmentation`, etc.
  * 
  * Example::
  * 
  *   >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
  *   >>>                            [2., -.4, 7.,   3., 0.2]])
  *   >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
  *   >>> print softmax_act.asnumpy()
  *   [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]
  *    [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]
  * 
  * 
  * 
  * Defined in src/operator/nn/softmax_activation.cc:L59
  * @return org.apache.mxnet.NDArray
  */
def SoftmaxActivation(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the gradient of cross entropy loss with respect to softmax output.
  * 
  * - This operator computes the gradient in two steps.
  *   The cross entropy loss does not actually need to be computed.
  * 
  *   - Applies softmax function on the input array.
  *   - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.
  * 
  * - The softmax function, cross entropy loss and gradient is given by:
  * 
  *   - Softmax Function:
  * 
  *     .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
  * 
  *   - Cross Entropy Function:
  * 
  *     .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)
  * 
  *   - The gradient of cross entropy loss w.r.t softmax output:
  * 
  *     .. math:: \text{gradient} = \text{output} - \text{label}
  * 
  * - During forward propagation, the softmax function is computed for each instance in the input array.
  * 
  *   For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is
  *   :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`
  *   and `multi_output` to specify the way to compute softmax:
  * 
  *   - By default, `preserve_shape` is ``false``. This operator will reshape the input array
  *     into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for
  *     each row in the reshaped array, and afterwards reshape it back to the original shape
  *     :math:`(d_1, d_2, ..., d_n)`.
  *   - If `preserve_shape` is ``true``, the softmax function will be computed along
  *     the last axis (`axis` = ``-1``).
  *   - If `multi_output` is ``true``, the softmax function will be computed along
  *     the second axis (`axis` = ``1``).
  * 
  * - During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.
  *   The provided label can be a one-hot label array or a probability label array.
  * 
  *   - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances
  *     with a particular label to be ignored during backward propagation. **This has no effect when
  *     softmax `output` has same shape as `label`**.
  * 
  *     Example::
  * 
  *       data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
  *       label = [1,0,2,3]
  *       ignore_label = 1
  *       SoftmaxOutput(data=data, label = label,\
  *                     multi_output=true, use_ignore=true,\
  *                     ignore_label=ignore_label)
  *       ## forward softmax output
  *       [[ 0.0320586   0.08714432  0.23688284  0.64391428]
  *        [ 0.25        0.25        0.25        0.25      ]
  *        [ 0.25        0.25        0.25        0.25      ]
  *        [ 0.25        0.25        0.25        0.25      ]]
  *       ## backward gradient output
  *       [[ 0.    0.    0.    0.  ]
  *        [-0.75  0.25  0.25  0.25]
  *        [ 0.25  0.25 -0.75  0.25]
  *        [ 0.25  0.25  0.25 -0.75]]
  *       ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.
  * 
  *   - The parameter `grad_scale` can be used to rescale the gradient, which is often used to
  *     give each loss function different weights.
  * 
  *   - This operator also supports various ways to normalize the gradient by `normalization`,
  *     The `normalization` is applied if softmax output has different shape than the labels.
  *     The `normalization` mode can be set to the followings:
  * 
  *     - ``'null'``: do nothing.
  *     - ``'batch'``: divide the gradient by the batch size.
  *     - ``'valid'``: divide the gradient by the number of instances which are not ignored.
  * 
  * 
  * 
  * Defined in src/operator/softmax_output.cc:L123
  * @return org.apache.mxnet.NDArray
  */
def SoftmaxOutput(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the gradient of cross entropy loss with respect to softmax output.
  * 
  * - This operator computes the gradient in two steps.
  *   The cross entropy loss does not actually need to be computed.
  * 
  *   - Applies softmax function on the input array.
  *   - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.
  * 
  * - The softmax function, cross entropy loss and gradient is given by:
  * 
  *   - Softmax Function:
  * 
  *     .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
  * 
  *   - Cross Entropy Function:
  * 
  *     .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)
  * 
  *   - The gradient of cross entropy loss w.r.t softmax output:
  * 
  *     .. math:: \text{gradient} = \text{output} - \text{label}
  * 
  * - During forward propagation, the softmax function is computed for each instance in the input array.
  * 
  *   For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is
  *   :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`
  *   and `multi_output` to specify the way to compute softmax:
  * 
  *   - By default, `preserve_shape` is ``false``. This operator will reshape the input array
  *     into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for
  *     each row in the reshaped array, and afterwards reshape it back to the original shape
  *     :math:`(d_1, d_2, ..., d_n)`.
  *   - If `preserve_shape` is ``true``, the softmax function will be computed along
  *     the last axis (`axis` = ``-1``).
  *   - If `multi_output` is ``true``, the softmax function will be computed along
  *     the second axis (`axis` = ``1``).
  * 
  * - During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.
  *   The provided label can be a one-hot label array or a probability label array.
  * 
  *   - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances
  *     with a particular label to be ignored during backward propagation. **This has no effect when
  *     softmax `output` has same shape as `label`**.
  * 
  *     Example::
  * 
  *       data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]
  *       label = [1,0,2,3]
  *       ignore_label = 1
  *       SoftmaxOutput(data=data, label = label,\
  *                     multi_output=true, use_ignore=true,\
  *                     ignore_label=ignore_label)
  *       ## forward softmax output
  *       [[ 0.0320586   0.08714432  0.23688284  0.64391428]
  *        [ 0.25        0.25        0.25        0.25      ]
  *        [ 0.25        0.25        0.25        0.25      ]
  *        [ 0.25        0.25        0.25        0.25      ]]
  *       ## backward gradient output
  *       [[ 0.    0.    0.    0.  ]
  *        [-0.75  0.25  0.25  0.25]
  *        [ 0.25  0.25 -0.75  0.25]
  *        [ 0.25  0.25  0.25 -0.75]]
  *       ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.
  * 
  *   - The parameter `grad_scale` can be used to rescale the gradient, which is often used to
  *     give each loss function different weights.
  * 
  *   - This operator also supports various ways to normalize the gradient by `normalization`,
  *     The `normalization` is applied if softmax output has different shape than the labels.
  *     The `normalization` mode can be set to the followings:
  * 
  *     - ``'null'``: do nothing.
  *     - ``'batch'``: divide the gradient by the batch size.
  *     - ``'valid'``: divide the gradient by the number of instances which are not ignored.
  * 
  * 
  * 
  * Defined in src/operator/softmax_output.cc:L123
  * @return org.apache.mxnet.NDArray
  */
def SoftmaxOutput(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a spatial transformer to input feature map.
  * @return org.apache.mxnet.NDArray
  */
def SpatialTransformer(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies a spatial transformer to input feature map.
  * @return org.apache.mxnet.NDArray
  */
def SpatialTransformer(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Interchanges two axes of an array.
  * 
  * Examples::
  * 
  *   x = [[1, 2, 3]])
  *   swapaxes(x, 0, 1) = [[ 1],
  *                        [ 2],
  *                        [ 3]]
  * 
  *   x = [[[ 0, 1],
  *         [ 2, 3]],
  *        [[ 4, 5],
  *         [ 6, 7]]]  // (2,2,2) array
  * 
  *  swapaxes(x, 0, 2) = [[[ 0, 4],
  *                        [ 2, 6]],
  *                       [[ 1, 5],
  *                        [ 3, 7]]]
  * 
  * 
  * Defined in src/operator/swapaxis.cc:L70
  * @return org.apache.mxnet.NDArray
  */
def SwapAxis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Interchanges two axes of an array.
  * 
  * Examples::
  * 
  *   x = [[1, 2, 3]])
  *   swapaxes(x, 0, 1) = [[ 1],
  *                        [ 2],
  *                        [ 3]]
  * 
  *   x = [[[ 0, 1],
  *         [ 2, 3]],
  *        [[ 4, 5],
  *         [ 6, 7]]]  // (2,2,2) array
  * 
  *  swapaxes(x, 0, 2) = [[[ 0, 4],
  *                        [ 2, 6]],
  *                       [[ 1, 5],
  *                        [ 3, 7]]]
  * 
  * 
  * Defined in src/operator/swapaxis.cc:L70
  * @return org.apache.mxnet.NDArray
  */
def SwapAxis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs nearest neighbor/bilinear up sampling to inputs.
  * @return org.apache.mxnet.NDArray
  */
def UpSampling(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs nearest neighbor/bilinear up sampling to inputs.
  * @return org.apache.mxnet.NDArray
  */
def UpSampling(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise absolute value of the input.
  * 
  * Example::
  * 
  *    abs([-2, 0, 3]) = [2, 0, 3]
  * 
  * The storage type of ``abs`` output depends upon the input storage type:
  * 
  *    - abs(default) = default
  *    - abs(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L460
  * @return org.apache.mxnet.NDArray
  */
def abs(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise absolute value of the input.
  * 
  * Example::
  * 
  *    abs([-2, 0, 3]) = [2, 0, 3]
  * 
  * The storage type of ``abs`` output depends upon the input storage type:
  * 
  *    - abs(default) = default
  *    - abs(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L460
  * @return org.apache.mxnet.NDArray
  */
def abs(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Adam optimizer. Adam is seen as a generalization
  * of AdaGrad.
  * 
  * Adam update consists of the following steps, where g represents gradient and m, v
  * are 1st and 2nd order moment estimates (mean and variance).
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
  *  W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
  * 
  * It updates the weights using::
  * 
  *  m = beta1*m + (1-beta1)*grad
  *  v = beta2*v + (1-beta2)*(grad**2)
  *  w += - learning_rate * m / (sqrt(v) + epsilon)
  * 
  * If w, m and v are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for w, m and v)::
  * 
  *  for row in grad.indices:
  *      m[row] = beta1*m[row] + (1-beta1)*grad[row]
  *      v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
  *      w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L454
  * @return org.apache.mxnet.NDArray
  */
def adam_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Adam optimizer. Adam is seen as a generalization
  * of AdaGrad.
  * 
  * Adam update consists of the following steps, where g represents gradient and m, v
  * are 1st and 2nd order moment estimates (mean and variance).
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
  *  W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }
  * 
  * It updates the weights using::
  * 
  *  m = beta1*m + (1-beta1)*grad
  *  v = beta2*v + (1-beta2)*(grad**2)
  *  w += - learning_rate * m / (sqrt(v) + epsilon)
  * 
  * If w, m and v are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for w, m and v)::
  * 
  *  for row in grad.indices:
  *      m[row] = beta1*m[row] + (1-beta1)*grad[row]
  *      v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)
  *      w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L454
  * @return org.apache.mxnet.NDArray
  */
def adam_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds all input arguments element-wise.
  * 
  * .. math::
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
  * 
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
  * 
  * The storage type of ``add_n`` output depends on storage types of inputs
  * 
  * - add_n(row_sparse, row_sparse, ..) = row_sparse
  * - otherwise, ``add_n`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_sum.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def add_n(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds all input arguments element-wise.
  * 
  * .. math::
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n
  * 
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.
  * 
  * The storage type of ``add_n`` output depends on storage types of inputs
  * 
  * - add_n(row_sparse, row_sparse, ..) = row_sparse
  * - otherwise, ``add_n`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_sum.cc:L150
  * @return org.apache.mxnet.NDArray
  */
def add_n(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse cosine of the input array.
  * 
  * The input should be in range `[-1, 1]`.
  * The output is in the closed interval :math:`[0, \pi]`
  * 
  * .. math::
  *    arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
  * 
  * The storage type of ``arccos`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L123
  * @return org.apache.mxnet.NDArray
  */
def arccos(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse cosine of the input array.
  * 
  * The input should be in range `[-1, 1]`.
  * The output is in the closed interval :math:`[0, \pi]`
  * 
  * .. math::
  *    arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]
  * 
  * The storage type of ``arccos`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L123
  * @return org.apache.mxnet.NDArray
  */
def arccos(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic cosine of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arccosh`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L264
  * @return org.apache.mxnet.NDArray
  */
def arccosh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic cosine of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arccosh`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L264
  * @return org.apache.mxnet.NDArray
  */
def arccosh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse sine of the input array.
  * 
  * The input should be in the range `[-1, 1]`.
  * The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
  * 
  * .. math::
  *    arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
  * 
  * The storage type of ``arcsin`` output depends upon the input storage type:
  * 
  *    - arcsin(default) = default
  *    - arcsin(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L104
  * @return org.apache.mxnet.NDArray
  */
def arcsin(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse sine of the input array.
  * 
  * The input should be in the range `[-1, 1]`.
  * The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].
  * 
  * .. math::
  *    arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]
  * 
  * The storage type of ``arcsin`` output depends upon the input storage type:
  * 
  *    - arcsin(default) = default
  *    - arcsin(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L104
  * @return org.apache.mxnet.NDArray
  */
def arcsin(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic sine of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arcsinh`` output depends upon the input storage type:
  * 
  *    - arcsinh(default) = default
  *    - arcsinh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L250
  * @return org.apache.mxnet.NDArray
  */
def arcsinh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic sine of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arcsinh`` output depends upon the input storage type:
  * 
  *    - arcsinh(default) = default
  *    - arcsinh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L250
  * @return org.apache.mxnet.NDArray
  */
def arcsinh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse tangent of the input array.
  * 
  * The output is in the closed interval :math:`[-\pi/2, \pi/2]`
  * 
  * .. math::
  *    arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
  * 
  * The storage type of ``arctan`` output depends upon the input storage type:
  * 
  *    - arctan(default) = default
  *    - arctan(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L144
  * @return org.apache.mxnet.NDArray
  */
def arctan(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse tangent of the input array.
  * 
  * The output is in the closed interval :math:`[-\pi/2, \pi/2]`
  * 
  * .. math::
  *    arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]
  * 
  * The storage type of ``arctan`` output depends upon the input storage type:
  * 
  *    - arctan(default) = default
  *    - arctan(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L144
  * @return org.apache.mxnet.NDArray
  */
def arctan(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic tangent of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arctanh`` output depends upon the input storage type:
  * 
  *    - arctanh(default) = default
  *    - arctanh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L281
  * @return org.apache.mxnet.NDArray
  */
def arctanh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the element-wise inverse hyperbolic tangent of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arctanh`` output depends upon the input storage type:
  * 
  *    - arctanh(default) = default
  *    - arctanh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L281
  * @return org.apache.mxnet.NDArray
  */
def arctanh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns indices of the maximum values along an axis.
  * 
  * In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   // argmax along axis 0
  *   argmax(x, axis=0) = [ 1.,  1.,  1.]
  * 
  *   // argmax along axis 1
  *   argmax(x, axis=1) = [ 2.,  2.]
  * 
  *   // argmax along axis 1 keeping same dims as an input array
  *   argmax(x, axis=1, keepdims=True) = [[ 2.],
  *                                       [ 2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L52
  * @return org.apache.mxnet.NDArray
  */
def argmax(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns indices of the maximum values along an axis.
  * 
  * In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   // argmax along axis 0
  *   argmax(x, axis=0) = [ 1.,  1.,  1.]
  * 
  *   // argmax along axis 1
  *   argmax(x, axis=1) = [ 2.,  2.]
  * 
  *   // argmax along axis 1 keeping same dims as an input array
  *   argmax(x, axis=1, keepdims=True) = [[ 2.],
  *                                       [ 2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L52
  * @return org.apache.mxnet.NDArray
  */
def argmax(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns argmax indices of each channel from the input array.
  * 
  * The result will be an NDArray of shape (num_channel,).
  * 
  * In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   argmax_channel(x) = [ 2.,  2.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L97
  * @return org.apache.mxnet.NDArray
  */
def argmax_channel(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns argmax indices of each channel from the input array.
  * 
  * The result will be an NDArray of shape (num_channel,).
  * 
  * In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   argmax_channel(x) = [ 2.,  2.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L97
  * @return org.apache.mxnet.NDArray
  */
def argmax_channel(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns indices of the minimum values along an axis.
  * 
  * In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   // argmin along axis 0
  *   argmin(x, axis=0) = [ 0.,  0.,  0.]
  * 
  *   // argmin along axis 1
  *   argmin(x, axis=1) = [ 0.,  0.]
  * 
  *   // argmin along axis 1 keeping same dims as an input array
  *   argmin(x, axis=1, keepdims=True) = [[ 0.],
  *                                       [ 0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L77
  * @return org.apache.mxnet.NDArray
  */
def argmin(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns indices of the minimum values along an axis.
  * 
  * In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrence
  * are returned.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.],
  *        [ 3.,  4.,  5.]]
  * 
  *   // argmin along axis 0
  *   argmin(x, axis=0) = [ 0.,  0.,  0.]
  * 
  *   // argmin along axis 1
  *   argmin(x, axis=1) = [ 0.,  0.]
  * 
  *   // argmin along axis 1 keeping same dims as an input array
  *   argmin(x, axis=1, keepdims=True) = [[ 0.],
  *                                       [ 0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L77
  * @return org.apache.mxnet.NDArray
  */
def argmin(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the indices that would sort an input array along the given axis.
  * 
  * This function performs sorting along the given axis and returns an array of indices having same shape
  * as an input array that index data in sorted order.
  * 
  * Examples::
  * 
  *   x = [[ 0.3,  0.2,  0.4],
  *        [ 0.1,  0.3,  0.2]]
  * 
  *   // sort along axis -1
  *   argsort(x) = [[ 1.,  0.,  2.],
  *                 [ 0.,  2.,  1.]]
  * 
  *   // sort along axis 0
  *   argsort(x, axis=0) = [[ 1.,  0.,  1.]
  *                         [ 0.,  1.,  0.]]
  * 
  *   // flatten and then sort
  *   argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L176
  * @return org.apache.mxnet.NDArray
  */
def argsort(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the indices that would sort an input array along the given axis.
  * 
  * This function performs sorting along the given axis and returns an array of indices having same shape
  * as an input array that index data in sorted order.
  * 
  * Examples::
  * 
  *   x = [[ 0.3,  0.2,  0.4],
  *        [ 0.1,  0.3,  0.2]]
  * 
  *   // sort along axis -1
  *   argsort(x) = [[ 1.,  0.,  2.],
  *                 [ 0.,  2.,  1.]]
  * 
  *   // sort along axis 0
  *   argsort(x, axis=0) = [[ 1.,  0.,  1.]
  *                         [ 0.,  1.,  0.]]
  * 
  *   // flatten and then sort
  *   argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L176
  * @return org.apache.mxnet.NDArray
  */
def argsort(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batchwise dot product.
  * 
  * ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
  * ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
  * 
  * For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
  * `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
  * which is computed by::
  * 
  *    batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
  * 
  * 
  * 
  * Defined in src/operator/tensor/dot.cc:L110
  * @return org.apache.mxnet.NDArray
  */
def batch_dot(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Batchwise dot product.
  * 
  * ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and
  * ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.
  * 
  * For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape
  * `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,
  * which is computed by::
  * 
  *    batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])
  * 
  * 
  * 
  * Defined in src/operator/tensor/dot.cc:L110
  * @return org.apache.mxnet.NDArray
  */
def batch_dot(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes elements from a data batch.
  * 
  * .. note::
  *   `batch_take` is deprecated. Use `pick` instead.
  * 
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
  * an output array of shape ``(i0,)`` with::
  * 
  *   output[i] = input[i, indices[i]]
  * 
  * Examples::
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // takes elements with specified indices
  *   batch_take(x, [0,1,0]) = [ 1.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L434
  * @return org.apache.mxnet.NDArray
  */
def batch_take(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes elements from a data batch.
  * 
  * .. note::
  *   `batch_take` is deprecated. Use `pick` instead.
  * 
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
  * an output array of shape ``(i0,)`` with::
  * 
  *   output[i] = input[i, indices[i]]
  * 
  * Examples::
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // takes elements with specified indices
  *   batch_take(x, [0,1,0]) = [ 1.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L434
  * @return org.apache.mxnet.NDArray
  */
def batch_take(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sum of the input arrays with broadcasting.
  * 
  * `broadcast_plus` is an alias to the function `broadcast_add`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],
  *                           [ 2.,  2.,  2.]]
  * 
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
  *                            [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L51
  * @return org.apache.mxnet.NDArray
  */
def broadcast_add(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sum of the input arrays with broadcasting.
  * 
  * `broadcast_plus` is an alias to the function `broadcast_add`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],
  *                           [ 2.,  2.,  2.]]
  * 
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
  *                            [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L51
  * @return org.apache.mxnet.NDArray
  */
def broadcast_add(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array over particular axes.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * Example::
  * 
  *    // given x of shape (1,2,1)
  *    x = [[[ 1.],
  *          [ 2.]]]
  * 
  *    // broadcast x on on axis 2
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
  *                                          [ 2.,  2.,  2.]]]
  *    // broadcast x on on axes 0 and 2
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]],
  *                                                 [[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]]]
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L208
  * @return org.apache.mxnet.NDArray
  */
def broadcast_axes(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array over particular axes.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * Example::
  * 
  *    // given x of shape (1,2,1)
  *    x = [[[ 1.],
  *          [ 2.]]]
  * 
  *    // broadcast x on on axis 2
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
  *                                          [ 2.,  2.,  2.]]]
  *    // broadcast x on on axes 0 and 2
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]],
  *                                                 [[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]]]
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L208
  * @return org.apache.mxnet.NDArray
  */
def broadcast_axes(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array over particular axes.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * Example::
  * 
  *    // given x of shape (1,2,1)
  *    x = [[[ 1.],
  *          [ 2.]]]
  * 
  *    // broadcast x on on axis 2
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
  *                                          [ 2.,  2.,  2.]]]
  *    // broadcast x on on axes 0 and 2
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]],
  *                                                 [[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]]]
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L208
  * @return org.apache.mxnet.NDArray
  */
def broadcast_axis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array over particular axes.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * Example::
  * 
  *    // given x of shape (1,2,1)
  *    x = [[[ 1.],
  *          [ 2.]]]
  * 
  *    // broadcast x on on axis 2
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],
  *                                          [ 2.,  2.,  2.]]]
  *    // broadcast x on on axes 0 and 2
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]],
  *                                                 [[ 1.,  1.,  1.],
  *                                                  [ 2.,  2.,  2.]]]
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L208
  * @return org.apache.mxnet.NDArray
  */
def broadcast_axis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise division of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 6.,  6.,  6.],
  *         [ 6.,  6.,  6.]]
  * 
  *    y = [[ 2.],
  *         [ 3.]]
  * 
  *    broadcast_div(x, y) = [[ 3.,  3.,  3.],
  *                           [ 2.,  2.,  2.]]
  * 
  * Supported sparse operations:
  *    broadcast_div(csr, dense(1D)) = csr (CPU only)
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L165
  * @return org.apache.mxnet.NDArray
  */
def broadcast_div(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise division of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 6.,  6.,  6.],
  *         [ 6.,  6.,  6.]]
  * 
  *    y = [[ 2.],
  *         [ 3.]]
  * 
  *    broadcast_div(x, y) = [[ 3.,  3.,  3.],
  *                           [ 2.,  2.,  2.]]
  * 
  * Supported sparse operations:
  *    broadcast_div(csr, dense(1D)) = csr (CPU only)
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L165
  * @return org.apache.mxnet.NDArray
  */
def broadcast_div(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_equal(x, y) = [[ 0.,  0.,  0.],
  *                             [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46
  * @return org.apache.mxnet.NDArray
  */
def broadcast_equal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_equal(x, y) = [[ 0.,  0.,  0.],
  *                             [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46
  * @return org.apache.mxnet.NDArray
  */
def broadcast_equal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_greater(x, y) = [[ 1.,  1.,  1.],
  *                               [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82
  * @return org.apache.mxnet.NDArray
  */
def broadcast_greater(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_greater(x, y) = [[ 1.,  1.,  1.],
  *                               [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82
  * @return org.apache.mxnet.NDArray
  */
def broadcast_greater(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
  *                                     [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100
  * @return org.apache.mxnet.NDArray
  */
def broadcast_greater_equal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],
  *                                     [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100
  * @return org.apache.mxnet.NDArray
  */
def broadcast_greater_equal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  *  Returns the hypotenuse of a right angled triangle, given its "legs"
  * with broadcasting.
  * 
  * It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
  * 
  * Example::
  * 
  *    x = [[ 3.,  3.,  3.]]
  * 
  *    y = [[ 4.],
  *         [ 4.]]
  * 
  *    broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
  *                             [ 5.,  5.,  5.]]
  * 
  *    z = [[ 0.],
  *         [ 4.]]
  * 
  *    broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
  *                             [ 5.,  5.,  5.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L156
  * @return org.apache.mxnet.NDArray
  */
def broadcast_hypot(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  *  Returns the hypotenuse of a right angled triangle, given its "legs"
  * with broadcasting.
  * 
  * It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.
  * 
  * Example::
  * 
  *    x = [[ 3.,  3.,  3.]]
  * 
  *    y = [[ 4.],
  *         [ 4.]]
  * 
  *    broadcast_hypot(x, y) = [[ 5.,  5.,  5.],
  *                             [ 5.,  5.,  5.]]
  * 
  *    z = [[ 0.],
  *         [ 4.]]
  * 
  *    broadcast_hypot(x, z) = [[ 3.,  3.,  3.],
  *                             [ 5.,  5.,  5.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L156
  * @return org.apache.mxnet.NDArray
  */
def broadcast_hypot(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
  *                              [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118
  * @return org.apache.mxnet.NDArray
  */
def broadcast_lesser(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_lesser(x, y) = [[ 0.,  0.,  0.],
  *                              [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118
  * @return org.apache.mxnet.NDArray
  */
def broadcast_lesser(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
  *                                    [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136
  * @return org.apache.mxnet.NDArray
  */
def broadcast_lesser_equal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],
  *                                    [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136
  * @return org.apache.mxnet.NDArray
  */
def broadcast_lesser_equal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise maximum of the input arrays with broadcasting.
  * 
  * This function compares two input arrays and returns a new array having the element-wise maxima.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
  *                               [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L80
  * @return org.apache.mxnet.NDArray
  */
def broadcast_maximum(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise maximum of the input arrays with broadcasting.
  * 
  * This function compares two input arrays and returns a new array having the element-wise maxima.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_maximum(x, y) = [[ 1.,  1.,  1.],
  *                               [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L80
  * @return org.apache.mxnet.NDArray
  */
def broadcast_maximum(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise minimum of the input arrays with broadcasting.
  * 
  * This function compares two input arrays and returns a new array having the element-wise minima.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
  *                               [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L115
  * @return org.apache.mxnet.NDArray
  */
def broadcast_minimum(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise minimum of the input arrays with broadcasting.
  * 
  * This function compares two input arrays and returns a new array having the element-wise minima.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_maximum(x, y) = [[ 0.,  0.,  0.],
  *                               [ 1.,  1.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L115
  * @return org.apache.mxnet.NDArray
  */
def broadcast_minimum(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise difference of the input arrays with broadcasting.
  * 
  * `broadcast_minus` is an alias to the function `broadcast_sub`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
  *                           [ 0.,  0.,  0.]]
  * 
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
  *                             [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L90
  * @return org.apache.mxnet.NDArray
  */
def broadcast_minus(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise difference of the input arrays with broadcasting.
  * 
  * `broadcast_minus` is an alias to the function `broadcast_sub`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
  *                           [ 0.,  0.,  0.]]
  * 
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
  *                             [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L90
  * @return org.apache.mxnet.NDArray
  */
def broadcast_minus(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise modulo of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 8.,  8.,  8.],
  *         [ 8.,  8.,  8.]]
  * 
  *    y = [[ 2.],
  *         [ 3.]]
  * 
  *    broadcast_mod(x, y) = [[ 0.,  0.,  0.],
  *                           [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L200
  * @return org.apache.mxnet.NDArray
  */
def broadcast_mod(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise modulo of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 8.,  8.,  8.],
  *         [ 8.,  8.,  8.]]
  * 
  *    y = [[ 2.],
  *         [ 3.]]
  * 
  *    broadcast_mod(x, y) = [[ 0.,  0.,  0.],
  *                           [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L200
  * @return org.apache.mxnet.NDArray
  */
def broadcast_mod(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise product of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_mul(x, y) = [[ 0.,  0.,  0.],
  *                           [ 1.,  1.,  1.]]
  * 
  * Supported sparse operations:
  *    broadcast_mul(csr, dense(1D)) = csr (CPU only)
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L126
  * @return org.apache.mxnet.NDArray
  */
def broadcast_mul(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise product of the input arrays with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_mul(x, y) = [[ 0.,  0.,  0.],
  *                           [ 1.,  1.,  1.]]
  * 
  * Supported sparse operations:
  *    broadcast_mul(csr, dense(1D)) = csr (CPU only)
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L126
  * @return org.apache.mxnet.NDArray
  */
def broadcast_mul(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
  *                                 [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64
  * @return org.apache.mxnet.NDArray
  */
def broadcast_not_equal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],
  *                                 [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64
  * @return org.apache.mxnet.NDArray
  */
def broadcast_not_equal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sum of the input arrays with broadcasting.
  * 
  * `broadcast_plus` is an alias to the function `broadcast_add`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],
  *                           [ 2.,  2.,  2.]]
  * 
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
  *                            [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L51
  * @return org.apache.mxnet.NDArray
  */
def broadcast_plus(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sum of the input arrays with broadcasting.
  * 
  * `broadcast_plus` is an alias to the function `broadcast_add`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],
  *                           [ 2.,  2.,  2.]]
  * 
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],
  *                            [ 2.,  2.,  2.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L51
  * @return org.apache.mxnet.NDArray
  */
def broadcast_plus(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns result of first array elements raised to powers from second array, element-wise with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_power(x, y) = [[ 2.,  2.,  2.],
  *                             [ 4.,  4.,  4.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45
  * @return org.apache.mxnet.NDArray
  */
def broadcast_power(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns result of first array elements raised to powers from second array, element-wise with broadcasting.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_power(x, y) = [[ 2.,  2.,  2.],
  *                             [ 4.,  4.,  4.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45
  * @return org.apache.mxnet.NDArray
  */
def broadcast_power(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise difference of the input arrays with broadcasting.
  * 
  * `broadcast_minus` is an alias to the function `broadcast_sub`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
  *                           [ 0.,  0.,  0.]]
  * 
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
  *                             [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L90
  * @return org.apache.mxnet.NDArray
  */
def broadcast_sub(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise difference of the input arrays with broadcasting.
  * 
  * `broadcast_minus` is an alias to the function `broadcast_sub`.
  * 
  * Example::
  * 
  *    x = [[ 1.,  1.,  1.],
  *         [ 1.,  1.,  1.]]
  * 
  *    y = [[ 0.],
  *         [ 1.]]
  * 
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],
  *                           [ 0.,  0.,  0.]]
  * 
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],
  *                             [ 0.,  0.,  0.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L90
  * @return org.apache.mxnet.NDArray
  */
def broadcast_sub(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array to a new shape.
  * 
  * Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
  * with arrays of different shapes efficiently without creating multiple copies of arrays.
  * Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * For example::
  * 
  *    broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
  *                                            [ 1.,  2.,  3.]])
  * 
  * The dimension which you do not want to change can also be kept as `0` which means copy the original value.
  * So with `shape=(2,0)`, we will obtain the same result as in the above example.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L232
  * @return org.apache.mxnet.NDArray
  */
def broadcast_to(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Broadcasts the input array to a new shape.
  * 
  * Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations
  * with arrays of different shapes efficiently without creating multiple copies of arrays.
  * Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.
  * 
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.
  * 
  * For example::
  * 
  *    broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],
  *                                            [ 1.,  2.,  3.]])
  * 
  * The dimension which you do not want to change can also be kept as `0` which means copy the original value.
  * So with `shape=(2,0)`, we will obtain the same result as in the above example.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L232
  * @return org.apache.mxnet.NDArray
  */
def broadcast_to(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts all elements of the input to a new type.
  * 
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
  * 
  * Example::
  * 
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L385
  * @return org.apache.mxnet.NDArray
  */
def cast(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts all elements of the input to a new type.
  * 
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.
  * 
  * Example::
  * 
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L385
  * @return org.apache.mxnet.NDArray
  */
def cast(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts tensor storage type to the new type.
  * 
  * When an NDArray with default storage type is cast to csr or row_sparse storage,
  * the result is compact, which means:
  * 
  * - for csr, zero values will not be retained
  * - for row_sparse, row slices of all zeros will not be retained
  * 
  * The storage type of ``cast_storage`` output depends on stype parameter:
  * 
  * - cast_storage(csr, 'default') = default
  * - cast_storage(row_sparse, 'default') = default
  * - cast_storage(default, 'csr') = csr
  * - cast_storage(default, 'row_sparse') = row_sparse
  * - cast_storage(csr, 'csr') = csr
  * - cast_storage(row_sparse, 'row_sparse') = row_sparse
  * 
  * Example::
  * 
  *     dense = [[ 0.,  1.,  0.],
  *              [ 2.,  0.,  3.],
  *              [ 0.,  0.,  0.],
  *              [ 0.,  0.,  0.]]
  * 
  *     # cast to row_sparse storage type
  *     rsp = cast_storage(dense, 'row_sparse')
  *     rsp.indices = [0, 1]
  *     rsp.values = [[ 0.,  1.,  0.],
  *                   [ 2.,  0.,  3.]]
  * 
  *     # cast to csr storage type
  *     csr = cast_storage(dense, 'csr')
  *     csr.indices = [1, 0, 2]
  *     csr.values = [ 1.,  2.,  3.]
  *     csr.indptr = [0, 1, 3, 3, 3]
  * 
  * 
  * 
  * Defined in src/operator/tensor/cast_storage.cc:L71
  * @return org.apache.mxnet.NDArray
  */
def cast_storage(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Casts tensor storage type to the new type.
  * 
  * When an NDArray with default storage type is cast to csr or row_sparse storage,
  * the result is compact, which means:
  * 
  * - for csr, zero values will not be retained
  * - for row_sparse, row slices of all zeros will not be retained
  * 
  * The storage type of ``cast_storage`` output depends on stype parameter:
  * 
  * - cast_storage(csr, 'default') = default
  * - cast_storage(row_sparse, 'default') = default
  * - cast_storage(default, 'csr') = csr
  * - cast_storage(default, 'row_sparse') = row_sparse
  * - cast_storage(csr, 'csr') = csr
  * - cast_storage(row_sparse, 'row_sparse') = row_sparse
  * 
  * Example::
  * 
  *     dense = [[ 0.,  1.,  0.],
  *              [ 2.,  0.,  3.],
  *              [ 0.,  0.,  0.],
  *              [ 0.,  0.,  0.]]
  * 
  *     # cast to row_sparse storage type
  *     rsp = cast_storage(dense, 'row_sparse')
  *     rsp.indices = [0, 1]
  *     rsp.values = [[ 0.,  1.,  0.],
  *                   [ 2.,  0.,  3.]]
  * 
  *     # cast to csr storage type
  *     csr = cast_storage(dense, 'csr')
  *     csr.indices = [1, 0, 2]
  *     csr.values = [ 1.,  2.,  3.]
  *     csr.indptr = [0, 1, 3, 3, 3]
  * 
  * 
  * 
  * Defined in src/operator/tensor/cast_storage.cc:L71
  * @return org.apache.mxnet.NDArray
  */
def cast_storage(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise cube-root value of the input.
  * 
  * .. math::
  *    cbrt(x) = \sqrt[3]{x}
  * 
  * Example::
  * 
  *    cbrt([1, 8, -125]) = [1, 2, -5]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L676
  * @return org.apache.mxnet.NDArray
  */
def cbrt(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise cube-root value of the input.
  * 
  * .. math::
  *    cbrt(x) = \sqrt[3]{x}
  * 
  * Example::
  * 
  *    cbrt([1, 8, -125]) = [1, 2, -5]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L676
  * @return org.apache.mxnet.NDArray
  */
def cbrt(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise ceiling of the input.
  * 
  * The ceil of the scalar x is the smallest integer i, such that i >= x.
  * 
  * Example::
  * 
  *    ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
  * 
  * The storage type of ``ceil`` output depends upon the input storage type:
  * 
  *    - ceil(default) = default
  *    - ceil(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L538
  * @return org.apache.mxnet.NDArray
  */
def ceil(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise ceiling of the input.
  * 
  * The ceil of the scalar x is the smallest integer i, such that i >= x.
  * 
  * Example::
  * 
  *    ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]
  * 
  * The storage type of ``ceil`` output depends upon the input storage type:
  * 
  *    - ceil(default) = default
  *    - ceil(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L538
  * @return org.apache.mxnet.NDArray
  */
def ceil(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
  * @return org.apache.mxnet.NDArray
  */
def choose_element_0index(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
  * @return org.apache.mxnet.NDArray
  */
def choose_element_0index(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Clips (limits) the values in an array.
  * 
  * Given an interval, values outside the interval are clipped to the interval edges.
  * Clipping ``x`` between `a_min` and `a_x` would be::
  * 
  *    clip(x, a_min, a_max) = max(min(x, a_max), a_min))
  * 
  * Example::
  * 
  *     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  * 
  *     clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
  * 
  * The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \
  * parameter values:
  * 
  *    - clip(default) = default
  *    - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
  *    - clip(csr, a_min <= 0, a_max >= 0) = csr
  *    - clip(row_sparse, a_min < 0, a_max < 0) = default
  *    - clip(row_sparse, a_min > 0, a_max > 0) = default
  *    - clip(csr, a_min < 0, a_max < 0) = csr
  *    - clip(csr, a_min > 0, a_max > 0) = csr
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L542
  * @return org.apache.mxnet.NDArray
  */
def clip(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Clips (limits) the values in an array.
  * 
  * Given an interval, values outside the interval are clipped to the interval edges.
  * Clipping ``x`` between `a_min` and `a_x` would be::
  * 
  *    clip(x, a_min, a_max) = max(min(x, a_max), a_min))
  * 
  * Example::
  * 
  *     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  * 
  *     clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]
  * 
  * The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \
  * parameter values:
  * 
  *    - clip(default) = default
  *    - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse
  *    - clip(csr, a_min <= 0, a_max >= 0) = csr
  *    - clip(row_sparse, a_min < 0, a_max < 0) = default
  *    - clip(row_sparse, a_min > 0, a_max > 0) = default
  *    - clip(csr, a_min < 0, a_max < 0) = csr
  *    - clip(csr, a_min > 0, a_max > 0) = csr
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L542
  * @return org.apache.mxnet.NDArray
  */
def clip(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Joins input arrays along a given axis.
  * 
  * .. note:: `Concat` is deprecated. Use `concat` instead.
  * 
  * The dimensions of the input arrays should be the same except the axis along
  * which they will be concatenated.
  * The dimension of the output array along the concatenated axis will be equal
  * to the sum of the corresponding dimensions of the input arrays.
  * 
  * Example::
  * 
  *    x = [[1,1],[2,2]]
  *    y = [[3,3],[4,4],[5,5]]
  *    z = [[6,6], [7,7],[8,8]]
  * 
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],
  *                           [ 2.,  2.],
  *                           [ 3.,  3.],
  *                           [ 4.,  4.],
  *                           [ 5.,  5.],
  *                           [ 6.,  6.],
  *                           [ 7.,  7.],
  *                           [ 8.,  8.]]
  * 
  *    Note that you cannot concat x,y,z along dimension 1 since dimension
  *    0 is not the same for all the input arrays.
  * 
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
  *                          [ 4.,  4.,  7.,  7.],
  *                          [ 5.,  5.,  8.,  8.]]
  * 
  * 
  * 
  * Defined in src/operator/nn/concat.cc:L235
  * @return org.apache.mxnet.NDArray
  */
def concat(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Joins input arrays along a given axis.
  * 
  * .. note:: `Concat` is deprecated. Use `concat` instead.
  * 
  * The dimensions of the input arrays should be the same except the axis along
  * which they will be concatenated.
  * The dimension of the output array along the concatenated axis will be equal
  * to the sum of the corresponding dimensions of the input arrays.
  * 
  * Example::
  * 
  *    x = [[1,1],[2,2]]
  *    y = [[3,3],[4,4],[5,5]]
  *    z = [[6,6], [7,7],[8,8]]
  * 
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],
  *                           [ 2.,  2.],
  *                           [ 3.,  3.],
  *                           [ 4.,  4.],
  *                           [ 5.,  5.],
  *                           [ 6.,  6.],
  *                           [ 7.,  7.],
  *                           [ 8.,  8.]]
  * 
  *    Note that you cannot concat x,y,z along dimension 1 since dimension
  *    0 is not the same for all the input arrays.
  * 
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],
  *                          [ 4.,  4.,  7.,  7.],
  *                          [ 5.,  5.,  8.,  8.]]
  * 
  * 
  * 
  * Defined in src/operator/nn/concat.cc:L235
  * @return org.apache.mxnet.NDArray
  */
def concat(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise cosine of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
  * 
  * The storage type of ``cos`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def cos(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise cosine of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]
  * 
  * The storage type of ``cos`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def cos(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic cosine  of the input array, computed element-wise.
  * 
  * .. math::
  *    cosh(x) = 0.5\times(exp(x) + exp(-x))
  * 
  * The storage type of ``cosh`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L216
  * @return org.apache.mxnet.NDArray
  */
def cosh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic cosine  of the input array, computed element-wise.
  * 
  * .. math::
  *    cosh(x) = 0.5\times(exp(x) + exp(-x))
  * 
  * The storage type of ``cosh`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L216
  * @return org.apache.mxnet.NDArray
  */
def cosh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices a region of the array.
  * 
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.
  * 
  * This function returns a sliced array between the indices given
  * by `begin` and `end` with the corresponding `step`.
  * 
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
  * where m <= n, results in an array with the shape
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
  * 
  * The resulting array's *k*-th dimension contains elements
  * from the *k*-th dimension of the input array starting
  * from index ``b_k`` (inclusive) with step ``s_k``
  * until reaching ``e_k`` (exclusive).
  * 
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,
  * and `step`, the following rule will be used to set default values.
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
  * else, set `b_k=d_k-1`, `e_k=-1`.
  * 
  * The storage type of ``slice`` output depends on storage types of inputs
  * 
  * - slice(csr) = csr
  * - otherwise, ``slice`` generates output with default storage
  * 
  * .. note:: When input data storage type is csr, it only supports
  * step=(), or step=(None,), or step=(1,) to generate a csr output.
  * For other step parameter values, it falls back to slicing
  * a dense tensor.
  * 
  * Example::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
  *                                      [ 6.,  7.,  8.]]
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
  *                                                             [5.,  7.],
  *                                                             [1.,  3.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L411
  * @return org.apache.mxnet.NDArray
  */
def crop(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices a region of the array.
  * 
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.
  * 
  * This function returns a sliced array between the indices given
  * by `begin` and `end` with the corresponding `step`.
  * 
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
  * where m <= n, results in an array with the shape
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
  * 
  * The resulting array's *k*-th dimension contains elements
  * from the *k*-th dimension of the input array starting
  * from index ``b_k`` (inclusive) with step ``s_k``
  * until reaching ``e_k`` (exclusive).
  * 
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,
  * and `step`, the following rule will be used to set default values.
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
  * else, set `b_k=d_k-1`, `e_k=-1`.
  * 
  * The storage type of ``slice`` output depends on storage types of inputs
  * 
  * - slice(csr) = csr
  * - otherwise, ``slice`` generates output with default storage
  * 
  * .. note:: When input data storage type is csr, it only supports
  * step=(), or step=(None,), or step=(1,) to generate a csr output.
  * For other step parameter values, it falls back to slicing
  * a dense tensor.
  * 
  * Example::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
  *                                      [ 6.,  7.,  8.]]
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
  *                                                             [5.,  7.],
  *                                                             [1.,  3.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L411
  * @return org.apache.mxnet.NDArray
  */
def crop(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Converts each element of the input array from radians to degrees.
  * 
  * .. math::
  *    degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
  * 
  * The storage type of ``degrees`` output depends upon the input storage type:
  * 
  *    - degrees(default) = default
  *    - degrees(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L163
  * @return org.apache.mxnet.NDArray
  */
def degrees(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Converts each element of the input array from radians to degrees.
  * 
  * .. math::
  *    degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]
  * 
  * The storage type of ``degrees`` output depends upon the input storage type:
  * 
  *    - degrees(default) = default
  *    - degrees(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L163
  * @return org.apache.mxnet.NDArray
  */
def degrees(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Dot product of two arrays.
  * 
  * ``dot``'s behavior depends on the input array dimensions:
  * 
  * - 1-D arrays: inner product of vectors
  * - 2-D arrays: matrix multiplication
  * - N-D arrays: a sum product over the last axis of the first input and the first
  *   axis of the second input
  * 
  *   For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  *   result array will have shape `(n,m,r,s)`. It is computed by::
  * 
  *     dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
  * 
  *   Example::
  * 
  *     x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
  *     y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
  *     dot(x,y)[0,0,1,1] = 0
  *     sum(x[0,0,:]*y[:,1,1]) = 0
  * 
  * The storage type of ``dot`` output depends on storage types of inputs and transpose options:
  * 
  * - dot(csr, default) = default
  * - dot(csr.T, default) = row_sparse
  * - dot(csr, row_sparse) = default
  * - dot(default, csr) = csr
  * - otherwise, ``dot`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/dot.cc:L62
  * @return org.apache.mxnet.NDArray
  */
def dot(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Dot product of two arrays.
  * 
  * ``dot``'s behavior depends on the input array dimensions:
  * 
  * - 1-D arrays: inner product of vectors
  * - 2-D arrays: matrix multiplication
  * - N-D arrays: a sum product over the last axis of the first input and the first
  *   axis of the second input
  * 
  *   For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the
  *   result array will have shape `(n,m,r,s)`. It is computed by::
  * 
  *     dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])
  * 
  *   Example::
  * 
  *     x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))
  *     y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))
  *     dot(x,y)[0,0,1,1] = 0
  *     sum(x[0,0,:]*y[:,1,1]) = 0
  * 
  * The storage type of ``dot`` output depends on storage types of inputs and transpose options:
  * 
  * - dot(csr, default) = default
  * - dot(csr.T, default) = row_sparse
  * - dot(csr, row_sparse) = default
  * - dot(default, csr) = csr
  * - otherwise, ``dot`` generates output with default storage
  * 
  * 
  * 
  * Defined in src/operator/tensor/dot.cc:L62
  * @return org.apache.mxnet.NDArray
  */
def dot(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds arguments element-wise.
  * 
  * The storage type of ``elemwise_add`` output depends on storage types of inputs
  * 
  *    - elemwise_add(row_sparse, row_sparse) = row_sparse
  *    - elemwise_add(csr, csr) = csr
  *    - otherwise, ``elemwise_add`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_add(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Adds arguments element-wise.
  * 
  * The storage type of ``elemwise_add`` output depends on storage types of inputs
  * 
  *    - elemwise_add(row_sparse, row_sparse) = row_sparse
  *    - elemwise_add(csr, csr) = csr
  *    - otherwise, ``elemwise_add`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_add(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Divides arguments element-wise.
  * 
  * The storage type of ``elemwise_div`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def elemwise_div(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Divides arguments element-wise.
  * 
  * The storage type of ``elemwise_div`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def elemwise_div(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Multiplies arguments element-wise.
  * 
  * The storage type of ``elemwise_mul`` output depends on storage types of inputs
  * 
  *    - elemwise_mul(default, default) = default
  *    - elemwise_mul(row_sparse, row_sparse) = row_sparse
  *    - elemwise_mul(default, row_sparse) = default
  *    - elemwise_mul(row_sparse, default) = default
  *    - elemwise_mul(csr, csr) = csr
  *    - otherwise, ``elemwise_mul`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_mul(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Multiplies arguments element-wise.
  * 
  * The storage type of ``elemwise_mul`` output depends on storage types of inputs
  * 
  *    - elemwise_mul(default, default) = default
  *    - elemwise_mul(row_sparse, row_sparse) = row_sparse
  *    - elemwise_mul(default, row_sparse) = default
  *    - elemwise_mul(row_sparse, default) = default
  *    - elemwise_mul(csr, csr) = csr
  *    - otherwise, ``elemwise_mul`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_mul(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Subtracts arguments element-wise.
  * 
  * The storage type of ``elemwise_sub`` output depends on storage types of inputs
  * 
  *    - elemwise_sub(row_sparse, row_sparse) = row_sparse
  *    - elemwise_sub(csr, csr) = csr
  *    - otherwise, ``elemwise_sub`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_sub(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Subtracts arguments element-wise.
  * 
  * The storage type of ``elemwise_sub`` output depends on storage types of inputs
  * 
  *    - elemwise_sub(row_sparse, row_sparse) = row_sparse
  *    - elemwise_sub(csr, csr) = csr
  *    - otherwise, ``elemwise_sub`` generates output with default storage
  * @return org.apache.mxnet.NDArray
  */
def elemwise_sub(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise exponential value of the input.
  * 
  * .. math::
  *    exp(x) = e^x \approx 2.718^x
  * 
  * Example::
  * 
  *    exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
  * 
  * The storage type of ``exp`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L716
  * @return org.apache.mxnet.NDArray
  */
def exp(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise exponential value of the input.
  * 
  * .. math::
  *    exp(x) = e^x \approx 2.718^x
  * 
  * Example::
  * 
  *    exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]
  * 
  * The storage type of ``exp`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L716
  * @return org.apache.mxnet.NDArray
  */
def exp(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Inserts a new axis of size 1 into the array shape
  * 
  * For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
  * will return a new array with shape ``(2,1,3,4)``.
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L345
  * @return org.apache.mxnet.NDArray
  */
def expand_dims(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Inserts a new axis of size 1 into the array shape
  * 
  * For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
  * will return a new array with shape ``(2,1,3,4)``.
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L345
  * @return org.apache.mxnet.NDArray
  */
def expand_dims(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns ``exp(x) - 1`` computed element-wise on the input.
  * 
  * This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.
  * 
  * The storage type of ``expm1`` output depends upon the input storage type:
  * 
  *    - expm1(default) = default
  *    - expm1(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L795
  * @return org.apache.mxnet.NDArray
  */
def expm1(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns ``exp(x) - 1`` computed element-wise on the input.
  * 
  * This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.
  * 
  * The storage type of ``expm1`` output depends upon the input storage type:
  * 
  *    - expm1(default) = default
  *    - expm1(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L795
  * @return org.apache.mxnet.NDArray
  */
def expm1(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
  * @return org.apache.mxnet.NDArray
  */
def fill_element_0index(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
  * @return org.apache.mxnet.NDArray
  */
def fill_element_0index(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest \
  * integer towards zero of the input.
  * 
  * Example::
  * 
  *    fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
  * 
  * The storage type of ``fix`` output depends upon the input storage type:
  * 
  *    - fix(default) = default
  *    - fix(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L595
  * @return org.apache.mxnet.NDArray
  */
def fix(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest \
  * integer towards zero of the input.
  * 
  * Example::
  * 
  *    fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]
  * 
  * The storage type of ``fix`` output depends upon the input storage type:
  * 
  *    - fix(default) = default
  *    - fix(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L595
  * @return org.apache.mxnet.NDArray
  */
def fix(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.
  * 
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.
  * 
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
  * the input array into an output array of shape ``(d1, d2*...*dk)``.
  * 
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,
  * which behaves similar to mxnet.ndarray.reshape((-1,)).
  * 
  * Example::
  * 
  *     x = [[
  *         [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ],
  *     [    [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ]],
  * 
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L257
  * @return org.apache.mxnet.NDArray
  */
def flatten(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.
  * 
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.
  * 
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes
  * the input array into an output array of shape ``(d1, d2*...*dk)``.
  * 
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,
  * which behaves similar to mxnet.ndarray.reshape((-1,)).
  * 
  * Example::
  * 
  *     x = [[
  *         [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ],
  *     [    [1,2,3],
  *         [4,5,6],
  *         [7,8,9]
  *     ]],
  * 
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L257
  * @return org.apache.mxnet.NDArray
  */
def flatten(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the order of elements along given axis while preserving array shape.
  * 
  * Note: reverse and flip are equivalent. We use reverse in the following examples.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.,  3.,  4.],
  *        [ 5.,  6.,  7.,  8.,  9.]]
  * 
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
  *                         [ 0.,  1.,  2.,  3.,  4.]]
  * 
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
  *                         [ 9.,  8.,  7.,  6.,  5.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L717
  * @return org.apache.mxnet.NDArray
  */
def flip(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the order of elements along given axis while preserving array shape.
  * 
  * Note: reverse and flip are equivalent. We use reverse in the following examples.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.,  3.,  4.],
  *        [ 5.,  6.,  7.,  8.,  9.]]
  * 
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
  *                         [ 0.,  1.,  2.,  3.,  4.]]
  * 
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
  *                         [ 9.,  8.,  7.,  6.,  5.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L717
  * @return org.apache.mxnet.NDArray
  */
def flip(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise floor of the input.
  * 
  * The floor of the scalar x is the largest integer i, such that i <= x.
  * 
  * Example::
  * 
  *    floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
  * 
  * The storage type of ``floor`` output depends upon the input storage type:
  * 
  *    - floor(default) = default
  *    - floor(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L557
  * @return org.apache.mxnet.NDArray
  */
def floor(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise floor of the input.
  * 
  * The floor of the scalar x is the largest integer i, such that i <= x.
  * 
  * Example::
  * 
  *    floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]
  * 
  * The storage type of ``floor`` output depends upon the input storage type:
  * 
  *    - floor(default) = default
  *    - floor(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L557
  * @return org.apache.mxnet.NDArray
  */
def floor(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * The FTML optimizer described in
  * *FTML - Follow the Moving Leader in Deep Learning*,
  * available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
  *  d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)
  *  \sigma_t = d_t - \beta_1 d_{t-1}
  *  z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
  *  W_t = - \frac{ z_t }{ d_t }
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L407
  * @return org.apache.mxnet.NDArray
  */
def ftml_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * The FTML optimizer described in
  * *FTML - Follow the Moving Leader in Deep Learning*,
  * available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\
  *  d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)
  *  \sigma_t = d_t - \beta_1 d_{t-1}
  *  z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}
  *  W_t = - \frac{ z_t }{ d_t }
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L407
  * @return org.apache.mxnet.NDArray
  */
def ftml_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Ftrl optimizer.
  * Referenced from *Ad Click Prediction: a View from the Trenches*, available at
  * http://dl.acm.org/citation.cfm?id=2488200.
  * 
  * It updates the weights using::
  * 
  *  rescaled_grad = clip(grad * rescale_grad, clip_gradient)
  *  z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
  *  n += rescaled_grad**2
  *  w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)
  * 
  * If w, z and n are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for w, z and n)::
  * 
  *  for row in grad.indices:
  *      rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
  *      z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
  *      n[row] += rescaled_grad[row]**2
  *      w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L591
  * @return org.apache.mxnet.NDArray
  */
def ftrl_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Ftrl optimizer.
  * Referenced from *Ad Click Prediction: a View from the Trenches*, available at
  * http://dl.acm.org/citation.cfm?id=2488200.
  * 
  * It updates the weights using::
  * 
  *  rescaled_grad = clip(grad * rescale_grad, clip_gradient)
  *  z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate
  *  n += rescaled_grad**2
  *  w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)
  * 
  * If w, z and n are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for w, z and n)::
  * 
  *  for row in grad.indices:
  *      rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)
  *      z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate
  *      n[row] += rescaled_grad[row]**2
  *      w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L591
  * @return org.apache.mxnet.NDArray
  */
def ftrl_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the gamma function (extension of the factorial function \
  * to the reals), computed element-wise on the input array.
  * 
  * The storage type of ``gamma`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def gamma(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the gamma function (extension of the factorial function \
  * to the reals), computed element-wise on the input array.
  * 
  * The storage type of ``gamma`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def gamma(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise log of the absolute value of the gamma function \
  * of the input.
  * 
  * The storage type of ``gammaln`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def gammaln(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise log of the absolute value of the gamma function \
  * of the input.
  * 
  * The storage type of ``gammaln`` output is always dense
  * @return org.apache.mxnet.NDArray
  */
def gammaln(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Gather elements or slices from `data` and store to a tensor whose
  * shape is defined by `indices`.
  * 
  * Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,
  * where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.
  * 
  * The elements in output is defined as follows::
  * 
  *   output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],
  *                                                       ...,
  *                                                       indices[M-1, y_0, ..., y_{K-1}],
  *                                                       x_M, ..., x_{N-1}]
  * 
  * Examples::
  * 
  *   data = [[0, 1], [2, 3]]
  *   indices = [[1, 1, 0], [0, 1, 0]]
  *   gather_nd(data, indices) = [2, 3, 0]
  * @return org.apache.mxnet.NDArray
  */
def gather_nd(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Gather elements or slices from `data` and store to a tensor whose
  * shape is defined by `indices`.
  * 
  * Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,
  * where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.
  * 
  * The elements in output is defined as follows::
  * 
  *   output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],
  *                                                       ...,
  *                                                       indices[M-1, y_0, ..., y_{K-1}],
  *                                                       x_M, ..., x_{N-1}]
  * 
  * Examples::
  * 
  *   data = [[0, 1], [2, 3]]
  *   indices = [[1, 1, 0], [0, 1, 0]]
  *   gather_nd(data, indices) = [2, 3, 0]
  * @return org.apache.mxnet.NDArray
  */
def gather_nd(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a copy of the input.
  * 
  * From:src/operator/tensor/elemwise_unary_op_basic.cc:176
  * @return org.apache.mxnet.NDArray
  */
def identity(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a copy of the input.
  * 
  * From:src/operator/tensor/elemwise_unary_op_basic.cc:176
  * @return org.apache.mxnet.NDArray
  */
def identity(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the Khatri-Rao product of the input matrices.
  * 
  * Given a collection of :math:`n` input matrices,
  * 
  * .. math::
  *    A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},
  * 
  * the (column-wise) Khatri-Rao product is defined as the matrix,
  * 
  * .. math::
  *    X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},
  * 
  * where the :math:`k` th column is equal to the column-wise outer product
  * :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
  * column of the ith matrix.
  * 
  * Example::
  * 
  *   >>> A = mx.nd.array([[1, -1],
  *   >>>                  [2, -3]])
  *   >>> B = mx.nd.array([[1, 4],
  *   >>>                  [2, 5],
  *   >>>                  [3, 6]])
  *   >>> C = mx.nd.khatri_rao(A, B)
  *   >>> print(C.asnumpy())
  *   [[  1.  -4.]
  *    [  2.  -5.]
  *    [  3.  -6.]
  *    [  2. -12.]
  *    [  4. -15.]
  *    [  6. -18.]]
  * 
  * 
  * 
  * Defined in src/operator/contrib/krprod.cc:L108
  * @return org.apache.mxnet.NDArray
  */
def khatri_rao(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the Khatri-Rao product of the input matrices.
  * 
  * Given a collection of :math:`n` input matrices,
  * 
  * .. math::
  *    A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},
  * 
  * the (column-wise) Khatri-Rao product is defined as the matrix,
  * 
  * .. math::
  *    X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},
  * 
  * where the :math:`k` th column is equal to the column-wise outer product
  * :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth
  * column of the ith matrix.
  * 
  * Example::
  * 
  *   >>> A = mx.nd.array([[1, -1],
  *   >>>                  [2, -3]])
  *   >>> B = mx.nd.array([[1, 4],
  *   >>>                  [2, 5],
  *   >>>                  [3, 6]])
  *   >>> C = mx.nd.khatri_rao(A, B)
  *   >>> print(C.asnumpy())
  *   [[  1.  -4.]
  *    [  2.  -5.]
  *    [  3.  -6.]
  *    [  2. -12.]
  *    [  4. -15.]
  *    [  6. -18.]]
  * 
  * 
  * 
  * Defined in src/operator/contrib/krprod.cc:L108
  * @return org.apache.mxnet.NDArray
  */
def khatri_rao(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * LQ factorization for general matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*
  * must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ
  * factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so
  * that:
  * 
  *    *A* = *L* \* *Q*
  * 
  * Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,
  * and *Q* is row-orthonormal, meaning that
  * 
  *    *Q* \* *Q*\ :sup:`T`
  * 
  * is equal to the identity matrix of shape *(x, x)*.
  * 
  * If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single LQ factorization
  *    A = [[1., 2., 3.], [4., 5., 6.]]
  *    Q, L = gelqf(A)
  *    Q = [[-0.26726124, -0.53452248, -0.80178373],
  *         [0.87287156, 0.21821789, -0.43643578]]
  *    L = [[-3.74165739, 0.],
  *         [-8.55235974, 1.96396101]]
  * 
  *    // Batch LQ factorization
  *    A = [[[1., 2., 3.], [4., 5., 6.]],
  *         [[7., 8., 9.], [10., 11., 12.]]]
  *    Q, L = gelqf(A)
  *    Q = [[[-0.26726124, -0.53452248, -0.80178373],
  *          [0.87287156, 0.21821789, -0.43643578]],
  *         [[-0.50257071, -0.57436653, -0.64616234],
  *          [0.7620735, 0.05862104, -0.64483142]]]
  *    L = [[[-3.74165739, 0.],
  *          [-8.55235974, 1.96396101]],
  *         [[-13.92838828, 0.],
  *          [-19.09768702, 0.52758934]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L529
  * @return org.apache.mxnet.NDArray
  */
def linalg_gelqf(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * LQ factorization for general matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*
  * must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ
  * factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so
  * that:
  * 
  *    *A* = *L* \* *Q*
  * 
  * Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,
  * and *Q* is row-orthonormal, meaning that
  * 
  *    *Q* \* *Q*\ :sup:`T`
  * 
  * is equal to the identity matrix of shape *(x, x)*.
  * 
  * If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single LQ factorization
  *    A = [[1., 2., 3.], [4., 5., 6.]]
  *    Q, L = gelqf(A)
  *    Q = [[-0.26726124, -0.53452248, -0.80178373],
  *         [0.87287156, 0.21821789, -0.43643578]]
  *    L = [[-3.74165739, 0.],
  *         [-8.55235974, 1.96396101]]
  * 
  *    // Batch LQ factorization
  *    A = [[[1., 2., 3.], [4., 5., 6.]],
  *         [[7., 8., 9.], [10., 11., 12.]]]
  *    Q, L = gelqf(A)
  *    Q = [[[-0.26726124, -0.53452248, -0.80178373],
  *          [0.87287156, 0.21821789, -0.43643578]],
  *         [[-0.50257071, -0.57436653, -0.64616234],
  *          [0.7620735, 0.05862104, -0.64483142]]]
  *    L = [[[-3.74165739, 0.],
  *          [-8.55235974, 1.96396101]],
  *         [[-13.92838828, 0.],
  *          [-19.09768702, 0.52758934]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L529
  * @return org.apache.mxnet.NDArray
  */
def linalg_gelqf(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs general matrix multiplication and accumulation.
  * Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, the BLAS3 function *gemm* is performed:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*
  * 
  * Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or
  * matrix transposition (depending on *transpose_a*, *transpose_b*).
  * 
  * If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply-add
  *    A = [[1.0, 1.0], [1.0, 1.0]]
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
  *    C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  *    gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)
  *            = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]
  * 
  *    // Batch matrix multiply-add
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    C = [[[10.0]], [[0.01]]]
  *    gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)
  *            = [[[104.0]], [[0.14]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L69
  * @return org.apache.mxnet.NDArray
  */
def linalg_gemm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs general matrix multiplication and accumulation.
  * Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, the BLAS3 function *gemm* is performed:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*
  * 
  * Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or
  * matrix transposition (depending on *transpose_a*, *transpose_b*).
  * 
  * If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply-add
  *    A = [[1.0, 1.0], [1.0, 1.0]]
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
  *    C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  *    gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)
  *            = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]
  * 
  *    // Batch matrix multiply-add
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    C = [[[10.0]], [[0.01]]]
  *    gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)
  *            = [[[104.0]], [[0.14]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L69
  * @return org.apache.mxnet.NDArray
  */
def linalg_gemm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs general matrix multiplication.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, the BLAS3 function *gemm* is performed:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)
  * 
  * Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix
  * transposition (depending on *transpose_a*, *transpose_b*).
  * 
  * If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply
  *    A = [[1.0, 1.0], [1.0, 1.0]]
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
  *    gemm2(A, B, transpose_b=True, alpha=2.0)
  *             = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
  * 
  *    // Batch matrix multiply
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    gemm2(A, B, transpose_b=True, alpha=2.0)
  *            = [[[4.0]], [[0.04 ]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L128
  * @return org.apache.mxnet.NDArray
  */
def linalg_gemm2(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs general matrix multiplication.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, the BLAS3 function *gemm* is performed:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)
  * 
  * Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix
  * transposition (depending on *transpose_a*, *transpose_b*).
  * 
  * If *n>2*, *gemm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply
  *    A = [[1.0, 1.0], [1.0, 1.0]]
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
  *    gemm2(A, B, transpose_b=True, alpha=2.0)
  *             = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]
  * 
  *    // Batch matrix multiply
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]
  *    gemm2(A, B, transpose_b=True, alpha=2.0)
  *            = [[[4.0]], [[0.04 ]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L128
  * @return org.apache.mxnet.NDArray
  */
def linalg_gemm2(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs Cholesky factorization of a symmetric positive-definite matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, the Cholesky factor *L* of the symmetric, positive definite matrix *A* is
  * computed. *L* is lower triangular (entries of upper triangle are all zero), has
  * positive diagonal entries, and:
  * 
  *   *A* = *L* \* *L*\ :sup:`T`
  * 
  * If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix factorization
  *    A = [[4.0, 1.0], [1.0, 4.25]]
  *    potrf(A) = [[2.0, 0], [0.5, 2.0]]
  * 
  *    // Batch matrix factorization
  *    A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
  *    potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L178
  * @return org.apache.mxnet.NDArray
  */
def linalg_potrf(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs Cholesky factorization of a symmetric positive-definite matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, the Cholesky factor *L* of the symmetric, positive definite matrix *A* is
  * computed. *L* is lower triangular (entries of upper triangle are all zero), has
  * positive diagonal entries, and:
  * 
  *   *A* = *L* \* *L*\ :sup:`T`
  * 
  * If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix factorization
  *    A = [[4.0, 1.0], [1.0, 4.25]]
  *    potrf(A) = [[2.0, 0], [0.5, 2.0]]
  * 
  *    // Batch matrix factorization
  *    A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]
  *    potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L178
  * @return org.apache.mxnet.NDArray
  */
def linalg_potrf(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs matrix inversion from a Cholesky factorization.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, *A* is a lower triangular matrix (entries of upper triangle are all zero)
  * with positive diagonal. We compute:
  * 
  *   *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1`
  * 
  * In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix
  * *B* (obtained by *potrf*), then
  * 
  *   *out* = *B*\ :sup:`-1`
  * 
  * If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * .. note:: Use this operator only if you are certain you need the inverse of *B*, and
  *           cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution
  *           (*trsm*). The latter is numerically much safer, and also cheaper.
  * 
  * Examples::
  * 
  *    // Single matrix inverse
  *    A = [[2.0, 0], [0.5, 2.0]]
  *    potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]
  * 
  *    // Batch matrix inverse
  *    A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
  *    potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
  *                [[0.06641, -0.01562], [-0.01562, 0,0625]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L236
  * @return org.apache.mxnet.NDArray
  */
def linalg_potri(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs matrix inversion from a Cholesky factorization.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, *A* is a lower triangular matrix (entries of upper triangle are all zero)
  * with positive diagonal. We compute:
  * 
  *   *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1`
  * 
  * In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix
  * *B* (obtained by *potrf*), then
  * 
  *   *out* = *B*\ :sup:`-1`
  * 
  * If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * .. note:: Use this operator only if you are certain you need the inverse of *B*, and
  *           cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution
  *           (*trsm*). The latter is numerically much safer, and also cheaper.
  * 
  * Examples::
  * 
  *    // Single matrix inverse
  *    A = [[2.0, 0], [0.5, 2.0]]
  *    potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]
  * 
  *    // Batch matrix inverse
  *    A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]
  *    potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],
  *                [[0.06641, -0.01562], [-0.01562, 0,0625]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L236
  * @return org.apache.mxnet.NDArray
  */
def linalg_potri(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of the logarithms of the diagonal elements of a square matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, *A* must be square with positive diagonal entries. We sum the natural
  * logarithms of the diagonal elements, the result has shape (1,).
  * 
  * If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix reduction
  *    A = [[1.0, 1.0], [1.0, 7.0]]
  *    sumlogdiag(A) = [1.9459]
  * 
  *    // Batch matrix reduction
  *    A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
  *    sumlogdiag(A) = [1.9459, 3.9318]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L405
  * @return org.apache.mxnet.NDArray
  */
def linalg_sumlogdiag(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of the logarithms of the diagonal elements of a square matrix.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, *A* must be square with positive diagonal entries. We sum the natural
  * logarithms of the diagonal elements, the result has shape (1,).
  * 
  * If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix reduction
  *    A = [[1.0, 1.0], [1.0, 7.0]]
  *    sumlogdiag(A) = [1.9459]
  * 
  *    // Batch matrix reduction
  *    A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]
  *    sumlogdiag(A) = [1.9459, 3.9318]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L405
  * @return org.apache.mxnet.NDArray
  */
def linalg_sumlogdiag(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Multiplication of matrix with its transpose.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, the operator performs the BLAS3 function *syrk*:
  * 
  *   *out* = *alpha* \* *A* \* *A*\ :sup:`T`
  * 
  * if *transpose=False*, or
  * 
  *   *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*
  * 
  * if *transpose=True*.
  * 
  * If *n>2*, *syrk* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply
  *    A = [[1., 2., 3.], [4., 5., 6.]]
  *    syrk(A, alpha=1., transpose=False)
  *             = [[14., 32.],
  *                [32., 77.]]
  *    syrk(A, alpha=1., transpose=True)
  *             = [[17., 22., 27.],
  *                [22., 29., 36.],
  *                [27., 36., 45.]]
  * 
  *    // Batch matrix multiply
  *    A = [[[1., 1.]], [[0.1, 0.1]]]
  *    syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L461
  * @return org.apache.mxnet.NDArray
  */
def linalg_syrk(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Multiplication of matrix with its transpose.
  * Input is a tensor *A* of dimension *n >= 2*.
  * 
  * If *n=2*, the operator performs the BLAS3 function *syrk*:
  * 
  *   *out* = *alpha* \* *A* \* *A*\ :sup:`T`
  * 
  * if *transpose=False*, or
  * 
  *   *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*
  * 
  * if *transpose=True*.
  * 
  * If *n>2*, *syrk* is performed separately on the trailing two dimensions for all
  * inputs (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix multiply
  *    A = [[1., 2., 3.], [4., 5., 6.]]
  *    syrk(A, alpha=1., transpose=False)
  *             = [[14., 32.],
  *                [32., 77.]]
  *    syrk(A, alpha=1., transpose=True)
  *             = [[17., 22., 27.],
  *                [22., 29., 36.],
  *                [27., 36., 45.]]
  * 
  *    // Batch matrix multiply
  *    A = [[[1., 1.]], [[0.1, 0.1]]]
  *    syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L461
  * @return org.apache.mxnet.NDArray
  */
def linalg_syrk(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs multiplication with a lower triangular matrix.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
  * *trmm*:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *B*
  * 
  * if *rightside=False*, or
  * 
  *    *out* = *alpha* \* *B* \* *op*\ (*A*)
  * 
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
  * identity or the matrix transposition (depending on *transpose*).
  * 
  * If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * 
  * Examples::
  * 
  *    // Single triangular matrix multiply
  *    A = [[1.0, 0], [1.0, 1.0]]
  *    B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  *    trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
  * 
  *    // Batch triangular matrix multiply
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
  *    B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
  *    trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
  *                             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def linalg_trmm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Performs multiplication with a lower triangular matrix.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
  * *trmm*:
  * 
  *    *out* = *alpha* \* *op*\ (*A*) \* *B*
  * 
  * if *rightside=False*, or
  * 
  *    *out* = *alpha* \* *B* \* *op*\ (*A*)
  * 
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
  * identity or the matrix transposition (depending on *transpose*).
  * 
  * If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * 
  * Examples::
  * 
  *    // Single triangular matrix multiply
  *    A = [[1.0, 0], [1.0, 1.0]]
  *    B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  *    trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
  * 
  *    // Batch triangular matrix multiply
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
  *    B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]
  *    trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
  *                             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def linalg_trmm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Solves matrix equation involving a lower triangular matrix.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
  * *trsm*, solving for *out* in:
  * 
  *    *op*\ (*A*) \* *out* = *alpha* \* *B*
  * 
  * if *rightside=False*, or
  * 
  *    *out* \* *op*\ (*A*) = *alpha* \* *B*
  * 
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
  * identity or the matrix transposition (depending on *transpose*).
  * 
  * If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix solve
  *    A = [[1.0, 0], [1.0, 1.0]]
  *    B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
  *    trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  * 
  *    // Batch matrix solve
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
  *    B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
  *         [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
  *    trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
  *                             [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L356
  * @return org.apache.mxnet.NDArray
  */
def linalg_trsm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Solves matrix equation involving a lower triangular matrix.
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape
  * on the leading *n-2* dimensions.
  * 
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function
  * *trsm*, solving for *out* in:
  * 
  *    *op*\ (*A*) \* *out* = *alpha* \* *B*
  * 
  * if *rightside=False*, or
  * 
  *    *out* \* *op*\ (*A*) = *alpha* \* *B*
  * 
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the
  * identity or the matrix transposition (depending on *transpose*).
  * 
  * If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs
  * (batch mode).
  * 
  * .. note:: The operator supports float32 and float64 data types only.
  * 
  * Examples::
  * 
  *    // Single matrix solve
  *    A = [[1.0, 0], [1.0, 1.0]]
  *    B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]
  *    trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
  * 
  *    // Batch matrix solve
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]
  *    B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],
  *         [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]
  *    trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
  *                             [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]
  * 
  * 
  * Defined in src/operator/tensor/la_op.cc:L356
  * @return org.apache.mxnet.NDArray
  */
def linalg_trsm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Natural logarithmic value of the input.
  * 
  * The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
  * 
  * The storage type of ``log`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L728
  * @return org.apache.mxnet.NDArray
  */
def log(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Natural logarithmic value of the input.
  * 
  * The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``
  * 
  * The storage type of ``log`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L728
  * @return org.apache.mxnet.NDArray
  */
def log(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Base-10 logarithmic value of the input.
  * 
  * ``10**log10(x) = x``
  * 
  * The storage type of ``log10`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L740
  * @return org.apache.mxnet.NDArray
  */
def log10(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Base-10 logarithmic value of the input.
  * 
  * ``10**log10(x) = x``
  * 
  * The storage type of ``log10`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L740
  * @return org.apache.mxnet.NDArray
  */
def log10(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise ``log(1 + x)`` value of the input.
  * 
  * This function is more accurate than ``log(1 + x)``  for small ``x`` so that
  * :math:`1+x\approx 1`
  * 
  * The storage type of ``log1p`` output depends upon the input storage type:
  * 
  *    - log1p(default) = default
  *    - log1p(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L777
  * @return org.apache.mxnet.NDArray
  */
def log1p(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise ``log(1 + x)`` value of the input.
  * 
  * This function is more accurate than ``log(1 + x)``  for small ``x`` so that
  * :math:`1+x\approx 1`
  * 
  * The storage type of ``log1p`` output depends upon the input storage type:
  * 
  *    - log1p(default) = default
  *    - log1p(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L777
  * @return org.apache.mxnet.NDArray
  */
def log1p(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Base-2 logarithmic value of the input.
  * 
  * ``2**log2(x) = x``
  * 
  * The storage type of ``log2`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L752
  * @return org.apache.mxnet.NDArray
  */
def log2(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise Base-2 logarithmic value of the input.
  * 
  * ``2**log2(x) = x``
  * 
  * The storage type of ``log2`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L752
  * @return org.apache.mxnet.NDArray
  */
def log2(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the log softmax of the input.
  * This is equivalent to computing softmax followed by log.
  * 
  * Examples::
  * 
  *   >>> x = mx.nd.array([1, 2, .1])
  *   >>> mx.nd.log_softmax(x).asnumpy()
  *   array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
  * 
  *   >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
  *   >>> mx.nd.log_softmax(x, axis=0).asnumpy()
  *   array([[-0.34115392, -0.69314718, -1.24115396],
  *          [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
  * @return org.apache.mxnet.NDArray
  */
def log_softmax(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the log softmax of the input.
  * This is equivalent to computing softmax followed by log.
  * 
  * Examples::
  * 
  *   >>> x = mx.nd.array([1, 2, .1])
  *   >>> mx.nd.log_softmax(x).asnumpy()
  *   array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)
  * 
  *   >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )
  *   >>> mx.nd.log_softmax(x, axis=0).asnumpy()
  *   array([[-0.34115392, -0.69314718, -1.24115396],
  *          [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)
  * @return org.apache.mxnet.NDArray
  */
def log_softmax(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Make your own loss function in network construction.
  * 
  * This operator accepts a customized loss function symbol as a terminal loss and
  * the symbol should be an operator with no backward dependency.
  * The output of this function is the gradient of loss with respect to the input data.
  * 
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::
  * 
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  *   loss = make_loss(cross_entropy)
  * 
  * We will need to use ``make_loss`` when we are creating our own loss function or we want to
  * combine multiple loss functions. Also we may want to stop some variables' gradients
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
  * 
  * The storage type of ``make_loss`` output depends upon the input storage type:
  * 
  *    - make_loss(default) = default
  *    - make_loss(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L274
  * @return org.apache.mxnet.NDArray
  */
def make_loss(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Make your own loss function in network construction.
  * 
  * This operator accepts a customized loss function symbol as a terminal loss and
  * the symbol should be an operator with no backward dependency.
  * The output of this function is the gradient of loss with respect to the input data.
  * 
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::
  * 
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  *   loss = make_loss(cross_entropy)
  * 
  * We will need to use ``make_loss`` when we are creating our own loss function or we want to
  * combine multiple loss functions. Also we may want to stop some variables' gradients
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.
  * 
  * The storage type of ``make_loss`` output depends upon the input storage type:
  * 
  *    - make_loss(default) = default
  *    - make_loss(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L274
  * @return org.apache.mxnet.NDArray
  */
def make_loss(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the max of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161
  * @return org.apache.mxnet.NDArray
  */
def max(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the max of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161
  * @return org.apache.mxnet.NDArray
  */
def max(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the max of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161
  * @return org.apache.mxnet.NDArray
  */
def max_axis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the max of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161
  * @return org.apache.mxnet.NDArray
  */
def max_axis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the mean of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L102
  * @return org.apache.mxnet.NDArray
  */
def mean(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the mean of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L102
  * @return org.apache.mxnet.NDArray
  */
def mean(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the min of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def min(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the min of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def min(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the min of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def min_axis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the min of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L175
  * @return org.apache.mxnet.NDArray
  */
def min_axis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Updater function for multi-precision sgd optimizer
  * @return org.apache.mxnet.NDArray
  */
def mp_sgd_mom_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Updater function for multi-precision sgd optimizer
  * @return org.apache.mxnet.NDArray
  */
def mp_sgd_mom_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Updater function for multi-precision sgd optimizer
  * @return org.apache.mxnet.NDArray
  */
def mp_sgd_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Updater function for multi-precision sgd optimizer
  * @return org.apache.mxnet.NDArray
  */
def mp_sgd_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L147
  * @return org.apache.mxnet.NDArray
  */
def nanprod(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L147
  * @return org.apache.mxnet.NDArray
  */
def nanprod(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L132
  * @return org.apache.mxnet.NDArray
  */
def nansum(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L132
  * @return org.apache.mxnet.NDArray
  */
def nansum(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Numerical negative of the argument, element-wise.
  * 
  * The storage type of ``negative`` output depends upon the input storage type:
  * 
  *    - negative(default) = default
  *    - negative(row_sparse) = row_sparse
  *    - negative(csr) = csr
  * @return org.apache.mxnet.NDArray
  */
def negative(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Numerical negative of the argument, element-wise.
  * 
  * The storage type of ``negative`` output depends upon the input storage type:
  * 
  *    - negative(default) = default
  *    - negative(row_sparse) = row_sparse
  *    - negative(csr) = csr
  * @return org.apache.mxnet.NDArray
  */
def negative(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the norm on an NDArray.
  * 
  * This operator computes the norm on an NDArray with the specified axis, depending
  * on the value of the ord parameter. By default, it computes the L2 norm on the entire
  * array.
  * 
  * Examples::
  * 
  *   x = [[1, 2],
  *        [3, 4]]
  * 
  *   norm(x) = [5.47722578]
  * 
  *   rsp = x.cast_storage('row_sparse')
  * 
  *   norm(rsp) = [5.47722578]
  * 
  *   csr = x.cast_storage('csr')
  * 
  *   norm(csr) = [5.47722578]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L271
  * @return org.apache.mxnet.NDArray
  */
def norm(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the norm on an NDArray.
  * 
  * This operator computes the norm on an NDArray with the specified axis, depending
  * on the value of the ord parameter. By default, it computes the L2 norm on the entire
  * array.
  * 
  * Examples::
  * 
  *   x = [[1, 2],
  *        [3, 4]]
  * 
  *   norm(x) = [5.47722578]
  * 
  *   rsp = x.cast_storage('row_sparse')
  * 
  *   norm(rsp) = [5.47722578]
  * 
  *   csr = x.cast_storage('csr')
  * 
  *   norm(csr) = [5.47722578]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L271
  * @return org.apache.mxnet.NDArray
  */
def norm(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a normal (Gaussian) distribution.
  * 
  * .. note:: The existing alias ``normal`` is deprecated.
  * 
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).
  * 
  * Example::
  * 
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
  *                                           [-1.23474145,  1.55807114]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L85
  * @return org.apache.mxnet.NDArray
  */
def normal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a normal (Gaussian) distribution.
  * 
  * .. note:: The existing alias ``normal`` is deprecated.
  * 
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).
  * 
  * Example::
  * 
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
  *                                           [-1.23474145,  1.55807114]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L85
  * @return org.apache.mxnet.NDArray
  */
def normal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a one-hot array.
  * 
  * The locations represented by `indices` take value `on_value`, while all
  * other locations take value `off_value`.
  * 
  * `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
  * in an output array of shape ``(i0, i1, d)`` with::
  * 
  *   output[i,j,:] = off_value
  *   output[i,j,indices[i,j]] = on_value
  * 
  * Examples::
  * 
  *   one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
  *                            [ 1.  0.  0.]
  *                            [ 0.  0.  1.]
  *                            [ 1.  0.  0.]]
  * 
  *   one_hot([1,0,2,0], 3, on_value=8, off_value=1,
  *           dtype='int32') = [[1 8 1]
  *                             [8 1 1]
  *                             [1 1 8]
  *                             [8 1 1]]
  * 
  *   one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
  *                                       [ 1.  0.  0.]]
  * 
  *                                      [[ 0.  1.  0.]
  *                                       [ 1.  0.  0.]]
  * 
  *                                      [[ 0.  0.  1.]
  *                                       [ 1.  0.  0.]]]
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L480
  * @return org.apache.mxnet.NDArray
  */
def one_hot(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a one-hot array.
  * 
  * The locations represented by `indices` take value `on_value`, while all
  * other locations take value `off_value`.
  * 
  * `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result
  * in an output array of shape ``(i0, i1, d)`` with::
  * 
  *   output[i,j,:] = off_value
  *   output[i,j,indices[i,j]] = on_value
  * 
  * Examples::
  * 
  *   one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]
  *                            [ 1.  0.  0.]
  *                            [ 0.  0.  1.]
  *                            [ 1.  0.  0.]]
  * 
  *   one_hot([1,0,2,0], 3, on_value=8, off_value=1,
  *           dtype='int32') = [[1 8 1]
  *                             [8 1 1]
  *                             [1 1 8]
  *                             [8 1 1]]
  * 
  *   one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]
  *                                       [ 1.  0.  0.]]
  * 
  *                                      [[ 0.  1.  0.]
  *                                       [ 1.  0.  0.]]
  * 
  *                                      [[ 0.  0.  1.]
  *                                       [ 1.  0.  0.]]]
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L480
  * @return org.apache.mxnet.NDArray
  */
def one_hot(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return an array of ones with the same shape and type
  * as the input array.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  0.,  0.],
  *        [ 0.,  0.,  0.]]
  * 
  *   ones_like(x) = [[ 1.,  1.,  1.],
  *                   [ 1.,  1.,  1.]]
  * @return org.apache.mxnet.NDArray
  */
def ones_like(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return an array of ones with the same shape and type
  * as the input array.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  0.,  0.],
  *        [ 0.,  0.,  0.]]
  * 
  *   ones_like(x) = [[ 1.,  1.,  1.],
  *                   [ 1.,  1.,  1.]]
  * @return org.apache.mxnet.NDArray
  */
def ones_like(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pads an input array with a constant or edge values of the array.
  * 
  * .. note:: `Pad` is deprecated. Use `pad` instead.
  * 
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
  * 
  * This operation pads an input array with either a `constant_value` or edge values
  * along each axis of the input array. The amount of padding is specified by `pad_width`.
  * 
  * `pad_width` is a tuple of integer padding widths for each axis of the format
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
  * where ``N`` is the number of dimensions of the array.
  * 
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
  * to add before and after the elements of the array along dimension ``N``.
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
  * ``after_2`` must be 0.
  * 
  * Example::
  * 
  *    x = [[[[  1.   2.   3.]
  *           [  4.   5.   6.]]
  * 
  *          [[  7.   8.   9.]
  *           [ 10.  11.  12.]]]
  * 
  * 
  *         [[[ 11.  12.  13.]
  *           [ 14.  15.  16.]]
  * 
  *          [[ 17.  18.  19.]
  *           [ 20.  21.  22.]]]]
  * 
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  1.   1.   2.   3.   3.]
  *             [  1.   1.   2.   3.   3.]
  *             [  4.   4.   5.   6.   6.]
  *             [  4.   4.   5.   6.   6.]]
  * 
  *            [[  7.   7.   8.   9.   9.]
  *             [  7.   7.   8.   9.   9.]
  *             [ 10.  10.  11.  12.  12.]
  *             [ 10.  10.  11.  12.  12.]]]
  * 
  * 
  *           [[[ 11.  11.  12.  13.  13.]
  *             [ 11.  11.  12.  13.  13.]
  *             [ 14.  14.  15.  16.  16.]
  *             [ 14.  14.  15.  16.  16.]]
  * 
  *            [[ 17.  17.  18.  19.  19.]
  *             [ 17.  17.  18.  19.  19.]
  *             [ 20.  20.  21.  22.  22.]
  *             [ 20.  20.  21.  22.  22.]]]]
  * 
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  0.   0.   0.   0.   0.]
  *             [  0.   1.   2.   3.   0.]
  *             [  0.   4.   5.   6.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.   7.   8.   9.   0.]
  *             [  0.  10.  11.  12.   0.]
  *             [  0.   0.   0.   0.   0.]]]
  * 
  * 
  *           [[[  0.   0.   0.   0.   0.]
  *             [  0.  11.  12.  13.   0.]
  *             [  0.  14.  15.  16.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.  17.  18.  19.   0.]
  *             [  0.  20.  21.  22.   0.]
  *             [  0.   0.   0.   0.   0.]]]]
  * 
  * 
  * 
  * 
  * Defined in src/operator/pad.cc:L766
  * @return org.apache.mxnet.NDArray
  */
def pad(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Pads an input array with a constant or edge values of the array.
  * 
  * .. note:: `Pad` is deprecated. Use `pad` instead.
  * 
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.
  * 
  * This operation pads an input array with either a `constant_value` or edge values
  * along each axis of the input array. The amount of padding is specified by `pad_width`.
  * 
  * `pad_width` is a tuple of integer padding widths for each axis of the format
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``
  * where ``N`` is the number of dimensions of the array.
  * 
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values
  * to add before and after the elements of the array along dimension ``N``.
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,
  * ``after_2`` must be 0.
  * 
  * Example::
  * 
  *    x = [[[[  1.   2.   3.]
  *           [  4.   5.   6.]]
  * 
  *          [[  7.   8.   9.]
  *           [ 10.  11.  12.]]]
  * 
  * 
  *         [[[ 11.  12.  13.]
  *           [ 14.  15.  16.]]
  * 
  *          [[ 17.  18.  19.]
  *           [ 20.  21.  22.]]]]
  * 
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  1.   1.   2.   3.   3.]
  *             [  1.   1.   2.   3.   3.]
  *             [  4.   4.   5.   6.   6.]
  *             [  4.   4.   5.   6.   6.]]
  * 
  *            [[  7.   7.   8.   9.   9.]
  *             [  7.   7.   8.   9.   9.]
  *             [ 10.  10.  11.  12.  12.]
  *             [ 10.  10.  11.  12.  12.]]]
  * 
  * 
  *           [[[ 11.  11.  12.  13.  13.]
  *             [ 11.  11.  12.  13.  13.]
  *             [ 14.  14.  15.  16.  16.]
  *             [ 14.  14.  15.  16.  16.]]
  * 
  *            [[ 17.  17.  18.  19.  19.]
  *             [ 17.  17.  18.  19.  19.]
  *             [ 20.  20.  21.  22.  22.]
  *             [ 20.  20.  21.  22.  22.]]]]
  * 
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =
  * 
  *          [[[[  0.   0.   0.   0.   0.]
  *             [  0.   1.   2.   3.   0.]
  *             [  0.   4.   5.   6.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.   7.   8.   9.   0.]
  *             [  0.  10.  11.  12.   0.]
  *             [  0.   0.   0.   0.   0.]]]
  * 
  * 
  *           [[[  0.   0.   0.   0.   0.]
  *             [  0.  11.  12.  13.   0.]
  *             [  0.  14.  15.  16.   0.]
  *             [  0.   0.   0.   0.   0.]]
  * 
  *            [[  0.   0.   0.   0.   0.]
  *             [  0.  17.  18.  19.   0.]
  *             [  0.  20.  21.  22.   0.]
  *             [  0.   0.   0.   0.   0.]]]]
  * 
  * 
  * 
  * 
  * Defined in src/operator/pad.cc:L766
  * @return org.apache.mxnet.NDArray
  */
def pad(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Picks elements from an input array according to the input indices along the given axis.
  * 
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
  * an output array of shape ``(i0,)`` with::
  * 
  *   output[i] = input[i, indices[i]]
  * 
  * By default, if any index mentioned is too large, it is replaced by the index that addresses
  * the last element along an axis (the `clip` mode).
  * 
  * This function supports n-dimensional input and (n-1)-dimensional indices arrays.
  * 
  * Examples::
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // picks elements with specified indices along axis 0
  *   pick(x, y=[0,1], 0) = [ 1.,  4.]
  * 
  *   // picks elements with specified indices along axis 1
  *   pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
  * 
  *   y = [[ 1.],
  *        [ 0.],
  *        [ 2.]]
  * 
  *   // picks elements with specified indices along axis 1 and dims are maintained
  *   pick(x,y, 1, keepdims=True) = [[ 2.],
  *                                  [ 3.],
  *                                  [ 6.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L145
  * @return org.apache.mxnet.NDArray
  */
def pick(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Picks elements from an input array according to the input indices along the given axis.
  * 
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be
  * an output array of shape ``(i0,)`` with::
  * 
  *   output[i] = input[i, indices[i]]
  * 
  * By default, if any index mentioned is too large, it is replaced by the index that addresses
  * the last element along an axis (the `clip` mode).
  * 
  * This function supports n-dimensional input and (n-1)-dimensional indices arrays.
  * 
  * Examples::
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // picks elements with specified indices along axis 0
  *   pick(x, y=[0,1], 0) = [ 1.,  4.]
  * 
  *   // picks elements with specified indices along axis 1
  *   pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]
  * 
  *   y = [[ 1.],
  *        [ 0.],
  *        [ 2.]]
  * 
  *   // picks elements with specified indices along axis 1 and dims are maintained
  *   pick(x,y, 1, keepdims=True) = [[ 2.],
  *                                  [ 3.],
  *                                  [ 6.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L145
  * @return org.apache.mxnet.NDArray
  */
def pick(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the product of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L117
  * @return org.apache.mxnet.NDArray
  */
def prod(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the product of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L117
  * @return org.apache.mxnet.NDArray
  */
def prod(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Converts each element of the input array from degrees to radians.
  * 
  * .. math::
  *    radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
  * 
  * The storage type of ``radians`` output depends upon the input storage type:
  * 
  *    - radians(default) = default
  *    - radians(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L182
  * @return org.apache.mxnet.NDArray
  */
def radians(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Converts each element of the input array from degrees to radians.
  * 
  * .. math::
  *    radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]
  * 
  * The storage type of ``radians`` output depends upon the input storage type:
  * 
  *    - radians(default) = default
  *    - radians(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L182
  * @return org.apache.mxnet.NDArray
  */
def radians(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from an exponential distribution.
  * 
  * Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).
  * 
  * Example::
  * 
  *    exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
  *                                       [ 0.04146638,  0.31715935]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L115
  * @return org.apache.mxnet.NDArray
  */
def random_exponential(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from an exponential distribution.
  * 
  * Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).
  * 
  * Example::
  * 
  *    exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],
  *                                       [ 0.04146638,  0.31715935]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L115
  * @return org.apache.mxnet.NDArray
  */
def random_exponential(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a gamma distribution.
  * 
  * Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).
  * 
  * Example::
  * 
  *    gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
  *                                             [ 3.91697288,  3.65933681]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L100
  * @return org.apache.mxnet.NDArray
  */
def random_gamma(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a gamma distribution.
  * 
  * Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).
  * 
  * Example::
  * 
  *    gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],
  *                                             [ 3.91697288,  3.65933681]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L100
  * @return org.apache.mxnet.NDArray
  */
def random_gamma(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a generalized negative binomial distribution.
  * 
  * Samples are distributed according to a generalized negative binomial distribution parametrized by
  * *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
  * number of unsuccessful experiments (generalized to real numbers).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
  *                                                                     [ 6.,  4.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L168
  * @return org.apache.mxnet.NDArray
  */
def random_generalized_negative_binomial(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a generalized negative binomial distribution.
  * 
  * Samples are distributed according to a generalized negative binomial distribution parametrized by
  * *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the
  * number of unsuccessful experiments (generalized to real numbers).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],
  *                                                                     [ 6.,  4.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L168
  * @return org.apache.mxnet.NDArray
  */
def random_generalized_negative_binomial(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a negative binomial distribution.
  * 
  * Samples are distributed according to a negative binomial distribution parametrized by
  * *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
  *                                                  [ 2.,  5.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L149
  * @return org.apache.mxnet.NDArray
  */
def random_negative_binomial(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a negative binomial distribution.
  * 
  * Samples are distributed according to a negative binomial distribution parametrized by
  * *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],
  *                                                  [ 2.,  5.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L149
  * @return org.apache.mxnet.NDArray
  */
def random_negative_binomial(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a normal (Gaussian) distribution.
  * 
  * .. note:: The existing alias ``normal`` is deprecated.
  * 
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).
  * 
  * Example::
  * 
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
  *                                           [-1.23474145,  1.55807114]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L85
  * @return org.apache.mxnet.NDArray
  */
def random_normal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a normal (Gaussian) distribution.
  * 
  * .. note:: The existing alias ``normal`` is deprecated.
  * 
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).
  * 
  * Example::
  * 
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],
  *                                           [-1.23474145,  1.55807114]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L85
  * @return org.apache.mxnet.NDArray
  */
def random_normal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a Poisson distribution.
  * 
  * Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
  *                                   [ 4.,  6.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L132
  * @return org.apache.mxnet.NDArray
  */
def random_poisson(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a Poisson distribution.
  * 
  * Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).
  * Samples will always be returned as a floating point data type.
  * 
  * Example::
  * 
  *    poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],
  *                                   [ 4.,  6.]]
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L132
  * @return org.apache.mxnet.NDArray
  */
def random_poisson(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a uniform distribution.
  * 
  * .. note:: The existing alias ``uniform`` is deprecated.
  * 
  * Samples are uniformly distributed over the half-open interval *[low, high)*
  * (includes *low*, but excludes *high*).
  * 
  * Example::
  * 
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
  *                                           [ 0.54488319,  0.84725171]]
  * 
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L66
  * @return org.apache.mxnet.NDArray
  */
def random_uniform(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a uniform distribution.
  * 
  * .. note:: The existing alias ``uniform`` is deprecated.
  * 
  * Samples are uniformly distributed over the half-open interval *[low, high)*
  * (includes *low*, but excludes *high*).
  * 
  * Example::
  * 
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
  *                                           [ 0.54488319,  0.84725171]]
  * 
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L66
  * @return org.apache.mxnet.NDArray
  */
def random_uniform(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse cube-root value of the input.
  * 
  * .. math::
  *    rcbrt(x) = 1/\sqrt[3]{x}
  * 
  * Example::
  * 
  *    rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L693
  * @return org.apache.mxnet.NDArray
  */
def rcbrt(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse cube-root value of the input.
  * 
  * .. math::
  *    rcbrt(x) = 1/\sqrt[3]{x}
  * 
  * Example::
  * 
  *    rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L693
  * @return org.apache.mxnet.NDArray
  */
def rcbrt(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the reciprocal of the argument, element-wise.
  * 
  * Calculates 1/x.
  * 
  * Example::
  * 
  *     reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L438
  * @return org.apache.mxnet.NDArray
  */
def reciprocal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the reciprocal of the argument, element-wise.
  * 
  * Calculates 1/x.
  * 
  * Example::
  * 
  *     reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L438
  * @return org.apache.mxnet.NDArray
  */
def reciprocal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes rectified linear.
  * 
  * .. math::
  *    max(features, 0)
  * 
  * The storage type of ``relu`` output depends upon the input storage type:
  * 
  *    - relu(default) = default
  *    - relu(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L84
  * @return org.apache.mxnet.NDArray
  */
def relu(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes rectified linear.
  * 
  * .. math::
  *    max(features, 0)
  * 
  * The storage type of ``relu`` output depends upon the input storage type:
  * 
  *    - relu(default) = default
  *    - relu(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L84
  * @return org.apache.mxnet.NDArray
  */
def relu(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Repeats elements of an array.
  * 
  * By default, ``repeat`` flattens the input array into 1-D and then repeats the
  * elements::
  * 
  *   x = [[ 1, 2],
  *        [ 3, 4]]
  * 
  *   repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
  * 
  * The parameter ``axis`` specifies the axis along which to perform repeat::
  * 
  *   repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
  *                                   [ 3.,  3.,  4.,  4.]]
  * 
  *   repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
  *                                   [ 1.,  2.],
  *                                   [ 3.,  4.],
  *                                   [ 3.,  4.]]
  * 
  *   repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
  *                                    [ 3.,  3.,  4.,  4.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L615
  * @return org.apache.mxnet.NDArray
  */
def repeat(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Repeats elements of an array.
  * 
  * By default, ``repeat`` flattens the input array into 1-D and then repeats the
  * elements::
  * 
  *   x = [[ 1, 2],
  *        [ 3, 4]]
  * 
  *   repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]
  * 
  * The parameter ``axis`` specifies the axis along which to perform repeat::
  * 
  *   repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],
  *                                   [ 3.,  3.,  4.,  4.]]
  * 
  *   repeat(x, repeats=2, axis=0) = [[ 1.,  2.],
  *                                   [ 1.,  2.],
  *                                   [ 3.,  4.],
  *                                   [ 3.,  4.]]
  * 
  *   repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],
  *                                    [ 3.,  3.,  4.,  4.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L615
  * @return org.apache.mxnet.NDArray
  */
def repeat(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshapes the input array.
  * 
  * .. note:: ``Reshape`` is deprecated, use ``reshape``
  * 
  * Given an array and a shape, this function returns a copy of the array in the new shape.
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.
  * 
  * Example::
  * 
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
  * 
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:
  * 
  * - ``0``  copy this dimension from the input to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
  * 
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  *   keeping the size of the new array same as that of the input array.
  *   At most one dimension of shape can be -1.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)
  * 
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
  * 
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
  * 
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
  * 
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.
  * 
  *   Example::
  * 
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
  *   - with reverse=1, output shape will be (50,4).
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L167
  * @return org.apache.mxnet.NDArray
  */
def reshape(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshapes the input array.
  * 
  * .. note:: ``Reshape`` is deprecated, use ``reshape``
  * 
  * Given an array and a shape, this function returns a copy of the array in the new shape.
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.
  * 
  * Example::
  * 
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]
  * 
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:
  * 
  * - ``0``  copy this dimension from the input to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)
  * 
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions
  *   keeping the size of the new array same as that of the input array.
  *   At most one dimension of shape can be -1.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)
  * 
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)
  * 
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)
  * 
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).
  * 
  *   Example::
  * 
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)
  * 
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.
  * 
  *   Example::
  * 
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)
  *   - with reverse=1, output shape will be (50,4).
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L167
  * @return org.apache.mxnet.NDArray
  */
def reshape(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshape lhs to have the same shape as rhs.
  * @return org.apache.mxnet.NDArray
  */
def reshape_like(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reshape lhs to have the same shape as rhs.
  * @return org.apache.mxnet.NDArray
  */
def reshape_like(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the order of elements along given axis while preserving array shape.
  * 
  * Note: reverse and flip are equivalent. We use reverse in the following examples.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.,  3.,  4.],
  *        [ 5.,  6.,  7.,  8.,  9.]]
  * 
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
  *                         [ 0.,  1.,  2.,  3.,  4.]]
  * 
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
  *                         [ 9.,  8.,  7.,  6.,  5.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L717
  * @return org.apache.mxnet.NDArray
  */
def reverse(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Reverses the order of elements along given axis while preserving array shape.
  * 
  * Note: reverse and flip are equivalent. We use reverse in the following examples.
  * 
  * Examples::
  * 
  *   x = [[ 0.,  1.,  2.,  3.,  4.],
  *        [ 5.,  6.,  7.,  8.,  9.]]
  * 
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],
  *                         [ 0.,  1.,  2.,  3.,  4.]]
  * 
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],
  *                         [ 9.,  8.,  7.,  6.,  5.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L717
  * @return org.apache.mxnet.NDArray
  */
def reverse(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest integer of the input.
  * 
  * .. note::
  *    - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
  *    - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
  * 
  * Example::
  * 
  *    rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
  * 
  * The storage type of ``rint`` output depends upon the input storage type:
  * 
  *    - rint(default) = default
  *    - rint(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L519
  * @return org.apache.mxnet.NDArray
  */
def rint(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest integer of the input.
  * 
  * .. note::
  *    - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.
  *    - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.
  * 
  * Example::
  * 
  *    rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]
  * 
  * The storage type of ``rint`` output depends upon the input storage type:
  * 
  *    - rint(default) = default
  *    - rint(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L519
  * @return org.apache.mxnet.NDArray
  */
def rint(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for `RMSProp` optimizer.
  * 
  * `RMSprop` is a variant of stochastic gradient descent where the gradients are
  * divided by a cache which grows with the sum of squares of recent gradients?
  * 
  * `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
  * tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for
  * each parameter monotonically over the course of training.
  * While this is analytically motivated for convex optimizations, it may not be ideal
  * for non-convex problems. `RMSProp` deals with this heuristically by allowing the
  * learning rates to rebound as the denominator decays over time.
  * 
  * Define the Root Mean Square (RMS) error criterion of the gradient as
  * :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
  * gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.
  * 
  * The :math:`E[g^2]_t` is given by:
  * 
  * .. math::
  *   E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
  * 
  * The update step is
  * 
  * .. math::
  *   \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
  * 
  * The RMSProp code follows the version in
  * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
  * Tieleman & Hinton, 2012.
  * 
  * Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
  * :math:`\eta` to be 0.001.
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L512
  * @return org.apache.mxnet.NDArray
  */
def rmsprop_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for `RMSProp` optimizer.
  * 
  * `RMSprop` is a variant of stochastic gradient descent where the gradients are
  * divided by a cache which grows with the sum of squares of recent gradients?
  * 
  * `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively
  * tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for
  * each parameter monotonically over the course of training.
  * While this is analytically motivated for convex optimizations, it may not be ideal
  * for non-convex problems. `RMSProp` deals with this heuristically by allowing the
  * learning rates to rebound as the denominator decays over time.
  * 
  * Define the Root Mean Square (RMS) error criterion of the gradient as
  * :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents
  * gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.
  * 
  * The :math:`E[g^2]_t` is given by:
  * 
  * .. math::
  *   E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2
  * 
  * The update step is
  * 
  * .. math::
  *   \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t
  * 
  * The RMSProp code follows the version in
  * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
  * Tieleman & Hinton, 2012.
  * 
  * Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate
  * :math:`\eta` to be 0.001.
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L512
  * @return org.apache.mxnet.NDArray
  */
def rmsprop_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for RMSPropAlex optimizer.
  * 
  * `RMSPropAlex` is non-centered version of `RMSProp`.
  * 
  * Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
  * :math:`E[g]_t` is the decaying average over past gradient.
  * 
  * .. math::
  *   E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
  *   E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
  *   \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\
  * 
  * The update step is
  * 
  * .. math::
  *   \theta_{t+1} = \theta_t + \Delta_t
  * 
  * The RMSPropAlex code follows the version in
  * http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
  * 
  * Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
  * to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L551
  * @return org.apache.mxnet.NDArray
  */
def rmspropalex_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for RMSPropAlex optimizer.
  * 
  * `RMSPropAlex` is non-centered version of `RMSProp`.
  * 
  * Define :math:`E[g^2]_t` is the decaying average over past squared gradient and
  * :math:`E[g]_t` is the decaying average over past gradient.
  * 
  * .. math::
  *   E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\
  *   E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\
  *   \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\
  * 
  * The update step is
  * 
  * .. math::
  *   \theta_{t+1} = \theta_t + \Delta_t
  * 
  * The RMSPropAlex code follows the version in
  * http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.
  * 
  * Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`
  * to be 0.9 and the learning rate :math:`\eta` to be 0.0001.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L551
  * @return org.apache.mxnet.NDArray
  */
def rmspropalex_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest integer of the input.
  * 
  * Example::
  * 
  *    round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
  * 
  * The storage type of ``round`` output depends upon the input storage type:
  * 
  *   - round(default) = default
  *   - round(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L498
  * @return org.apache.mxnet.NDArray
  */
def round(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise rounded value to the nearest integer of the input.
  * 
  * Example::
  * 
  *    round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]
  * 
  * The storage type of ``round`` output depends upon the input storage type:
  * 
  *   - round(default) = default
  *   - round(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L498
  * @return org.apache.mxnet.NDArray
  */
def round(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse square-root value of the input.
  * 
  * .. math::
  *    rsqrt(x) = 1/\sqrt{x}
  * 
  * Example::
  * 
  *    rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
  * 
  * The storage type of ``rsqrt`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L659
  * @return org.apache.mxnet.NDArray
  */
def rsqrt(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise inverse square-root value of the input.
  * 
  * .. math::
  *    rsqrt(x) = 1/\sqrt{x}
  * 
  * Example::
  * 
  *    rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]
  * 
  * The storage type of ``rsqrt`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L659
  * @return org.apache.mxnet.NDArray
  */
def rsqrt(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * exponential distributions with parameters lambda (rate).
  * 
  * The parameters of the distributions are provided as an input array.
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input value at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input array.
  * 
  * Examples::
  * 
  *    lam = [ 1.0, 8.5 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_exponential(lam) = [ 0.51837951,  0.09994757]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
  *                                          [ 0.09994757,  0.50447971]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L284
  * @return org.apache.mxnet.NDArray
  */
def sample_exponential(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * exponential distributions with parameters lambda (rate).
  * 
  * The parameters of the distributions are provided as an input array.
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input value at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input array.
  * 
  * Examples::
  * 
  *    lam = [ 1.0, 8.5 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_exponential(lam) = [ 0.51837951,  0.09994757]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],
  *                                          [ 0.09994757,  0.50447971]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L284
  * @return org.apache.mxnet.NDArray
  */
def sample_exponential(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * gamma distributions with parameters *alpha* (shape) and *beta* (scale).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    alpha = [ 0.0, 2.5 ]
  *    beta = [ 1.0, 0.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
  *                                            [ 2.25797319,  1.70734084]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L282
  * @return org.apache.mxnet.NDArray
  */
def sample_gamma(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * gamma distributions with parameters *alpha* (shape) and *beta* (scale).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    alpha = [ 0.0, 2.5 ]
  *    beta = [ 1.0, 0.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],
  *                                            [ 2.25797319,  1.70734084]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L282
  * @return org.apache.mxnet.NDArray
  */
def sample_gamma(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    mu = [ 2.0, 2.5 ]
  *    alpha = [ 1.0, 0.1 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
  *                                                                  [ 3.,  1.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def sample_generalized_negative_binomial(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    mu = [ 2.0, 2.5 ]
  *    alpha = [ 1.0, 0.1 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],
  *                                                                  [ 3.,  1.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def sample_generalized_negative_binomial(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple multinomial distributions.
  * 
  * *data* is an *n* dimensional array whose last dimension has length *k*, where
  * *k* is the number of possible outcomes of each multinomial distribution. This
  * operator will draw *shape* samples from each distribution. If shape is empty
  * one sample will be drawn from each distribution.
  * 
  * If *get_prob* is true, a second array containing log likelihood of the drawn
  * samples will also be returned. This is usually used for reinforcement learning
  * where you can provide reward as head gradient for this array to estimate
  * gradient.
  * 
  * Note that the input distribution must be normalized, i.e. *data* must sum to
  * 1 along its last axis.
  * 
  * Examples::
  * 
  *    probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]
  * 
  *    // Draw a single sample for each distribution
  *    sample_multinomial(probs) = [3, 0]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_multinomial(probs, shape=(2)) = [[4, 2],
  *                                            [0, 0]]
  * 
  *    // requests log likelihood
  *    sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
  * @return org.apache.mxnet.NDArray
  */
def sample_multinomial(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple multinomial distributions.
  * 
  * *data* is an *n* dimensional array whose last dimension has length *k*, where
  * *k* is the number of possible outcomes of each multinomial distribution. This
  * operator will draw *shape* samples from each distribution. If shape is empty
  * one sample will be drawn from each distribution.
  * 
  * If *get_prob* is true, a second array containing log likelihood of the drawn
  * samples will also be returned. This is usually used for reinforcement learning
  * where you can provide reward as head gradient for this array to estimate
  * gradient.
  * 
  * Note that the input distribution must be normalized, i.e. *data* must sum to
  * 1 along its last axis.
  * 
  * Examples::
  * 
  *    probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]
  * 
  *    // Draw a single sample for each distribution
  *    sample_multinomial(probs) = [3, 0]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_multinomial(probs, shape=(2)) = [[4, 2],
  *                                            [0, 0]]
  * 
  *    // requests log likelihood
  *    sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]
  * @return org.apache.mxnet.NDArray
  */
def sample_multinomial(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    k = [ 20, 49 ]
  *    p = [ 0.4 , 0.77 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_negative_binomial(k, p) = [ 15.,  16.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
  *                                                 [ 16.,  12.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L289
  * @return org.apache.mxnet.NDArray
  */
def sample_negative_binomial(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    k = [ 20, 49 ]
  *    p = [ 0.4 , 0.77 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_negative_binomial(k, p) = [ 15.,  16.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],
  *                                                 [ 16.,  12.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L289
  * @return org.apache.mxnet.NDArray
  */
def sample_negative_binomial(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    mu = [ 0.0, 2.5 ]
  *    sigma = [ 1.0, 3.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_normal(mu, sigma) = [-0.56410581,  0.95934606]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
  *                                           [ 0.95934606,  4.48287058]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L279
  * @return org.apache.mxnet.NDArray
  */
def sample_normal(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    mu = [ 0.0, 2.5 ]
  *    sigma = [ 1.0, 3.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_normal(mu, sigma) = [-0.56410581,  0.95934606]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],
  *                                           [ 0.95934606,  4.48287058]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L279
  * @return org.apache.mxnet.NDArray
  */
def sample_normal(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * Poisson distributions with parameters lambda (rate).
  * 
  * The parameters of the distributions are provided as an input array.
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input value at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input array.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    lam = [ 1.0, 8.5 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_poisson(lam) = [  0.,  13.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_poisson(lam, shape=(2)) = [[  0.,   4.],
  *                                      [ 13.,   8.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L286
  * @return org.apache.mxnet.NDArray
  */
def sample_poisson(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * Poisson distributions with parameters lambda (rate).
  * 
  * The parameters of the distributions are provided as an input array.
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input value at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input array.
  * 
  * Samples will always be returned as a floating point data type.
  * 
  * Examples::
  * 
  *    lam = [ 1.0, 8.5 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_poisson(lam) = [  0.,  13.]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_poisson(lam, shape=(2)) = [[  0.,   4.],
  *                                      [ 13.,   8.]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L286
  * @return org.apache.mxnet.NDArray
  */
def sample_poisson(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * uniform distributions on the intervals given by *[low,high)*.
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    low = [ 0.0, 2.5 ]
  *    high = [ 1.0, 3.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_uniform(low, high) = [ 0.40451524,  3.18687344]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
  *                                            [ 3.18687344,  3.68352246]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L277
  * @return org.apache.mxnet.NDArray
  */
def sample_uniform(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Concurrent sampling from multiple
  * uniform distributions on the intervals given by *[low,high)*.
  * 
  * The parameters of the distributions are provided as input arrays.
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*
  * be the shape specified as the parameter of the operator, and *m* be the dimension
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.
  * 
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution
  * which is parameterized by the input values at index *i*. If the shape parameter of the
  * operator is not set, then one sample will be drawn per distribution and the output array
  * has the same shape as the input arrays.
  * 
  * Examples::
  * 
  *    low = [ 0.0, 2.5 ]
  *    high = [ 1.0, 3.7 ]
  * 
  *    // Draw a single sample for each distribution
  *    sample_uniform(low, high) = [ 0.40451524,  3.18687344]
  * 
  *    // Draw a vector containing two samples for each distribution
  *    sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],
  *                                            [ 3.18687344,  3.68352246]]
  * 
  * 
  * Defined in src/operator/random/multisample_op.cc:L277
  * @return org.apache.mxnet.NDArray
  */
def sample_uniform(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Scatters data into a new tensor according to indices.
  * 
  * Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
  * where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.
  * 
  * The elements in output is defined as follows::
  * 
  *   output[indices[0, y_0, ..., y_{K-1}],
  *          ...,
  *          indices[M-1, y_0, ..., y_{K-1}],
  *          x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]
  * 
  * all other entries in output are 0.
  * 
  * .. warning::
  * 
  *     If the indices have duplicates, the result will be non-deterministic and
  *     the gradient of `scatter_nd` will not be correct!!
  * 
  * 
  * Examples::
  * 
  *   data = [2, 3, 0]
  *   indices = [[1, 1, 0], [0, 1, 0]]
  *   shape = (2, 2)
  *   scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]
  * @return org.apache.mxnet.NDArray
  */
def scatter_nd(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Scatters data into a new tensor according to indices.
  * 
  * Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,
  * where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.
  * 
  * The elements in output is defined as follows::
  * 
  *   output[indices[0, y_0, ..., y_{K-1}],
  *          ...,
  *          indices[M-1, y_0, ..., y_{K-1}],
  *          x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]
  * 
  * all other entries in output are 0.
  * 
  * .. warning::
  * 
  *     If the indices have duplicates, the result will be non-deterministic and
  *     the gradient of `scatter_nd` will not be correct!!
  * 
  * 
  * Examples::
  * 
  *   data = [2, 3, 0]
  *   indices = [[1, 1, 0], [0, 1, 0]]
  *   shape = (2, 2)
  *   scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]
  * @return org.apache.mxnet.NDArray
  */
def scatter_nd(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
  * 
  * Momentum update has better convergence rates on neural networks. Mathematically it looks
  * like below:
  * 
  * .. math::
  * 
  *   v_1 = \alpha * \nabla J(W_0)\\
  *   v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  *   W_t = W_{t-1} + v_t
  * 
  * It updates the weights using::
  * 
  *   v = momentum * v - learning_rate * gradient
  *   weight += v
  * 
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.
  * 
  * If weight and grad are both of ``row_sparse`` storage type and momentum is of ``default`` storage type,
  * standard update is applied.
  * 
  * If weight, grad and momentum are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::
  * 
  *   for row in gradient.indices:
  *       v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
  *       weight[row] += v[row]
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L336
  * @return org.apache.mxnet.NDArray
  */
def sgd_mom_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Momentum update function for Stochastic Gradient Descent (SDG) optimizer.
  * 
  * Momentum update has better convergence rates on neural networks. Mathematically it looks
  * like below:
  * 
  * .. math::
  * 
  *   v_1 = \alpha * \nabla J(W_0)\\
  *   v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\
  *   W_t = W_{t-1} + v_t
  * 
  * It updates the weights using::
  * 
  *   v = momentum * v - learning_rate * gradient
  *   weight += v
  * 
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.
  * 
  * If weight and grad are both of ``row_sparse`` storage type and momentum is of ``default`` storage type,
  * standard update is applied.
  * 
  * If weight, grad and momentum are all of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::
  * 
  *   for row in gradient.indices:
  *       v[row] = momentum[row] * v[row] - learning_rate * gradient[row]
  *       weight[row] += v[row]
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L336
  * @return org.apache.mxnet.NDArray
  */
def sgd_mom_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Stochastic Gradient Descent (SDG) optimizer.
  * 
  * It updates the weights using::
  * 
  *  weight = weight - learning_rate * gradient
  * 
  * If weight is of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated::
  * 
  *  for row in gradient.indices:
  *      weight[row] = weight[row] - learning_rate * gradient[row]
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def sgd_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for Stochastic Gradient Descent (SDG) optimizer.
  * 
  * It updates the weights using::
  * 
  *  weight = weight - learning_rate * gradient
  * 
  * If weight is of ``row_sparse`` storage type,
  * only the row slices whose indices appear in grad.indices are updated::
  * 
  *  for row in gradient.indices:
  *      weight[row] = weight[row] - learning_rate * gradient[row]
  * 
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L293
  * @return org.apache.mxnet.NDArray
  */
def sgd_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Randomly shuffle the elements.
  * 
  * This shuffles the array along the first axis.
  * The order of the elements in each subarray does not change.
  * For example, if a 2D array is given, the order of the rows randomly changes,
  * but the order of the elements in each row does not change.
  * @return org.apache.mxnet.NDArray
  */
def shuffle(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Randomly shuffle the elements.
  * 
  * This shuffles the array along the first axis.
  * The order of the elements in each subarray does not change.
  * For example, if a 2D array is given, the order of the rows randomly changes,
  * but the order of the elements in each row does not change.
  * @return org.apache.mxnet.NDArray
  */
def shuffle(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes sigmoid of x element-wise.
  * 
  * .. math::
  *    y = 1 / (1 + exp(-x))
  * 
  * The storage type of ``sigmoid`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L103
  * @return org.apache.mxnet.NDArray
  */
def sigmoid(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes sigmoid of x element-wise.
  * 
  * .. math::
  *    y = 1 / (1 + exp(-x))
  * 
  * The storage type of ``sigmoid`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L103
  * @return org.apache.mxnet.NDArray
  */
def sigmoid(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sign of the input.
  * 
  * Example::
  * 
  *    sign([-2, 0, 3]) = [-1, 0, 1]
  * 
  * The storage type of ``sign`` output depends upon the input storage type:
  * 
  *    - sign(default) = default
  *    - sign(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L479
  * @return org.apache.mxnet.NDArray
  */
def sign(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise sign of the input.
  * 
  * Example::
  * 
  *    sign([-2, 0, 3]) = [-1, 0, 1]
  * 
  * The storage type of ``sign`` output depends upon the input storage type:
  * 
  *    - sign(default) = default
  *    - sign(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L479
  * @return org.apache.mxnet.NDArray
  */
def sign(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for SignSGD optimizer.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  W_t = W_{t-1} - \eta_t \text{sign}(g_t)
  * 
  * It updates the weights using::
  * 
  *  weight = weight - learning_rate * sign(gradient)
  * 
  * .. note:: 
  *    - sparse ndarray not supported for this optimizer yet.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L57
  * @return org.apache.mxnet.NDArray
  */
def signsgd_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Update function for SignSGD optimizer.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  W_t = W_{t-1} - \eta_t \text{sign}(g_t)
  * 
  * It updates the weights using::
  * 
  *  weight = weight - learning_rate * sign(gradient)
  * 
  * .. note:: 
  *    - sparse ndarray not supported for this optimizer yet.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L57
  * @return org.apache.mxnet.NDArray
  */
def signsgd_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * SIGN momentUM (Signum) optimizer.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  m_t = \beta m_{t-1} + (1 - \beta) g_t\\
  *  W_t = W_{t-1} - \eta_t \text{sign}(m_t)
  * 
  * It updates the weights using::
  *  state = momentum * state + (1-momentum) * gradient
  *  weight = weight - learning_rate * sign(state)
  * 
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.
  * 
  * .. note:: 
  *    - sparse ndarray not supported for this optimizer yet.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def signum_update(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * SIGN momentUM (Signum) optimizer.
  * 
  * .. math::
  * 
  *  g_t = \nabla J(W_{t-1})\\
  *  m_t = \beta m_{t-1} + (1 - \beta) g_t\\
  *  W_t = W_{t-1} - \eta_t \text{sign}(m_t)
  * 
  * It updates the weights using::
  *  state = momentum * state + (1-momentum) * gradient
  *  weight = weight - learning_rate * sign(state)
  * 
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.
  * 
  * .. note:: 
  *    - sparse ndarray not supported for this optimizer yet.
  * 
  * 
  * Defined in src/operator/optimizer_op.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def signum_update(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise sine of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
  * 
  * The storage type of ``sin`` output depends upon the input storage type:
  * 
  *    - sin(default) = default
  *    - sin(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L46
  * @return org.apache.mxnet.NDArray
  */
def sin(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise sine of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]
  * 
  * The storage type of ``sin`` output depends upon the input storage type:
  * 
  *    - sin(default) = default
  *    - sin(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L46
  * @return org.apache.mxnet.NDArray
  */
def sin(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic sine of the input array, computed element-wise.
  * 
  * .. math::
  *    sinh(x) = 0.5\times(exp(x) - exp(-x))
  * 
  * The storage type of ``sinh`` output depends upon the input storage type:
  * 
  *    - sinh(default) = default
  *    - sinh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L201
  * @return org.apache.mxnet.NDArray
  */
def sinh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic sine of the input array, computed element-wise.
  * 
  * .. math::
  *    sinh(x) = 0.5\times(exp(x) - exp(-x))
  * 
  * The storage type of ``sinh`` output depends upon the input storage type:
  * 
  *    - sinh(default) = default
  *    - sinh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L201
  * @return org.apache.mxnet.NDArray
  */
def sinh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices a region of the array.
  * 
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.
  * 
  * This function returns a sliced array between the indices given
  * by `begin` and `end` with the corresponding `step`.
  * 
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
  * where m <= n, results in an array with the shape
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
  * 
  * The resulting array's *k*-th dimension contains elements
  * from the *k*-th dimension of the input array starting
  * from index ``b_k`` (inclusive) with step ``s_k``
  * until reaching ``e_k`` (exclusive).
  * 
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,
  * and `step`, the following rule will be used to set default values.
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
  * else, set `b_k=d_k-1`, `e_k=-1`.
  * 
  * The storage type of ``slice`` output depends on storage types of inputs
  * 
  * - slice(csr) = csr
  * - otherwise, ``slice`` generates output with default storage
  * 
  * .. note:: When input data storage type is csr, it only supports
  * step=(), or step=(None,), or step=(1,) to generate a csr output.
  * For other step parameter values, it falls back to slicing
  * a dense tensor.
  * 
  * Example::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
  *                                      [ 6.,  7.,  8.]]
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
  *                                                             [5.,  7.],
  *                                                             [1.,  3.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L411
  * @return org.apache.mxnet.NDArray
  */
def slice(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices a region of the array.
  * 
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.
  * 
  * This function returns a sliced array between the indices given
  * by `begin` and `end` with the corresponding `step`.
  * 
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,
  * where m <= n, results in an array with the shape
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.
  * 
  * The resulting array's *k*-th dimension contains elements
  * from the *k*-th dimension of the input array starting
  * from index ``b_k`` (inclusive) with step ``s_k``
  * until reaching ``e_k`` (exclusive).
  * 
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,
  * and `step`, the following rule will be used to set default values.
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;
  * else, set `b_k=d_k-1`, `e_k=-1`.
  * 
  * The storage type of ``slice`` output depends on storage types of inputs
  * 
  * - slice(csr) = csr
  * - otherwise, ``slice`` generates output with default storage
  * 
  * .. note:: When input data storage type is csr, it only supports
  * step=(), or step=(None,), or step=(1,) to generate a csr output.
  * For other step parameter values, it falls back to slicing
  * a dense tensor.
  * 
  * Example::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],
  *                                      [ 6.,  7.,  8.]]
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],
  *                                                             [5.,  7.],
  *                                                             [1.,  3.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L411
  * @return org.apache.mxnet.NDArray
  */
def slice(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices along a given axis.
  * 
  * Returns an array slice along a given `axis` starting from the `begin` index
  * to the `end` index.
  * 
  * Examples::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
  *                                            [  9.,  10.,  11.,  12.]]
  * 
  *   slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
  *                                            [  5.,   6.],
  *                                            [  9.,  10.]]
  * 
  *   slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
  *                                              [  6.,   7.],
  *                                              [ 10.,  11.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L498
  * @return org.apache.mxnet.NDArray
  */
def slice_axis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Slices along a given axis.
  * 
  * Returns an array slice along a given `axis` starting from the `begin` index
  * to the `end` index.
  * 
  * Examples::
  * 
  *   x = [[  1.,   2.,   3.,   4.],
  *        [  5.,   6.,   7.,   8.],
  *        [  9.,  10.,  11.,  12.]]
  * 
  *   slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],
  *                                            [  9.,  10.,  11.,  12.]]
  * 
  *   slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],
  *                                            [  5.,   6.],
  *                                            [  9.,  10.]]
  * 
  *   slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],
  *                                              [  6.,   7.],
  *                                              [ 10.,  11.]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L498
  * @return org.apache.mxnet.NDArray
  */
def slice_axis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Calculate Smooth L1 Loss(lhs, scalar) by summing
  * 
  * .. math::
  * 
  *     f(x) =
  *     \begin{cases}
  *     (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
  *     |x|-0.5/\sigma^2,& \text{otherwise}
  *     \end{cases}
  * 
  * where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.
  * 
  * Example::
  * 
  *   smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L103
  * @return org.apache.mxnet.NDArray
  */
def smooth_l1(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Calculate Smooth L1 Loss(lhs, scalar) by summing
  * 
  * .. math::
  * 
  *     f(x) =
  *     \begin{cases}
  *     (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\
  *     |x|-0.5/\sigma^2,& \text{otherwise}
  *     \end{cases}
  * 
  * where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.
  * 
  * Example::
  * 
  *   smooth_l1([1, 2, 3, 4], sigma=1) = [0.5, 1.5, 2.5, 3.5]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L103
  * @return org.apache.mxnet.NDArray
  */
def smooth_l1(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies the softmax function.
  * 
  * The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.
  * 
  * .. math::
  *    softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
  * 
  * for :math:`j = 1, ..., K`
  * 
  * Example::
  * 
  *   x = [[ 1.  1.  1.]
  *        [ 1.  1.  1.]]
  * 
  *   softmax(x,axis=0) = [[ 0.5  0.5  0.5]
  *                        [ 0.5  0.5  0.5]]
  * 
  *   softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
  *                        [ 0.33333334,  0.33333334,  0.33333334]]
  * 
  * 
  * 
  * Defined in src/operator/nn/softmax.cc:L97
  * @return org.apache.mxnet.NDArray
  */
def softmax(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Applies the softmax function.
  * 
  * The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.
  * 
  * .. math::
  *    softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}
  * 
  * for :math:`j = 1, ..., K`
  * 
  * Example::
  * 
  *   x = [[ 1.  1.  1.]
  *        [ 1.  1.  1.]]
  * 
  *   softmax(x,axis=0) = [[ 0.5  0.5  0.5]
  *                        [ 0.5  0.5  0.5]]
  * 
  *   softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],
  *                        [ 0.33333334,  0.33333334,  0.33333334]]
  * 
  * 
  * 
  * Defined in src/operator/nn/softmax.cc:L97
  * @return org.apache.mxnet.NDArray
  */
def softmax(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Calculate cross entropy of softmax output and one-hot label.
  * 
  * - This operator computes the cross entropy in two steps:
  *   - Applies softmax function on the input array.
  *   - Computes and returns the cross entropy loss between the softmax output and the labels.
  * 
  * - The softmax function and cross entropy loss is given by:
  * 
  *   - Softmax Function:
  * 
  *   .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
  * 
  *   - Cross Entropy Function:
  * 
  *   .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)
  * 
  * Example::
  * 
  *   x = [[1, 2, 3],
  *        [11, 7, 5]]
  * 
  *   label = [2, 0]
  * 
  *   softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
  *                 [0.97962922, 0.01794253, 0.00242826]]
  * 
  *   softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871
  * 
  * 
  * 
  * Defined in src/operator/loss_binary_op.cc:L59
  * @return org.apache.mxnet.NDArray
  */
def softmax_cross_entropy(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Calculate cross entropy of softmax output and one-hot label.
  * 
  * - This operator computes the cross entropy in two steps:
  *   - Applies softmax function on the input array.
  *   - Computes and returns the cross entropy loss between the softmax output and the labels.
  * 
  * - The softmax function and cross entropy loss is given by:
  * 
  *   - Softmax Function:
  * 
  *   .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}
  * 
  *   - Cross Entropy Function:
  * 
  *   .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)
  * 
  * Example::
  * 
  *   x = [[1, 2, 3],
  *        [11, 7, 5]]
  * 
  *   label = [2, 0]
  * 
  *   softmax(x) = [[0.09003057, 0.24472848, 0.66524094],
  *                 [0.97962922, 0.01794253, 0.00242826]]
  * 
  *   softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871
  * 
  * 
  * 
  * Defined in src/operator/loss_binary_op.cc:L59
  * @return org.apache.mxnet.NDArray
  */
def softmax_cross_entropy(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes softsign of x element-wise.
  * 
  * .. math::
  *    y = x / (1 + abs(x))
  * 
  * The storage type of ``softsign`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L119
  * @return org.apache.mxnet.NDArray
  */
def softsign(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes softsign of x element-wise.
  * 
  * .. math::
  *    y = x / (1 + abs(x))
  * 
  * The storage type of ``softsign`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L119
  * @return org.apache.mxnet.NDArray
  */
def softsign(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a sorted copy of an input array along the given axis.
  * 
  * Examples::
  * 
  *   x = [[ 1, 4],
  *        [ 3, 1]]
  * 
  *   // sorts along the last axis
  *   sort(x) = [[ 1.,  4.],
  *              [ 1.,  3.]]
  * 
  *   // flattens and then sorts
  *   sort(x) = [ 1.,  1.,  3.,  4.]
  * 
  *   // sorts along the first axis
  *   sort(x, axis=0) = [[ 1.,  1.],
  *                      [ 3.,  4.]]
  * 
  *   // in a descend order
  *   sort(x, is_ascend=0) = [[ 4.,  1.],
  *                           [ 3.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L126
  * @return org.apache.mxnet.NDArray
  */
def sort(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns a sorted copy of an input array along the given axis.
  * 
  * Examples::
  * 
  *   x = [[ 1, 4],
  *        [ 3, 1]]
  * 
  *   // sorts along the last axis
  *   sort(x) = [[ 1.,  4.],
  *              [ 1.,  3.]]
  * 
  *   // flattens and then sorts
  *   sort(x) = [ 1.,  1.,  3.,  4.]
  * 
  *   // sorts along the first axis
  *   sort(x, axis=0) = [[ 1.,  1.],
  *                      [ 3.,  4.]]
  * 
  *   // in a descend order
  *   sort(x, is_ascend=0) = [[ 4.,  1.],
  *                           [ 3.,  1.]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L126
  * @return org.apache.mxnet.NDArray
  */
def sort(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Splits an array along a particular axis into multiple sub-arrays.
  * 
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
  * 
  * **Note** that `num_outputs` should evenly divide the length of the axis
  * along which to split the array.
  * 
  * Example::
  * 
  *    x  = [[[ 1.]
  *           [ 2.]]
  *          [[ 3.]
  *           [ 4.]]
  *          [[ 5.]
  *           [ 6.]]]
  *    x.shape = (3, 2, 1)
  * 
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
  *    y = [[[ 1.]]
  *         [[ 3.]]
  *         [[ 5.]]]
  * 
  *        [[[ 2.]]
  *         [[ 4.]]
  *         [[ 6.]]]
  * 
  *    y[0].shape = (3, 1, 1)
  * 
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
  *    z = [[[ 1.]
  *          [ 2.]]]
  * 
  *        [[[ 3.]
  *          [ 4.]]]
  * 
  *        [[[ 5.]
  *          [ 6.]]]
  * 
  *    z[0].shape = (1, 2, 1)
  * 
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
  * along the `axis` which it is split.
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.
  * 
  * Example::
  * 
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
  *    z = [[ 1.]
  *         [ 2.]]
  * 
  *        [[ 3.]
  *         [ 4.]]
  * 
  *        [[ 5.]
  *         [ 6.]]
  *    z[0].shape = (2 ,1 )
  * 
  * 
  * 
  * Defined in src/operator/slice_channel.cc:L107
  * @return org.apache.mxnet.NDArray
  */
def split(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Splits an array along a particular axis into multiple sub-arrays.
  * 
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.
  * 
  * **Note** that `num_outputs` should evenly divide the length of the axis
  * along which to split the array.
  * 
  * Example::
  * 
  *    x  = [[[ 1.]
  *           [ 2.]]
  *          [[ 3.]
  *           [ 4.]]
  *          [[ 5.]
  *           [ 6.]]]
  *    x.shape = (3, 2, 1)
  * 
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)
  *    y = [[[ 1.]]
  *         [[ 3.]]
  *         [[ 5.]]]
  * 
  *        [[[ 2.]]
  *         [[ 4.]]
  *         [[ 6.]]]
  * 
  *    y[0].shape = (3, 1, 1)
  * 
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)
  *    z = [[[ 1.]
  *          [ 2.]]]
  * 
  *        [[[ 3.]
  *          [ 4.]]]
  * 
  *        [[[ 5.]
  *          [ 6.]]]
  * 
  *    z[0].shape = (1, 2, 1)
  * 
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only
  * along the `axis` which it is split.
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.
  * 
  * Example::
  * 
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)
  *    z = [[ 1.]
  *         [ 2.]]
  * 
  *        [[ 3.]
  *         [ 4.]]
  * 
  *        [[ 5.]
  *         [ 6.]]
  *    z[0].shape = (2 ,1 )
  * 
  * 
  * 
  * Defined in src/operator/slice_channel.cc:L107
  * @return org.apache.mxnet.NDArray
  */
def split(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise square-root value of the input.
  * 
  * .. math::
  *    \textrm{sqrt}(x) = \sqrt{x}
  * 
  * Example::
  * 
  *    sqrt([4, 9, 16]) = [2, 3, 4]
  * 
  * The storage type of ``sqrt`` output depends upon the input storage type:
  * 
  *    - sqrt(default) = default
  *    - sqrt(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L639
  * @return org.apache.mxnet.NDArray
  */
def sqrt(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise square-root value of the input.
  * 
  * .. math::
  *    \textrm{sqrt}(x) = \sqrt{x}
  * 
  * Example::
  * 
  *    sqrt([4, 9, 16]) = [2, 3, 4]
  * 
  * The storage type of ``sqrt`` output depends upon the input storage type:
  * 
  *    - sqrt(default) = default
  *    - sqrt(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L639
  * @return org.apache.mxnet.NDArray
  */
def sqrt(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise squared value of the input.
  * 
  * .. math::
  *    square(x) = x^2
  * 
  * Example::
  * 
  *    square([2, 3, 4]) = [4, 9, 16]
  * 
  * The storage type of ``square`` output depends upon the input storage type:
  * 
  *    - square(default) = default
  *    - square(row_sparse) = row_sparse
  *    - square(csr) = csr
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L616
  * @return org.apache.mxnet.NDArray
  */
def square(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns element-wise squared value of the input.
  * 
  * .. math::
  *    square(x) = x^2
  * 
  * Example::
  * 
  *    square([2, 3, 4]) = [4, 9, 16]
  * 
  * The storage type of ``square`` output depends upon the input storage type:
  * 
  *    - square(default) = default
  *    - square(row_sparse) = row_sparse
  *    - square(csr) = csr
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L616
  * @return org.apache.mxnet.NDArray
  */
def square(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Remove single-dimensional entries from the shape of an array.
  * Same behavior of defining the output tensor shape as numpy.squeeze for the most of cases.
  * See the following note for exception.
  * 
  * Examples::
  * 
  *   data = [[[0], [1], [2]]]
  *   squeeze(data) = [0, 1, 2]
  *   squeeze(data, axis=0) = [[0], [1], [2]]
  *   squeeze(data, axis=2) = [[0, 1, 2]]
  *   squeeze(data, axis=(0, 2)) = [0, 1, 2]
  * 
  * .. Note::
  *   The output of this operator will keep at least one dimension not removed. For example,
  *   squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.
  * @return org.apache.mxnet.NDArray
  */
def squeeze(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Remove single-dimensional entries from the shape of an array.
  * Same behavior of defining the output tensor shape as numpy.squeeze for the most of cases.
  * See the following note for exception.
  * 
  * Examples::
  * 
  *   data = [[[0], [1], [2]]]
  *   squeeze(data) = [0, 1, 2]
  *   squeeze(data, axis=0) = [[0], [1], [2]]
  *   squeeze(data, axis=2) = [[0, 1, 2]]
  *   squeeze(data, axis=(0, 2)) = [0, 1, 2]
  * 
  * .. Note::
  *   The output of this operator will keep at least one dimension not removed. For example,
  *   squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.
  * @return org.apache.mxnet.NDArray
  */
def squeeze(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Join a sequence of arrays along a new axis.
  * 
  * The axis parameter specifies the index of the new axis in the dimensions of the
  * result. For example, if axis=0 it will be the first dimension and if axis=-1 it
  * will be the last dimension.
  * 
  * Examples::
  * 
  *   x = [1, 2]
  *   y = [3, 4]
  * 
  *   stack(x, y) = [[1, 2],
  *                  [3, 4]]
  *   stack(x, y, axis=1) = [[1, 3],
  *                          [2, 4]]
  * @return org.apache.mxnet.NDArray
  */
def stack(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Join a sequence of arrays along a new axis.
  * 
  * The axis parameter specifies the index of the new axis in the dimensions of the
  * result. For example, if axis=0 it will be the first dimension and if axis=-1 it
  * will be the last dimension.
  * 
  * Examples::
  * 
  *   x = [1, 2]
  *   y = [3, 4]
  * 
  *   stack(x, y) = [[1, 2],
  *                  [3, 4]]
  *   stack(x, y, axis=1) = [[1, 3],
  *                          [2, 4]]
  * @return org.apache.mxnet.NDArray
  */
def stack(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Stops gradient computation.
  * 
  * Stops the accumulated gradient of the inputs from flowing through this operator
  * in the backward direction. In other words, this operator prevents the contribution
  * of its inputs to be taken into account for computing gradients.
  * 
  * Example::
  * 
  *   v1 = [1, 2]
  *   v2 = [0, 1]
  *   a = Variable('a')
  *   b = Variable('b')
  *   b_stop_grad = stop_gradient(3 * b)
  *   loss = MakeLoss(b_stop_grad + a)
  * 
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  *   executor.forward(is_train=True, a=v1, b=v2)
  *   executor.outputs
  *   [ 1.  5.]
  * 
  *   executor.backward()
  *   executor.grad_arrays
  *   [ 0.  0.]
  *   [ 1.  1.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L241
  * @return org.apache.mxnet.NDArray
  */
def stop_gradient(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Stops gradient computation.
  * 
  * Stops the accumulated gradient of the inputs from flowing through this operator
  * in the backward direction. In other words, this operator prevents the contribution
  * of its inputs to be taken into account for computing gradients.
  * 
  * Example::
  * 
  *   v1 = [1, 2]
  *   v2 = [0, 1]
  *   a = Variable('a')
  *   b = Variable('b')
  *   b_stop_grad = stop_gradient(3 * b)
  *   loss = MakeLoss(b_stop_grad + a)
  * 
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))
  *   executor.forward(is_train=True, a=v1, b=v2)
  *   executor.outputs
  *   [ 1.  5.]
  * 
  *   executor.backward()
  *   executor.grad_arrays
  *   [ 0.  0.]
  *   [ 1.  1.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L241
  * @return org.apache.mxnet.NDArray
  */
def stop_gradient(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes.
  * 
  * .. Note::
  * 
  *   `sum` and `sum_axis` are equivalent.
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.
  * 
  * Example::
  * 
  *   data = [[[1,2],[2,3],[1,3]],
  *           [[1,4],[4,3],[5,2]],
  *           [[7,1],[7,2],[7,3]]]
  * 
  *   sum(data, axis=1)
  *   [[  4.   8.]
  *    [ 10.   9.]
  *    [ 21.   6.]]
  * 
  *   sum(data, axis=[1,2])
  *   [ 12.  19.  27.]
  * 
  *   data = [[1,2,0],
  *           [3,0,1],
  *           [4,1,0]]
  * 
  *   csr = cast_storage(data, 'csr')
  * 
  *   sum(csr, axis=0)
  *   [ 8.  3.  1.]
  * 
  *   sum(csr, axis=1)
  *   [ 3.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def sum(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes.
  * 
  * .. Note::
  * 
  *   `sum` and `sum_axis` are equivalent.
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.
  * 
  * Example::
  * 
  *   data = [[[1,2],[2,3],[1,3]],
  *           [[1,4],[4,3],[5,2]],
  *           [[7,1],[7,2],[7,3]]]
  * 
  *   sum(data, axis=1)
  *   [[  4.   8.]
  *    [ 10.   9.]
  *    [ 21.   6.]]
  * 
  *   sum(data, axis=[1,2])
  *   [ 12.  19.  27.]
  * 
  *   data = [[1,2,0],
  *           [3,0,1],
  *           [4,1,0]]
  * 
  *   csr = cast_storage(data, 'csr')
  * 
  *   sum(csr, axis=0)
  *   [ 8.  3.  1.]
  * 
  *   sum(csr, axis=1)
  *   [ 3.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def sum(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes.
  * 
  * .. Note::
  * 
  *   `sum` and `sum_axis` are equivalent.
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.
  * 
  * Example::
  * 
  *   data = [[[1,2],[2,3],[1,3]],
  *           [[1,4],[4,3],[5,2]],
  *           [[7,1],[7,2],[7,3]]]
  * 
  *   sum(data, axis=1)
  *   [[  4.   8.]
  *    [ 10.   9.]
  *    [ 21.   6.]]
  * 
  *   sum(data, axis=[1,2])
  *   [ 12.  19.  27.]
  * 
  *   data = [[1,2,0],
  *           [3,0,1],
  *           [4,1,0]]
  * 
  *   csr = cast_storage(data, 'csr')
  * 
  *   sum(csr, axis=0)
  *   [ 8.  3.  1.]
  * 
  *   sum(csr, axis=1)
  *   [ 3.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def sum_axis(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the sum of array elements over given axes.
  * 
  * .. Note::
  * 
  *   `sum` and `sum_axis` are equivalent.
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.
  * 
  * Example::
  * 
  *   data = [[[1,2],[2,3],[1,3]],
  *           [[1,4],[4,3],[5,2]],
  *           [[7,1],[7,2],[7,3]]]
  * 
  *   sum(data, axis=1)
  *   [[  4.   8.]
  *    [ 10.   9.]
  *    [ 21.   6.]]
  * 
  *   sum(data, axis=[1,2])
  *   [ 12.  19.  27.]
  * 
  *   data = [[1,2,0],
  *           [3,0,1],
  *           [4,1,0]]
  * 
  *   csr = cast_storage(data, 'csr')
  * 
  *   sum(csr, axis=0)
  *   [ 8.  3.  1.]
  * 
  *   sum(csr, axis=1)
  *   [ 3.  4.  5.]
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L86
  * @return org.apache.mxnet.NDArray
  */
def sum_axis(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Interchanges two axes of an array.
  * 
  * Examples::
  * 
  *   x = [[1, 2, 3]])
  *   swapaxes(x, 0, 1) = [[ 1],
  *                        [ 2],
  *                        [ 3]]
  * 
  *   x = [[[ 0, 1],
  *         [ 2, 3]],
  *        [[ 4, 5],
  *         [ 6, 7]]]  // (2,2,2) array
  * 
  *  swapaxes(x, 0, 2) = [[[ 0, 4],
  *                        [ 2, 6]],
  *                       [[ 1, 5],
  *                        [ 3, 7]]]
  * 
  * 
  * Defined in src/operator/swapaxis.cc:L70
  * @return org.apache.mxnet.NDArray
  */
def swapaxes(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Interchanges two axes of an array.
  * 
  * Examples::
  * 
  *   x = [[1, 2, 3]])
  *   swapaxes(x, 0, 1) = [[ 1],
  *                        [ 2],
  *                        [ 3]]
  * 
  *   x = [[[ 0, 1],
  *         [ 2, 3]],
  *        [[ 4, 5],
  *         [ 6, 7]]]  // (2,2,2) array
  * 
  *  swapaxes(x, 0, 2) = [[[ 0, 4],
  *                        [ 2, 6]],
  *                       [[ 1, 5],
  *                        [ 3, 7]]]
  * 
  * 
  * Defined in src/operator/swapaxis.cc:L70
  * @return org.apache.mxnet.NDArray
  */
def swapaxes(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes elements from an input array along the given axis.
  * 
  * This function slices the input array along a particular axis with the provided indices.
  * 
  * Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the output
  * will have shape ``(i0, i1, d1, d2)``, computed by::
  * 
  *   output[i,j,:,:] = input[indices[i,j],:,:]
  * 
  * .. note::
  *    - `axis`- Only slicing along axis 0 is supported for now.
  *    - `mode`- Only `clip` mode is supported for now.
  * 
  * Examples::
  *   x = [4.  5.  6.]
  * 
  *   // Trivial case, take the second element along the first axis.
  *   take(x, [1]) = [ 5. ]
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0
  *   take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
  *                              [ 3.,  4.]],
  * 
  *                             [[ 3.,  4.],
  *                              [ 5.,  6.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L379
  * @return org.apache.mxnet.NDArray
  */
def take(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Takes elements from an input array along the given axis.
  * 
  * This function slices the input array along a particular axis with the provided indices.
  * 
  * Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the output
  * will have shape ``(i0, i1, d1, d2)``, computed by::
  * 
  *   output[i,j,:,:] = input[indices[i,j],:,:]
  * 
  * .. note::
  *    - `axis`- Only slicing along axis 0 is supported for now.
  *    - `mode`- Only `clip` mode is supported for now.
  * 
  * Examples::
  *   x = [4.  5.  6.]
  * 
  *   // Trivial case, take the second element along the first axis.
  *   take(x, [1]) = [ 5. ]
  * 
  *   x = [[ 1.,  2.],
  *        [ 3.,  4.],
  *        [ 5.,  6.]]
  * 
  *   // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0
  *   take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],
  *                              [ 3.,  4.]],
  * 
  *                             [[ 3.,  4.],
  *                              [ 5.,  6.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/indexing_op.cc:L379
  * @return org.apache.mxnet.NDArray
  */
def take(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise tangent of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
  * 
  * The storage type of ``tan`` output depends upon the input storage type:
  * 
  *    - tan(default) = default
  *    - tan(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L83
  * @return org.apache.mxnet.NDArray
  */
def tan(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Computes the element-wise tangent of the input array.
  * 
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).
  * 
  * .. math::
  *    tan([0, \pi/4, \pi/2]) = [0, 1, -inf]
  * 
  * The storage type of ``tan`` output depends upon the input storage type:
  * 
  *    - tan(default) = default
  *    - tan(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L83
  * @return org.apache.mxnet.NDArray
  */
def tan(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic tangent of the input array, computed element-wise.
  * 
  * .. math::
  *    tanh(x) = sinh(x) / cosh(x)
  * 
  * The storage type of ``tanh`` output depends upon the input storage type:
  * 
  *    - tanh(default) = default
  *    - tanh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L234
  * @return org.apache.mxnet.NDArray
  */
def tanh(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the hyperbolic tangent of the input array, computed element-wise.
  * 
  * .. math::
  *    tanh(x) = sinh(x) / cosh(x)
  * 
  * The storage type of ``tanh`` output depends upon the input storage type:
  * 
  *    - tanh(default) = default
  *    - tanh(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L234
  * @return org.apache.mxnet.NDArray
  */
def tanh(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Repeats the whole array multiple times.
  * 
  * If ``reps`` has length *d*, and input array has dimension of *n*. There are
  * three cases:
  * 
  * - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
  * 
  *     x = [[1, 2],
  *          [3, 4]]
  * 
  *     tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                            [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.]]
  * 
  * - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
  *   an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
  * 
  * 
  *     tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
  *                           [ 3.,  4.,  3.,  4.]]
  * 
  * - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a
  *   shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
  * 
  *     tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]],
  * 
  *                              [[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L676
  * @return org.apache.mxnet.NDArray
  */
def tile(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Repeats the whole array multiple times.
  * 
  * If ``reps`` has length *d*, and input array has dimension of *n*. There are
  * three cases:
  * 
  * - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::
  * 
  *     x = [[1, 2],
  *          [3, 4]]
  * 
  *     tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                            [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.]]
  * 
  * - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for
  *   an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::
  * 
  * 
  *     tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],
  *                           [ 3.,  4.,  3.,  4.]]
  * 
  * - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a
  *   shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::
  * 
  *     tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]],
  * 
  *                              [[ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L676
  * @return org.apache.mxnet.NDArray
  */
def tile(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the top *k* elements in an input array along the given axis.
  * 
  * Examples::
  * 
  *   x = [[ 0.3,  0.2,  0.4],
  *        [ 0.1,  0.3,  0.2]]
  * 
  *   // returns an index of the largest element on last axis
  *   topk(x) = [[ 2.],
  *              [ 1.]]
  * 
  *   // returns the value of top-2 largest elements on last axis
  *   topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
  *                                    [ 0.3,  0.2]]
  * 
  *   // returns the value of top-2 smallest elements on last axis
  *   topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
  *                                                [ 0.1 ,  0.2]]
  * 
  *   // returns the value of top-2 largest elements on axis 0
  *   topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
  *                                            [ 0.1,  0.2,  0.2]]
  * 
  *   // flattens and then returns list of both values and indices
  *   topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def topk(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Returns the top *k* elements in an input array along the given axis.
  * 
  * Examples::
  * 
  *   x = [[ 0.3,  0.2,  0.4],
  *        [ 0.1,  0.3,  0.2]]
  * 
  *   // returns an index of the largest element on last axis
  *   topk(x) = [[ 2.],
  *              [ 1.]]
  * 
  *   // returns the value of top-2 largest elements on last axis
  *   topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],
  *                                    [ 0.3,  0.2]]
  * 
  *   // returns the value of top-2 smallest elements on last axis
  *   topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],
  *                                                [ 0.1 ,  0.2]]
  * 
  *   // returns the value of top-2 largest elements on axis 0
  *   topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],
  *                                            [ 0.1,  0.2,  0.2]]
  * 
  *   // flattens and then returns list of both values and indices
  *   topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/ordering_op.cc:L63
  * @return org.apache.mxnet.NDArray
  */
def topk(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Permutes the dimensions of an array.
  * 
  * Examples::
  * 
  *   x = [[ 1, 2],
  *        [ 3, 4]]
  * 
  *   transpose(x) = [[ 1.,  3.],
  *                   [ 2.,  4.]]
  * 
  *   x = [[[ 1.,  2.],
  *         [ 3.,  4.]],
  * 
  *        [[ 5.,  6.],
  *         [ 7.,  8.]]]
  * 
  *   transpose(x) = [[[ 1.,  5.],
  *                    [ 3.,  7.]],
  * 
  *                   [[ 2.,  6.],
  *                    [ 4.,  8.]]]
  * 
  *   transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
  *                                  [ 5.,  6.]],
  * 
  *                                 [[ 3.,  4.],
  *                                  [ 7.,  8.]]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L309
  * @return org.apache.mxnet.NDArray
  */
def transpose(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Permutes the dimensions of an array.
  * 
  * Examples::
  * 
  *   x = [[ 1, 2],
  *        [ 3, 4]]
  * 
  *   transpose(x) = [[ 1.,  3.],
  *                   [ 2.,  4.]]
  * 
  *   x = [[[ 1.,  2.],
  *         [ 3.,  4.]],
  * 
  *        [[ 5.,  6.],
  *         [ 7.,  8.]]]
  * 
  *   transpose(x) = [[[ 1.,  5.],
  *                    [ 3.,  7.]],
  * 
  *                   [[ 2.,  6.],
  *                    [ 4.,  8.]]]
  * 
  *   transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],
  *                                  [ 5.,  6.]],
  * 
  *                                 [[ 3.,  4.],
  *                                  [ 7.,  8.]]]
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L309
  * @return org.apache.mxnet.NDArray
  */
def transpose(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return the element-wise truncated value of the input.
  * 
  * The truncated value of the scalar x is the nearest integer i which is closer to
  * zero than x is. In short, the fractional part of the signed number x is discarded.
  * 
  * Example::
  * 
  *    trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
  * 
  * The storage type of ``trunc`` output depends upon the input storage type:
  * 
  *    - trunc(default) = default
  *    - trunc(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L577
  * @return org.apache.mxnet.NDArray
  */
def trunc(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return the element-wise truncated value of the input.
  * 
  * The truncated value of the scalar x is the nearest integer i which is closer to
  * zero than x is. In short, the fractional part of the signed number x is discarded.
  * 
  * Example::
  * 
  *    trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]
  * 
  * The storage type of ``trunc`` output depends upon the input storage type:
  * 
  *    - trunc(default) = default
  *    - trunc(row_sparse) = row_sparse
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L577
  * @return org.apache.mxnet.NDArray
  */
def trunc(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a uniform distribution.
  * 
  * .. note:: The existing alias ``uniform`` is deprecated.
  * 
  * Samples are uniformly distributed over the half-open interval *[low, high)*
  * (includes *low*, but excludes *high*).
  * 
  * Example::
  * 
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
  *                                           [ 0.54488319,  0.84725171]]
  * 
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L66
  * @return org.apache.mxnet.NDArray
  */
def uniform(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Draw random samples from a uniform distribution.
  * 
  * .. note:: The existing alias ``uniform`` is deprecated.
  * 
  * Samples are uniformly distributed over the half-open interval *[low, high)*
  * (includes *low*, but excludes *high*).
  * 
  * Example::
  * 
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],
  *                                           [ 0.54488319,  0.84725171]]
  * 
  * 
  * 
  * Defined in src/operator/random/sample_op.cc:L66
  * @return org.apache.mxnet.NDArray
  */
def uniform(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return the elements, either from x or y, depending on the condition.
  * 
  * Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
  * depending on the elements from condition are true or false. x and y must have the same shape.
  * If condition has the same shape as x, each element in the output array is from x if the
  * corresponding element in the condition is true, and from y if false.
  * 
  * If condition does not have the same shape as x, it must be a 1D array whose size is
  * the same as x's first dimension size. Each row of the output array is from x's row
  * if the corresponding element from condition is true, and from y's row if false.
  * 
  * Note that all non-zero values are interpreted as ``True`` in condition.
  * 
  * Examples::
  * 
  *   x = [[1, 2], [3, 4]]
  *   y = [[5, 6], [7, 8]]
  *   cond = [[0, 1], [-1, 0]]
  * 
  *   where(cond, x, y) = [[5, 2], [3, 8]]
  * 
  *   csr_cond = cast_storage(cond, 'csr')
  * 
  *   where(csr_cond, x, y) = [[5, 2], [3, 8]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/control_flow_op.cc:L57
  * @return org.apache.mxnet.NDArray
  */
def where(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return the elements, either from x or y, depending on the condition.
  * 
  * Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,
  * depending on the elements from condition are true or false. x and y must have the same shape.
  * If condition has the same shape as x, each element in the output array is from x if the
  * corresponding element in the condition is true, and from y if false.
  * 
  * If condition does not have the same shape as x, it must be a 1D array whose size is
  * the same as x's first dimension size. Each row of the output array is from x's row
  * if the corresponding element from condition is true, and from y's row if false.
  * 
  * Note that all non-zero values are interpreted as ``True`` in condition.
  * 
  * Examples::
  * 
  *   x = [[1, 2], [3, 4]]
  *   y = [[5, 6], [7, 8]]
  *   cond = [[0, 1], [-1, 0]]
  * 
  *   where(cond, x, y) = [[5, 2], [3, 8]]
  * 
  *   csr_cond = cast_storage(cond, 'csr')
  * 
  *   where(csr_cond, x, y) = [[5, 2], [3, 8]]
  * 
  * 
  * 
  * Defined in src/operator/tensor/control_flow_op.cc:L57
  * @return org.apache.mxnet.NDArray
  */
def where(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return an array of zeros with the same shape and type
  * as the input array.
  * 
  * The storage type of ``zeros_like`` output depends on the storage type of the input
  * 
  * - zeros_like(row_sparse) = row_sparse
  * - zeros_like(csr) = csr
  * - zeros_like(default) = default
  * 
  * Examples::
  * 
  *   x = [[ 1.,  1.,  1.],
  *        [ 1.,  1.,  1.]]
  * 
  *   zeros_like(x) = [[ 0.,  0.,  0.],
  *                    [ 0.,  0.,  0.]]
  * @return org.apache.mxnet.NDArray
  */
def zeros_like(kwargs: Map[String, Any] = null)(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
  /**
  * Return an array of zeros with the same shape and type
  * as the input array.
  * 
  * The storage type of ``zeros_like`` output depends on the storage type of the input
  * 
  * - zeros_like(row_sparse) = row_sparse
  * - zeros_like(csr) = csr
  * - zeros_like(default) = default
  * 
  * Examples::
  * 
  *   x = [[ 1.,  1.,  1.],
  *        [ 1.,  1.,  1.]]
  * 
  *   zeros_like(x) = [[ 0.,  0.,  0.],
  *                    [ 0.,  0.,  0.]]
  * @return org.apache.mxnet.NDArray
  */
def zeros_like(args: Any*) : org.apache.mxnet.NDArrayFuncReturn
}
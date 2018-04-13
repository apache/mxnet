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

package org.apache.mxnet

trait SymbolBase {

  // scalastyle:off
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
  * @param data		The input array.
  * @param act_type		Activation function to be applied.
  * @return null
  */
  def Activation(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to batch normalization
  * @param gamma		gamma array
  * @param beta		beta array
  * @param moving_mean		running mean of input
  * @param moving_var		running variance of input
  * @param eps		Epsilon to prevent div 0. Must be no less than CUDNN_BN_MIN_EPSILON defined in cudnn.h when using cudnn (usually 1e-5)
  * @param momentum		Momentum for moving average
  * @param fix_gamma		Fix gamma while training
  * @param use_global_stats		Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
  * @param output_mean_var		Output the mean and inverse std 
  * @param axis		Specify which shape axis the channel is specified
  * @param cudnn_off		Do not select CUDNN operator, if available
  * @return null
  */
  def BatchNorm(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to batch normalization
  * @param gamma		gamma array
  * @param beta		beta array
  * @param eps		Epsilon to prevent div 0
  * @param momentum		Momentum for moving average
  * @param fix_gamma		Fix gamma while training
  * @param use_global_stats		Whether use global moving statistics instead of local batch-norm. This will force change batch-norm into a scale shift operator.
  * @param output_mean_var		Output All,normal mean and var
  * @return null
  */
  def BatchNorm_v1(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the BilinearsamplerOp.
  * @param grid		Input grid to the BilinearsamplerOp.grid has two channels: x_src, y_src
  * @return null
  */
  def BilinearSampler(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def BlockGrad(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input.
  * @param dtype		Output data type.
  * @return null
  */
  def Cast(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		List of arrays to concatenate
  * @param num_args		Number of inputs to be concated.
  * @param dim		the dimension to be concated.
  * @return null
  */
  def Concat(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the ConvolutionOp.
  * @param weight		Weight matrix.
  * @param bias		Bias parameter.
  * @param kernel		Convolution kernel size: (w,), (h, w) or (d, h, w)
  * @param stride		Convolution stride: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
  * @param dilate		Convolution dilate: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
  * @param pad		Zero pad for convolution: (w,), (h, w) or (d, h, w). Defaults to no padding.
  * @param num_filter		Convolution filter(channel) number
  * @param num_group		Number of group partitions.
  * @param workspace		Maximum temporary workspace allowed (MB) in convolution.This parameter has two usages. When CUDNN is not used, it determines the effective batch size of the convolution kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is used.
  * @param no_bias		Whether to disable bias parameter.
  * @param cudnn_tune		Whether to pick convolution algo by running performance test.
  * @param cudnn_off		Turn off cudnn for this layer.
  * @param layout		Set layout for input, output and weight. Empty for
    default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.
  * @return null
  */
  def Convolution(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * This operator is DEPRECATED. Apply convolution to input then add a bias.
  * @param data		Input data to the ConvolutionV1Op.
  * @param weight		Weight matrix.
  * @param bias		Bias parameter.
  * @param kernel		convolution kernel size: (h, w) or (d, h, w)
  * @param stride		convolution stride: (h, w) or (d, h, w)
  * @param dilate		convolution dilate: (h, w) or (d, h, w)
  * @param pad		pad for convolution: (h, w) or (d, h, w)
  * @param num_filter		convolution filter(channel) number
  * @param num_group		Number of group partitions. Equivalent to slicing input into num_group
    partitions, apply convolution on each, then concatenate the results
  * @param workspace		Maximum temporary workspace allowed for convolution (MB).This parameter determines the effective batch size of the convolution kernel, which may be smaller than the given batch size. Also, the workspace will be automatically enlarged to make sure that we can run the kernel with batch_size=1
  * @param no_bias		Whether to disable bias parameter.
  * @param cudnn_tune		Whether to pick convolution algo by running performance test.
    Leads to higher startup time but may give faster speed. Options are:
    'off': no tuning
    'limited_workspace': run test and pick the fastest algorithm that doesn't exceed workspace limit.
    'fastest': pick the fastest algorithm and ignore workspace limit.
    If set to None (default), behavior is determined by environment
    variable MXNET_CUDNN_AUTOTUNE_DEFAULT: 0 for off,
    1 for limited workspace (default), 2 for fastest.
  * @param cudnn_off		Turn off cudnn for this layer.
  * @param layout		Set layout for input, output and weight. Empty for
    default layout: NCHW for 2d and NCDHW for 3d.
  * @return null
  */
  def Convolution_v1(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data1		Input data1 to the correlation.
  * @param data2		Input data2 to the correlation.
  * @param kernel_size		kernel size for Correlation must be an odd number
  * @param max_displacement		Max displacement of Correlation 
  * @param stride1		stride1 quantize data1 globally
  * @param stride2		stride2 quantize data2 within the neighborhood centered around data1
  * @param pad_size		pad for Correlation
  * @param is_multiply		operation type is either multiplication or subduction
  * @return null
  */
  def Correlation(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Tensor or List of Tensors, the second input will be used as crop_like shape reference
  * @param num_args		Number of inputs for crop, if equals one, then we will use the h_wfor crop height and width, else if equals two, then we will use the heightand width of the second input symbol, we name crop_like here
  * @param offset		crop offset coordinate: (y, x)
  * @param h_w		crop height and width: (h, w)
  * @param center_crop		If set to true, then it will use be the center_crop,or it will crop using the shape of crop_like
  * @return null
  */
  def Crop(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data for the custom operator.
  * @param op_type		Name of the custom operator. This is the name that is passed to `mx.operator.register` to register the operator.
  * @return null
  */
  def Custom(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.
  * @param data		Input tensor to the deconvolution operation.
  * @param weight		Weights representing the kernel.
  * @param bias		Bias added to the result after the deconvolution operation.
  * @param kernel		Deconvolution kernel size: (w,), (h, w) or (d, h, w). This is same as the kernel size used for the corresponding convolution
  * @param stride		The stride used for the corresponding convolution: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
  * @param dilate		Dilation factor for each dimension of the input: (w,), (h, w) or (d, h, w). Defaults to 1 for each dimension.
  * @param pad		The amount of implicit zero padding added during convolution for each dimension of the input: (w,), (h, w) or (d, h, w). ``(kernel-1)/2`` is usually a good choice. If `target_shape` is set, `pad` will be ignored and a padding that will generate the target shape will be used. Defaults to no padding.
  * @param adj		Adjustment for output shape: (w,), (h, w) or (d, h, w). If `target_shape` is set, `adj` will be ignored and computed accordingly.
  * @param target_shape		Shape of the output tensor: (w,), (h, w) or (d, h, w).
  * @param num_filter		Number of output filters.
  * @param num_group		Number of groups partition.
  * @param workspace		Maximum temporary workspace allowed (MB) in deconvolution.This parameter has two usages. When CUDNN is not used, it determines the effective batch size of the deconvolution kernel. When CUDNN is used, it controls the maximum temporary storage used for tuning the best CUDNN kernel when `limited_workspace` strategy is used.
  * @param no_bias		Whether to disable bias parameter.
  * @param cudnn_tune		Whether to pick convolution algorithm by running performance test.
  * @param cudnn_off		Turn off cudnn for this layer.
  * @param layout		Set layout for input, output and weight. Empty for default layout, NCW for 1d, NCHW for 2d and NCDHW for 3d.
  * @return null
  */
  def Deconvolution(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array to which dropout will be applied.
  * @param p		Fraction of the input that gets dropped out during training time.
  * @param mode		Whether to only turn on dropout during training or to also turn on for inference.
  * @param axes		Axes for variational dropout kernel.
  * @return null
  */
  def Dropout(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param args		Positional input arguments
  * @return null
  */
  def add_n(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array to the embedding operator.
  * @param weight		The embedding weight matrix.
  * @param input_dim		Vocabulary size of the input indices.
  * @param output_dim		Dimension of the embedding vectors.
  * @param dtype		Data type of weight.
  * @return null
  */
  def Embedding(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @return null
  */
  def Flatten(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data.
  * @param weight		Weight matrix.
  * @param bias		Bias parameter.
  * @param num_hidden		Number of hidden nodes of the output.
  * @param no_bias		Whether to disable bias parameter.
  * @param flatten		Whether to collapse all but the first axis of the input data tensor.
  * @return null
  */
  def FullyConnected(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Generates 2D sampling grid for bilinear sampling.
  * @param data		Input data to the function.
  * @param transform_type		The type of transformation. For `affine`, input data should be an affine matrix of size (batch, 6). For `warp`, input data should be an optical flow of size (batch, 2, h, w).
  * @param target_shape		Specifies the output shape (H, W). This is required if transformation type is `affine`. If transformation type is `warp`, this parameter is ignored.
  * @return null
  */
  def GridGenerator(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Apply a sparse regularization to the output a sigmoid activation function.
  * @param data		Input data.
  * @param sparseness_target		The sparseness target
  * @param penalty		The tradeoff parameter for the sparseness penalty
  * @param momentum		The momentum for running average
  * @return null
  */
  def IdentityAttachKLSparseReg(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		An n-dimensional input array (n > 2) of the form [batch, channel, spatial_dim1, spatial_dim2, ...].
  * @param gamma		A vector of length 'channel', which multiplies the normalized input.
  * @param beta		A vector of length 'channel', which is added to the product of the normalized input and the weight.
  * @param eps		An `epsilon` parameter to prevent division by 0.
  * @return null
  */
  def InstanceNorm(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array to normalize.
  * @param eps		A small constant for numerical stability.
  * @param mode		Specify the dimension along which to compute L2 norm.
  * @return null
  */
  def L2Normalization(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to LRN
  * @param alpha		The variance scaling parameter :math:`lpha` in the LRN expression.
  * @param beta		The power parameter :math:`eta` in the LRN expression.
  * @param knorm		The parameter :math:`k` in the LRN expression.
  * @param nsize		normalization window width in elements.
  * @return null
  */
  def LRN(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to layer normalization
  * @param gamma		gamma array
  * @param beta		beta array
  * @param axis		The axis to perform layer normalization. Usually, this should be be axis of the channel dimension. Negative values means indexing from right to left.
  * @param eps		An `epsilon` parameter to prevent division by 0.
  * @param output_mean_var		Output the mean and std calculated along the given axis.
  * @return null
  */
  def LayerNorm(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to activation function.
  * @param gamma		Slope parameter for PReLU. Only required when act_type is 'prelu'. It should be either a vector of size 1, or the same size as the second dimension of data.
  * @param act_type		Activation function to be applied.
  * @param slope		Init slope for the activation. (For leaky and elu only)
  * @param lower_bound		Lower bound of random slope. (For rrelu only)
  * @param upper_bound		Upper bound of random slope. (For rrelu only)
  * @return null
  */
  def LeakyReLU(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the function.
  * @param label		Input label to the function.
  * @param grad_scale		Scale the gradient by a float factor
  * @return null
  */
  def LinearRegressionOutput(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the function.
  * @param label		Input label to the function.
  * @param grad_scale		Scale the gradient by a float factor
  * @return null
  */
  def LogisticRegressionOutput(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the function.
  * @param label		Input label to the function.
  * @param grad_scale		Scale the gradient by a float factor
  * @return null
  */
  def MAERegressionOutput(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @param grad_scale		Gradient scale as a supplement to unary and binary operators
  * @param valid_thresh		clip each element in the array to 0 when it is less than ``valid_thresh``. This is used when ``normalization`` is set to ``'valid'``.
  * @param normalization		If this is set to null, the output gradient will not be normalized. If this is set to batch, the output gradient will be divided by the batch size. If this is set to valid, the output gradient will be divided by the number of valid input elements.
  * @return null
  */
  def MakeLoss(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		An n-dimensional input array.
  * @param mode		Padding type to use. "constant" pads with `constant_value` "edge" pads using the edge values of the input array "reflect" pads by reflecting values with respect to the edges.
  * @param pad_width		Widths of the padding regions applied to the edges of each axis. It is a tuple of integer padding widths for each axis of the format ``(before_1, after_1, ... , before_N, after_N)``. It should be of length ``2*N`` where ``N`` is the number of dimensions of the array.This is equivalent to pad_width in numpy.pad, but flattened.
  * @param constant_value		The value used for padding when `mode` is "constant".
  * @return null
  */
  def Pad(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the pooling operator.
  * @param kernel		Pooling kernel size: (y, x) or (d, y, x)
  * @param pool_type		Pooling type to be applied.
  * @param global_pool		Ignore kernel size, do global pooling based on current input feature map. 
  * @param cudnn_off		Turn off cudnn pooling and use MXNet pooling operator. 
  * @param pooling_convention		Pooling convention to be applied.
  * @param stride		Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.
  * @param pad		Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.
  * @return null
  */
  def Pooling(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to the pooling operator.
  * @param kernel		pooling kernel size: (y, x) or (d, y, x)
  * @param pool_type		Pooling type to be applied.
  * @param global_pool		Ignore kernel size, do global pooling based on current input feature map. 
  * @param pooling_convention		Pooling convention to be applied.
  * @param stride		stride: for pooling (y, x) or (d, y, x)
  * @param pad		pad for pooling: (y, x) or (d, y, x)
  * @return null
  */
  def Pooling_v1(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Applies a recurrent layer to input.
  * @param data		Input data to RNN
  * @param parameters		Vector of all RNN trainable parameters concatenated
  * @param state		initial hidden state of the RNN
  * @param state_cell		initial cell state for LSTM networks (only for LSTM)
  * @param state_size		size of the state for each layer
  * @param num_layers		number of stacked layers
  * @param bidirectional		whether to use bidirectional recurrent layers
  * @param mode		the type of RNN to compute
  * @param p		Dropout probability, fraction of the input that gets dropped out at training time
  * @param state_outputs		Whether to have the states as symbol outputs.
  * @return null
  */
  def RNN(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array to the pooling operator,  a 4D Feature maps 
  * @param rois		Bounding box coordinates, a 2D array of [[batch_index, x1, y1, x2, y2]], where (x1, y1) and (x2, y2) are top left and bottom right corners of designated region of interest. `batch_index` indicates the index of corresponding image in the input array
  * @param pooled_size		ROI pooling output shape (h,w) 
  * @param spatial_scale		Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal of total stride in convolutional layers
  * @return null
  */
  def ROIPooling(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data to reshape.
  * @param shape		The target shape
  * @param reverse		If true then the special values are inferred from right to left
  * @param target_shape		(Deprecated! Use ``shape`` instead.) Target new shape. One and only one dim can be 0, in which case it will be inferred from the rest of dims
  * @param keep_highest		(Deprecated! Use ``shape`` instead.) Whether keep the highest dim unchanged.If set to true, then the first dim in target_shape is ignored,and always fixed as input
  * @return null
  */
  def Reshape(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes support vector machine based transformation of the input.
  * 
  * This tutorial demonstrates using SVM as output layer for classification instead of softmax:
  * https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.
  * @param data		Input data for SVM transformation.
  * @param label		Class label for the input data.
  * @param margin		The loss function penalizes outputs that lie outside this margin. Default margin is 1.
  * @param regularization_coefficient		Regularization parameter for the SVM. This balances the tradeoff between coefficient size and error.
  * @param use_linear		Whether to use L1-SVM objective. L2-SVM objective is used by default.
  * @return null
  */
  def SVMOutput(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2
  * @param sequence_length		vector of sequence lengths of the form [batch_size]
  * @param use_sequence_length		If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
  * @param axis		The sequence axis. Only values of 0 and 1 are currently supported.
  * @return null
  */
  def SequenceLast(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims] where n>2
  * @param sequence_length		vector of sequence lengths of the form [batch_size]
  * @param use_sequence_length		If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
  * @param value		The value to be used as a mask.
  * @param axis		The sequence axis. Only values of 0 and 1 are currently supported.
  * @return null
  */
  def SequenceMask(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		n-dimensional input array of the form [max_sequence_length, batch_size, other dims] where n>2 
  * @param sequence_length		vector of sequence lengths of the form [batch_size]
  * @param use_sequence_length		If set to true, this layer takes in an extra input parameter `sequence_length` to specify variable length sequence
  * @param axis		The sequence axis. Only 0 is currently supported.
  * @return null
  */
  def SequenceReverse(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param num_outputs		Number of splits. Note that this should evenly divide the length of the `axis`.
  * @param axis		Axis along which to split.
  * @param squeeze_axis		If true, Removes the axis with length 1 from the shapes of the output arrays. **Note** that setting `squeeze_axis` to ``true`` removes axis with length 1 only along the `axis` which it is split. Also `squeeze_axis` can be set to ``true`` only if ``input.shape[axis] == num_outputs``.
  * @return null
  */
  def SliceChannel(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @param grad_scale		Scales the gradient by a float factor.
  * @param ignore_label		The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).
  * @param multi_output		If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.
  * @param use_ignore		If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.
  * @param preserve_shape		If set to ``true``, the softmax function will be computed along the last axis (``-1``).
  * @param normalization		Normalizes the gradient.
  * @param out_grad		Multiplies gradient with output gradient element-wise.
  * @param smooth_alpha		Constant for computing a label smoothed version of cross-entropyfor the backwards pass.  This constant gets subtracted from theone-hot encoding of the gold label and distributed uniformly toall other labels.
  * @return null
  */
  def Softmax(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @param mode		Specifies how to compute the softmax. If set to ``instance``, it computes softmax for each instance. If set to ``channel``, It computes cross channel softmax for each position of each instance.
  * @return null
  */
  def SoftmaxActivation(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @param label		Ground truth label.
  * @param grad_scale		Scales the gradient by a float factor.
  * @param ignore_label		The instances whose `labels` == `ignore_label` will be ignored during backward, if `use_ignore` is set to ``true``).
  * @param multi_output		If set to ``true``, the softmax function will be computed along axis ``1``. This is applied when the shape of input array differs from the shape of label array.
  * @param use_ignore		If set to ``true``, the `ignore_label` value will not contribute to the backward gradient.
  * @param preserve_shape		If set to ``true``, the softmax function will be computed along the last axis (``-1``).
  * @param normalization		Normalizes the gradient.
  * @param out_grad		Multiplies gradient with output gradient element-wise.
  * @param smooth_alpha		Constant for computing a label smoothed version of cross-entropyfor the backwards pass.  This constant gets subtracted from theone-hot encoding of the gold label and distributed uniformly toall other labels.
  * @return null
  */
  def SoftmaxOutput(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Applies a spatial transformer to input feature map.
  * @param data		Input data to the SpatialTransformerOp.
  * @param loc		localisation net, the output dim should be 6 when transform_type is affine. You shold initialize the weight and bias with identity tranform.
  * @param target_shape		output shape(h, w) of spatial transformer: (y, x)
  * @param transform_type		transformation type
  * @param sampler_type		sampling type
  * @return null
  */
  def SpatialTransformer(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @param dim1		the first axis to be swapped.
  * @param dim2		the second axis to be swapped.
  * @return null
  */
  def SwapAxis(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Performs nearest neighbor/bilinear up sampling to inputs.
  * @param data		Array of tensors to upsample
  * @param scale		Up sampling scale
  * @param num_filter		Input filter. Only used by bilinear sample_type.
  * @param sample_type		upsampling method
  * @param multi_input_mode		How to handle multiple input. concat means concatenate upsampled images along the channel dimension. sum means add all images together, only available for nearest neighbor upsampling.
  * @param num_args		Number of inputs to be upsampled. For nearest neighbor upsampling, this can be 1-N; the size of output will be(scale*h_0,scale*w_0) and all other inputs will be upsampled to thesame size. For bilinear upsampling this must be 2; 1 input and 1 weight.
  * @param workspace		Tmp workspace for deconvolution (MB)
  * @return null
  */
  def UpSampling(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Divides arguments element-wise.
  * 
  * The storage type of ``elemwise_div`` output is always dense
  * @param lhs		first input
  * @param rhs		second input
  * @return null
  */
  def elemwise_div(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Subtracts arguments element-wise.
  * 
  * The storage type of ``elemwise_sub`` output depends on storage types of inputs
  * 
  *    - elemwise_sub(row_sparse, row_sparse) = row_sparse
  *    - elemwise_sub(csr, csr) = csr
  *    - otherwise, ``elemwise_sub`` generates output with default storage
  * @param lhs		first input
  * @param rhs		second input
  * @return null
  */
  def elemwise_sub(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		first input
  * @param rhs		second input
  * @return null
  */
  def elemwise_mul(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Adds arguments element-wise.
  * 
  * The storage type of ``elemwise_add`` output depends on storage types of inputs
  * 
  *    - elemwise_add(row_sparse, row_sparse) = row_sparse
  *    - elemwise_add(csr, csr) = csr
  *    - otherwise, ``elemwise_add`` generates output with default storage
  * @param lhs		first input
  * @param rhs		second input
  * @return null
  */
  def elemwise_add(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def abs(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param mean		Moving mean
  * @param var		Moving variance
  * @param lr		Learning rate
  * @param beta1		The decay rate for the 1st moment estimates.
  * @param beta2		The decay rate for the 2nd moment estimates.
  * @param epsilon		A small constant for numerical stability.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def adam_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def arccos(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Returns the element-wise inverse hyperbolic cosine of the input array, \
  * computed element-wise.
  * 
  * The storage type of ``arccosh`` output is always dense
  * 
  * 
  * 
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L264
  * @param data		The input array.
  * @return null
  */
  def arccosh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def arcsin(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def arcsinh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def arctan(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def arctanh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input.
  * @param stype		Output storage type.
  * @return null
  */
  def cast_storage(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def ceil(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input array.
  * @param a_min		Minimum value
  * @param a_max		Maximum value
  * @return null
  */
  def clip(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def cos(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def cosh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def degrees(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		The first input
  * @param rhs		The second input
  * @param transpose_a		If true then transpose the first input before dot.
  * @param transpose_b		If true then transpose the second input before dot.
  * @return null
  */
  def dot(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def exp(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def expm1(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def fix(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def floor(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param z		z
  * @param n		Square of grad
  * @param lr		Learning rate
  * @param lamda1		The L1 regularization coefficient.
  * @param beta		Per-Coordinate Learning Rate beta.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def ftrl_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Returns the gamma function (extension of the factorial function \
  * to the reals), computed element-wise on the input array.
  * 
  * The storage type of ``gamma`` output is always dense
  * @param data		The input array.
  * @return null
  */
  def gamma(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Returns element-wise log of the absolute value of the gamma function \
  * of the input.
  * 
  * The storage type of ``gammaln`` output is always dense
  * @param data		The input array.
  * @return null
  */
  def gammaln(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def log(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def log10(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def log1p(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def log2(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def make_loss(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the mean of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L102
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def mean(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Numerical negative of the argument, element-wise.
  * 
  * The storage type of ``negative`` output depends upon the input storage type:
  * 
  *    - negative(default) = default
  *    - negative(row_sparse) = row_sparse
  *    - negative(csr) = csr
  * @param data		The input array.
  * @return null
  */
  def negative(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param ord		Order of the norm. Currently ord=2 is supported.
  * @param axis		The axis or axes along which to perform the reduction.
      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.
      If `axis` is int, a reduction is performed on a particular axis.
      If `axis` is a 2-tuple, it specifies the axes that hold 2-D matrices,
      and the matrix norms of these matrices are computed.
  * @param keepdims		If this is set to `True`, the reduced axis is left in the result as dimension with size one.
  * @return null
  */
  def norm(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def radians(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def relu(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def rint(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def round(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def rsqrt(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param mom		Momentum
  * @param lr		Learning rate
  * @param momentum		The decay rate of momentum estimates at each epoch.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def sgd_mom_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param lr		Learning rate
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def sgd_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def sigmoid(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def sign(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def sin(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def sinh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Source input
  * @param begin		starting indices for the slice operation, supports negative indices.
  * @param end		ending indices for the slice operation, supports negative indices.
  * @param step		step for the slice operation, supports negative values.
  * @return null
  */
  def slice(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def softsign(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def sqrt(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def square(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def sum(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def tan(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def tanh(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def trunc(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param condition		condition array
  * @param x		
  * @param y		
  * @return null
  */
  def where(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @return null
  */
  def zeros_like(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param axis		The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``
  * @param keepdims		If this is set to `True`, the reduced axis is left in the result as dimension with size one.
  * @return null
  */
  def argmax(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array
  * @return null
  */
  def argmax_channel(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param axis		The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``
  * @param keepdims		If this is set to `True`, the reduced axis is left in the result as dimension with size one.
  * @return null
  */
  def argmin(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array
  * @param axis		Axis along which to sort the input tensor. If not given, the flattened array is used. Default is -1.
  * @param is_ascend		Whether to sort in ascending or descending order.
  * @return null
  */
  def argsort(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		The first input
  * @param rhs		The second input
  * @param transpose_a		If true then transpose the first input before dot.
  * @param transpose_b		If true then transpose the second input before dot.
  * @return null
  */
  def batch_dot(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param a		The input array
  * @param indices		The index array
  * @return null
  */
  def batch_take(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_add(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param axis		The axes to perform the broadcasting.
  * @param size		Target sizes of the broadcasting axes.
  * @return null
  */
  def broadcast_axis(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_div(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_equal(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_greater(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_greater_equal(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_hypot(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_lesser(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_lesser_equal(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_maximum(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_minimum(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_sub(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_mod(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_mul(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_not_equal(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param lhs		First input to the function
  * @param rhs		Second input to the function
  * @return null
  */
  def broadcast_power(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @param shape		The shape of the desired array. We can set the dim to zero if it's same as the original. E.g `A = broadcast_to(B, shape=(10, 0, 0))` has the same meaning as `A = broadcast_axis(B, axis=0, size=10)`.
  * @return null
  */
  def broadcast_to(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def cbrt(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.
  * @param lhs		Left operand to the function.
  * @param rhs		Right operand to the function.
  * @return null
  */
  def choose_element_0index(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Inserts a new axis of size 1 into the array shape
  * 
  * For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``
  * will return a new array with shape ``(2,1,3,4)``.
  * 
  * 
  * 
  * Defined in src/operator/tensor/matrix_op.cc:L345
  * @param data		Source input
  * @param axis		Position where new axis is to be inserted. Suppose that the input `NDArray`'s dimension is `ndim`, the range of the inserted axis is `[-ndim, ndim]`
  * @return null
  */
  def expand_dims(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.
  * @param lhs		Left operand to the function.
  * @param mhs		Middle operand to the function.
  * @param rhs		Right operand to the function.
  * @return null
  */
  def fill_element_0index(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data array
  * @param axis		The axis which to reverse elements.
  * @return null
  */
  def reverse(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param d		Internal state ``d_t``
  * @param v		Internal state ``v_t``
  * @param z		Internal state ``z_t``
  * @param lr		Learning rate
  * @param beta1		The decay rate for the 1st moment estimates.
  * @param beta2		The decay rate for the 2nd moment estimates.
  * @param epsilon		A small constant for numerical stability.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def ftml_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		data
  * @param indices		indices
  * @return null
  */
  def gather_nd(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param args		Positional input matrices
  * @return null
  */
  def khatri_rao(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @param axis		The axis along which to compute softmax.
  * @return null
  */
  def log_softmax(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the max of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def max(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the min of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L175
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def min(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Updater function for multi-precision sgd optimizer
  * @param weight		Weight
  * @param grad		Gradient
  * @param mom		Momentum
  * @param weight32		Weight32
  * @param lr		Learning rate
  * @param momentum		The decay rate of momentum estimates at each epoch.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def mp_sgd_mom_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Updater function for multi-precision sgd optimizer
  * @param weight		Weight
  * @param grad		gradient
  * @param weight32		Weight32
  * @param lr		Learning rate
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def mp_sgd_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L147
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def nanprod(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.
  * 
  * 
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L132
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def nansum(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param indices		array of locations where to set on_value
  * @param depth		Depth of the one hot dimension.
  * @param on_value		The value assigned to the locations represented by indices.
  * @param off_value		The value assigned to the locations not represented by indices.
  * @param dtype		DType of the output
  * @return null
  */
  def one_hot(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input
  * @return null
  */
  def ones_like(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array
  * @param index		The index array
  * @param axis		The axis along which to perform the reduction. Negative values means indexing from right to left. ``Requires axis to be set as int, because global reduction is not supported yet.``
  * @param keepdims		If this is set to `True`, the reduced axis is left in the result as dimension with size one.
  * @return null
  */
  def pick(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Computes the product of array elements over given axes.
  * 
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L117
  * @param data		The input
  * @param axis		The axis or axes along which to perform the reduction.

      The default, `axis=()`, will compute over all elements into a
      scalar array with shape `(1,)`.

      If `axis` is int, a reduction is performed on a particular axis.

      If `axis` is a tuple of ints, a reduction is performed on all the axes
      specified in the tuple.

      If `exclude` is true, reduction will be performed on the axes that are
      NOT in axis instead.

      Negative values means indexing from right to left.
  * @param keepdims		If this is set to `True`, the reduced axes are left in the result as dimension with size one.
  * @param exclude		Whether to perform reduction on axis that are NOT in axis instead.
  * @return null
  */
  def prod(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def rcbrt(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @return null
  */
  def reciprocal(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data array
  * @param repeats		The number of repetitions for each element.
  * @param axis		The axis along which to repeat values. The negative numbers are interpreted counting from the backward. By default, use the flattened input array, and return a flat output array.
  * @return null
  */
  def repeat(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

  /**
  * Reshape lhs to have the same shape as rhs.
  * @param lhs		First input.
  * @param rhs		Second input.
  * @return null
  */
  def reshape_like(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param n		n
  * @param lr		Learning rate
  * @param gamma1		The decay rate of momentum estimates.
  * @param epsilon		A small constant for numerical stability.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @param clip_weights		Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).
  * @return null
  */
  def rmsprop_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param n		n
  * @param g		g
  * @param delta		delta
  * @param lr		Learning rate
  * @param gamma1		Decay rate.
  * @param gamma2		Decay rate.
  * @param epsilon		A small constant for numerical stability.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @param clip_weights		Clip weights to the range of [-clip_weights, clip_weights] If clip_weights <= 0, weight clipping is turned off. weights = max(min(weights, clip_weights), -clip_weights).
  * @return null
  */
  def rmspropalex_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		data
  * @param indices		indices
  * @param shape		Shape of output.
  * @return null
  */
  def scatter_nd(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param lr		Learning rate
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @return null
  */
  def signsgd_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param weight		Weight
  * @param grad		Gradient
  * @param mom		Momentum
  * @param lr		Learning rate
  * @param momentum		The decay rate of momentum estimates at each epoch.
  * @param wd		Weight decay augments the objective function with a regularization term that penalizes large weights. The penalty scales with the square of the magnitude of each weight.
  * @param rescale_grad		Rescale gradient to grad = rescale_grad*grad.
  * @param clip_gradient		Clip gradient to the range of [-clip_gradient, clip_gradient] If clip_gradient <= 0, gradient clipping is turned off. grad = max(min(grad, clip_gradient), -clip_gradient).
  * @param wd_lh		The amount of weight decay that does not go into gradient/momentum calculationsotherwise do weight decay algorithmically only.
  * @return null
  */
  def signum_update(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Source input
  * @param axis		Axis along which to be sliced, supports negative indexes.
  * @param begin		The beginning index along the axis to be sliced,  supports negative indexes.
  * @param end		The ending index along the axis to be sliced,  supports negative indexes.
  * @return null
  */
  def slice_axis(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		source input
  * @param scalar		scalar input
  * @return null
  */
  def smooth_l1(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array.
  * @param axis		The axis along which to compute softmax.
  * @return null
  */
  def softmax(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data
  * @param label		Input label
  * @return null
  */
  def softmax_cross_entropy(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array
  * @param axis		Axis along which to choose sort the input tensor. If not given, the flattened array is used. Default is -1.
  * @param is_ascend		Whether to sort in ascending or descending order.
  * @return null
  */
  def sort(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		data to squeeze
  * @param axis		Selects a subset of the single-dimensional entries in the shape. If an axis is selected with shape entry greater than one, an error is raised.
  * @return null
  */
  def squeeze(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		List of arrays to stack
  * @param axis		The axis in the result array along which the input arrays are stacked.
  * @param num_args		Number of inputs to be stacked.
  * @return null
  */
  def stack(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param a		The input array.
  * @param indices		The indices of the values to be extracted.
  * @param axis		The axis of input array to be taken.
  * @param mode		Specify how out-of-bound indices bahave. "clip" means clip to the range. So, if all indices mentioned are too large, they are replaced by the index that addresses the last element along an axis.  "wrap" means to wrap around.  "raise" means to raise an error. 
  * @return null
  */
  def take(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Input data array
  * @param reps		The number of times for repeating the tensor a. If reps has length d, the result will have dimension of max(d, a.ndim); If a.ndim < d, a is promoted to be d-dimensional by prepending new axes. If a.ndim > d, reps is promoted to a.ndim by pre-pending 1's to it.
  * @return null
  */
  def tile(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		The input array
  * @param axis		Axis along which to choose the top k indices. If not given, the flattened array is used. Default is -1.
  * @param k		Number of top elements to select, should be always smaller than or equal to the element number in the given axis. A global sort is performed if set k < 1.
  * @param ret_typ		The return type.
 "value" means to return the top k values, "indices" means to return the indices of the top k values, "mask" means to return a mask array containing 0 and 1. 1 means the top k values. "both" means to return a list of both values and indices of top k elements.
  * @param is_ascend		Whether to choose k largest or k smallest elements. Top K largest elements will be chosen if set to false.
  * @return null
  */
  def topk(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol

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
  * @param data		Source input
  * @param axes		Target axis order. By default the axes will be inverted.
  * @return null
  */
  def transpose(name : scala.Predef.String, attr : scala.Predef.Map[scala.Predef.String, scala.Predef.String])(args : org.apache.mxnet.Symbol*)(kwargs : scala.Predef.Map[scala.Predef.String, scala.Any]) : org.apache.mxnet.Symbol



}
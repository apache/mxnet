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
import org.apache.mxnet.annotation.Experimental
abstract class SymbolBase {
  /**
  * Applies an activation function element-wise to the input.<br>
  * <br>
  * The following activation functions are supported:<br>
  * <br>
  * - `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`<br>
  * - `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`<br>
  * - `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`<br>
  * - `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`<br>
  * - `softsign`: :math:`y = \frac{x}{1 + abs(x)}`<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/activation.cc:L161<br>
  * @return org.apache.mxnet.Symbol
  */
def Activation(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Batch normalization.<br>
  * <br>
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as<br>
  * well as offset ``beta``.<br>
  * <br>
  * Assume the input has more than one dimension and we normalize along axis 1.<br>
  * We first compute the mean and variance along this axis:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   data\_mean[i] = mean(data[:,i,:,...]) \\<br>
  *   data\_var[i] = var(data[:,i,:,...])<br>
  * <br>
  * Then compute the normalized output, which has the same shape as input, as following:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]<br>
  * <br>
  * Both *mean* and *var* returns a scalar by treating the input as a vector.<br>
  * <br>
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``<br>
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and<br>
  * the inverse of ``data_var``, which are needed for the backward pass. Note that gradient of these <br>
  * two outputs are blocked.<br>
  * <br>
  * Besides the inputs and the outputs, this operator accepts two auxiliary<br>
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length<br>
  * vectors. They are global statistics for the whole dataset, which are updated<br>
  * by::<br>
  * <br>
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)<br>
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)<br>
  * <br>
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and<br>
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute<br>
  * the output. It is often used during inference.<br>
  * <br>
  * The parameter ``axis`` specifies which axis of the input shape denotes<br>
  * the 'channel' (separately normalized groups).  The default is 1.  Specifying -1 sets the channel<br>
  * axis to be the last item in the input shape.<br>
  * <br>
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,<br>
  * then set ``gamma`` to 1 and its gradient to 0.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/batch_norm.cc:L575<br>
  * @return org.apache.mxnet.Symbol
  */
def BatchNorm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Batch normalization.<br>
  * <br>
  * This operator is DEPRECATED. Perform BatchNorm on the input.<br>
  * <br>
  * Normalizes a data batch by mean and variance, and applies a scale ``gamma`` as<br>
  * well as offset ``beta``.<br>
  * <br>
  * Assume the input has more than one dimension and we normalize along axis 1.<br>
  * We first compute the mean and variance along this axis:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   data\_mean[i] = mean(data[:,i,:,...]) \\<br>
  *   data\_var[i] = var(data[:,i,:,...])<br>
  * <br>
  * Then compute the normalized output, which has the same shape as input, as following:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   out[:,i,:,...] = \frac{data[:,i,:,...] - data\_mean[i]}{\sqrt{data\_var[i]+\epsilon}} * gamma[i] + beta[i]<br>
  * <br>
  * Both *mean* and *var* returns a scalar by treating the input as a vector.<br>
  * <br>
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``<br>
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and<br>
  * ``data_var`` as well, which are needed for the backward pass.<br>
  * <br>
  * Besides the inputs and the outputs, this operator accepts two auxiliary<br>
  * states, ``moving_mean`` and ``moving_var``, which are *k*-length<br>
  * vectors. They are global statistics for the whole dataset, which are updated<br>
  * by::<br>
  * <br>
  *   moving_mean = moving_mean * momentum + data_mean * (1 - momentum)<br>
  *   moving_var = moving_var * momentum + data_var * (1 - momentum)<br>
  * <br>
  * If ``use_global_stats`` is set to be true, then ``moving_mean`` and<br>
  * ``moving_var`` are used instead of ``data_mean`` and ``data_var`` to compute<br>
  * the output. It is often used during inference.<br>
  * <br>
  * Both ``gamma`` and ``beta`` are learnable parameters. But if ``fix_gamma`` is true,<br>
  * then set ``gamma`` to 1 and its gradient to 0.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/batch_norm_v1.cc:L92<br>
  * @return org.apache.mxnet.Symbol
  */
def BatchNorm_v1(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies bilinear sampling to input feature map.<br>
  * <br>
  * Bilinear Sampling is the key of  [NIPS2015] \"Spatial Transformer Networks\". The usage of the operator is very similar to remap function in OpenCV,<br>
  * except that the operator has the backward pass.<br>
  * <br>
  * Given :math:`data` and :math:`grid`, then the output is computed by<br>
  * <br>
  * .. math::<br>
  *   x_{src} = grid[batch, 0, y_{dst}, x_{dst}] \\<br>
  *   y_{src} = grid[batch, 1, y_{dst}, x_{dst}] \\<br>
  *   output[batch, channel, y_{dst}, x_{dst}] = G(data[batch, channel, y_{src}, x_{src})<br>
  * <br>
  * :math:`x_{dst}`, :math:`y_{dst}` enumerate all spatial locations in :math:`output`, and :math:`G()` denotes the bilinear interpolation kernel.<br>
  * The out-boundary points will be padded with zeros.The shape of the output will be (data.shape[0], data.shape[1], grid.shape[2], grid.shape[3]).<br>
  * <br>
  * The operator assumes that :math:`data` has 'NCHW' layout and :math:`grid` has been normalized to [-1, 1].<br>
  * <br>
  * BilinearSampler often cooperates with GridGenerator which generates sampling grids for BilinearSampler.<br>
  * GridGenerator supports two kinds of transformation: ``affine`` and ``warp``.<br>
  * If users want to design a CustomOp to manipulate :math:`grid`, please firstly refer to the code of GridGenerator.<br>
  * <br>
  * Example 1::<br>
  * <br>
  *   ## Zoom out data two times<br>
  *   data = array([[[[1, 4, 3, 6],<br>
  *                   [1, 8, 8, 9],<br>
  *                   [0, 4, 1, 5],<br>
  *                   [1, 0, 1, 3]]]])<br>
  * <br>
  *   affine_matrix = array([[2, 0, 0],<br>
  *                          [0, 2, 0]])<br>
  * <br>
  *   affine_matrix = reshape(affine_matrix, shape=(1, 6))<br>
  * <br>
  *   grid = GridGenerator(data=affine_matrix, transform_type='affine', target_shape=(4, 4))<br>
  * <br>
  *   out = BilinearSampler(data, grid)<br>
  * <br>
  *   out<br>
  *   [[[[ 0,   0,     0,   0],<br>
  *      [ 0,   3.5,   6.5, 0],<br>
  *      [ 0,   1.25,  2.5, 0],<br>
  *      [ 0,   0,     0,   0]]]<br>
  * <br>
  * <br>
  * Example 2::<br>
  * <br>
  *   ## shift data horizontally by -1 pixel<br>
  * <br>
  *   data = array([[[[1, 4, 3, 6],<br>
  *                   [1, 8, 8, 9],<br>
  *                   [0, 4, 1, 5],<br>
  *                   [1, 0, 1, 3]]]])<br>
  * <br>
  *   warp_maxtrix = array([[[[1, 1, 1, 1],<br>
  *                           [1, 1, 1, 1],<br>
  *                           [1, 1, 1, 1],<br>
  *                           [1, 1, 1, 1]],<br>
  *                          [[0, 0, 0, 0],<br>
  *                           [0, 0, 0, 0],<br>
  *                           [0, 0, 0, 0],<br>
  *                           [0, 0, 0, 0]]]])<br>
  * <br>
  *   grid = GridGenerator(data=warp_matrix, transform_type='warp')<br>
  *   out = BilinearSampler(data, grid)<br>
  * <br>
  *   out<br>
  *   [[[[ 4,  3,  6,  0],<br>
  *      [ 8,  8,  9,  0],<br>
  *      [ 4,  1,  5,  0],<br>
  *      [ 0,  1,  3,  0]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/bilinear_sampler.cc:L245<br>
  * @return org.apache.mxnet.Symbol
  */
def BilinearSampler(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Stops gradient computation.<br>
  * <br>
  * Stops the accumulated gradient of the inputs from flowing through this operator<br>
  * in the backward direction. In other words, this operator prevents the contribution<br>
  * of its inputs to be taken into account for computing gradients.<br>
  * <br>
  * Example::<br>
  * <br>
  *   v1 = [1, 2]<br>
  *   v2 = [0, 1]<br>
  *   a = Variable('a')<br>
  *   b = Variable('b')<br>
  *   b_stop_grad = stop_gradient(3 * b)<br>
  *   loss = MakeLoss(b_stop_grad + a)<br>
  * <br>
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))<br>
  *   executor.forward(is_train=True, a=v1, b=v2)<br>
  *   executor.outputs<br>
  *   [ 1.  5.]<br>
  * <br>
  *   executor.backward()<br>
  *   executor.grad_arrays<br>
  *   [ 0.  0.]<br>
  *   [ 1.  1.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L270<br>
  * @return org.apache.mxnet.Symbol
  */
def BlockGrad(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Casts all elements of the input to a new type.<br>
  * <br>
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.<br>
  * <br>
  * Example::<br>
  * <br>
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]<br>
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]<br>
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L415<br>
  * @return org.apache.mxnet.Symbol
  */
def Cast(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Joins input arrays along a given axis.<br>
  * <br>
  * .. note:: `Concat` is deprecated. Use `concat` instead.<br>
  * <br>
  * The dimensions of the input arrays should be the same except the axis along<br>
  * which they will be concatenated.<br>
  * The dimension of the output array along the concatenated axis will be equal<br>
  * to the sum of the corresponding dimensions of the input arrays.<br>
  * <br>
  * The storage type of ``concat`` output depends on storage types of inputs<br>
  * <br>
  * - concat(csr, csr, ..., csr, dim=0) = csr<br>
  * - otherwise, ``concat`` generates output with default storage<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[1,1],[2,2]]<br>
  *    y = [[3,3],[4,4],[5,5]]<br>
  *    z = [[6,6], [7,7],[8,8]]<br>
  * <br>
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],<br>
  *                           [ 2.,  2.],<br>
  *                           [ 3.,  3.],<br>
  *                           [ 4.,  4.],<br>
  *                           [ 5.,  5.],<br>
  *                           [ 6.,  6.],<br>
  *                           [ 7.,  7.],<br>
  *                           [ 8.,  8.]]<br>
  * <br>
  *    Note that you cannot concat x,y,z along dimension 1 since dimension<br>
  *    0 is not the same for all the input arrays.<br>
  * <br>
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],<br>
  *                          [ 4.,  4.,  7.,  7.],<br>
  *                          [ 5.,  5.,  8.,  8.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/concat.cc:L260<br>
  * @return org.apache.mxnet.Symbol
  */
def Concat(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Compute *N*-D convolution on *(N+2)*-D input.<br>
  * <br>
  * In the 2-D convolution, given input data with shape *(batch_size,<br>
  * channel, height, width)*, the output is computed by<br>
  * <br>
  * .. math::<br>
  * <br>
  *    out[n,i,:,:] = bias[i] + \sum_{j=0}^{channel} data[n,j,:,:] \star<br>
  *    weight[i,j,:,:]<br>
  * <br>
  * where :math:`\star` is the 2-D cross-correlation operator.<br>
  * <br>
  * For general 2-D convolution, the shapes are<br>
  * <br>
  * - **data**: *(batch_size, channel, height, width)*<br>
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1])*<br>
  * - **bias**: *(num_filter,)*<br>
  * - **out**: *(batch_size, num_filter, out_height, out_width)*.<br>
  * <br>
  * Define::<br>
  * <br>
  *   f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1<br>
  * <br>
  * then we have::<br>
  * <br>
  *   out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])<br>
  *   out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])<br>
  * <br>
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.<br>
  * <br>
  * The default data ``layout`` is *NCHW*, namely *(batch_size, channel, height,<br>
  * width)*. We can choose other layouts such as *NHWC*.<br>
  * <br>
  * If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``<br>
  * evenly into *g* parts along the channel axis, and also evenly split ``weight``<br>
  * along the first dimension. Next compute the convolution on the *i*-th part of<br>
  * the data with the *i*-th weight part. The output is obtained by concatenating all<br>
  * the *g* results.<br>
  * <br>
  * 1-D convolution does not have *height* dimension but only *width* in space.<br>
  * <br>
  * - **data**: *(batch_size, channel, width)*<br>
  * - **weight**: *(num_filter, channel, kernel[0])*<br>
  * - **bias**: *(num_filter,)*<br>
  * - **out**: *(batch_size, num_filter, out_width)*.<br>
  * <br>
  * 3-D convolution adds an additional *depth* dimension besides *height* and<br>
  * *width*. The shapes are<br>
  * <br>
  * - **data**: *(batch_size, channel, depth, height, width)*<br>
  * - **weight**: *(num_filter, channel, kernel[0], kernel[1], kernel[2])*<br>
  * - **bias**: *(num_filter,)*<br>
  * - **out**: *(batch_size, num_filter, out_depth, out_height, out_width)*.<br>
  * <br>
  * Both ``weight`` and ``bias`` are learnable parameters.<br>
  * <br>
  * There are other options to tune the performance.<br>
  * <br>
  * - **cudnn_tune**: enable this option leads to higher startup time but may give<br>
  *   faster speed. Options are<br>
  * <br>
  *   - **off**: no tuning<br>
  *   - **limited_workspace**:run test and pick the fastest algorithm that doesn't<br>
  *     exceed workspace limit.<br>
  *   - **fastest**: pick the fastest algorithm and ignore workspace limit.<br>
  *   - **None** (default): the behavior is determined by environment variable<br>
  *     ``MXNET_CUDNN_AUTOTUNE_DEFAULT``. 0 for off, 1 for limited workspace<br>
  *     (default), 2 for fastest.<br>
  * <br>
  * - **workspace**: A large number leads to more (GPU) memory usage but may improve<br>
  *   the performance.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/convolution.cc:L470<br>
  * @return org.apache.mxnet.Symbol
  */
def Convolution(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * This operator is DEPRECATED. Apply convolution to input then add a bias.<br>
  * @return org.apache.mxnet.Symbol
  */
def Convolution_v1(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies correlation to inputs.<br>
  * <br>
  * The correlation layer performs multiplicative patch comparisons between two feature maps.<br>
  * <br>
  * Given two multi-channel feature maps :math:`f_{1}, f_{2}`, with :math:`w`, :math:`h`, and :math:`c` being their width, height, and number of channels,<br>
  * the correlation layer lets the network compare each patch from :math:`f_{1}` with each patch from :math:`f_{2}`.<br>
  * <br>
  * For now we consider only a single comparison of two patches. The 'correlation' of two patches centered at :math:`x_{1}` in the first map and<br>
  * :math:`x_{2}` in the second map is then defined as:<br>
  * <br>
  * .. math::<br>
  * <br>
  *    c(x_{1}, x_{2}) = \sum_{o \in [-k,k] \times [-k,k]} <f_{1}(x_{1} + o), f_{2}(x_{2} + o)><br>
  * <br>
  * for a square patch of size :math:`K:=2k+1`.<br>
  * <br>
  * Note that the equation above is identical to one step of a convolution in neural networks, but instead of convolving data with a filter, it convolves data with other<br>
  * data. For this reason, it has no training weights.<br>
  * <br>
  * Computing :math:`c(x_{1}, x_{2})` involves :math:`c * K^{2}` multiplications. Comparing all patch combinations involves :math:`w^{2}*h^{2}` such computations.<br>
  * <br>
  * Given a maximum displacement :math:`d`, for each location :math:`x_{1}` it computes correlations :math:`c(x_{1}, x_{2})` only in a neighborhood of size :math:`D:=2d+1`,<br>
  * by limiting the range of :math:`x_{2}`. We use strides :math:`s_{1}, s_{2}`, to quantize :math:`x_{1}` globally and to quantize :math:`x_{2}` within the neighborhood<br>
  * centered around :math:`x_{1}`.<br>
  * <br>
  * The final output is defined by the following expression:<br>
  * <br>
  * .. math::<br>
  *   out[n, q, i, j] = c(x_{i, j}, x_{q})<br>
  * <br>
  * where :math:`i` and :math:`j` enumerate spatial locations in :math:`f_{1}`, and :math:`q` denotes the :math:`q^{th}` neighborhood of :math:`x_{i,j}`.<br>
  * <br>
  * <br>
  * Defined in src/operator/correlation.cc:L198<br>
  * @return org.apache.mxnet.Symbol
  */
def Correlation(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * <br>
  * <br>
  * .. note:: `Crop` is deprecated. Use `slice` instead.<br>
  * <br>
  * Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or<br>
  * with width and height of the second input symbol, i.e., with one input, we need h_w to<br>
  * specify the crop height and width, otherwise the second input symbol's size will be used<br>
  * <br>
  * <br>
  * Defined in src/operator/crop.cc:L50<br>
  * @return org.apache.mxnet.Symbol
  */
def Crop(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Apply a custom operator implemented in a frontend language (like Python).<br>
  * <br>
  * Custom operators should override required methods like `forward` and `backward`.<br>
  * The custom operator must be registered before it can be used.<br>
  * Please check the tutorial here: http://mxnet.io/faq/new_op.html.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/custom/custom.cc:L547<br>
  * @return org.apache.mxnet.Symbol
  */
def Custom(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes 1D or 2D transposed convolution (aka fractionally strided convolution) of the input tensor. This operation can be seen as the gradient of Convolution operation with respect to its input. Convolution usually reduces the size of the input. Transposed convolution works the other way, going from a smaller input to a larger output while preserving the connectivity pattern.<br>
  * @return org.apache.mxnet.Symbol
  */
def Deconvolution(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies dropout operation to input array.<br>
  * <br>
  * - During training, each element of the input is set to zero with probability p.<br>
  *   The whole array is rescaled by :math:`1/(1-p)` to keep the expected<br>
  *   sum of the input unchanged.<br>
  * <br>
  * - During testing, this operator does not change the input if mode is 'training'.<br>
  *   If mode is 'always', the same computaion as during training will be applied.<br>
  * <br>
  * Example::<br>
  * <br>
  *   random.seed(998)<br>
  *   input_array = array([[3., 0.5,  -0.5,  2., 7.],<br>
  *                       [2., -0.4,   7.,  3., 0.2]])<br>
  *   a = symbol.Variable('a')<br>
  *   dropout = symbol.Dropout(a, p = 0.2)<br>
  *   executor = dropout.simple_bind(a = input_array.shape)<br>
  * <br>
  *   ## If training<br>
  *   executor.forward(is_train = True, a = input_array)<br>
  *   executor.outputs<br>
  *   [[ 3.75   0.625 -0.     2.5    8.75 ]<br>
  *    [ 2.5   -0.5    8.75   3.75   0.   ]]<br>
  * <br>
  *   ## If testing<br>
  *   executor.forward(is_train = False, a = input_array)<br>
  *   executor.outputs<br>
  *   [[ 3.     0.5   -0.5    2.     7.   ]<br>
  *    [ 2.    -0.4    7.     3.     0.2  ]]<br>
  * <br>
  * <br>
  * Defined in src/operator/nn/dropout.cc:L76<br>
  * @return org.apache.mxnet.Symbol
  */
def Dropout(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Adds all input arguments element-wise.<br>
  * <br>
  * .. math::<br>
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n<br>
  * <br>
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.<br>
  * <br>
  * The storage type of ``add_n`` output depends on storage types of inputs<br>
  * <br>
  * - add_n(row_sparse, row_sparse, ..) = row_sparse<br>
  * - otherwise, ``add_n`` generates output with default storage<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_sum.cc:L150<br>
  * @return org.apache.mxnet.Symbol
  */
def ElementWiseSum(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Maps integer indices to vector representations (embeddings).<br>
  * <br>
  * This operator maps words to real-valued vectors in a high-dimensional space,<br>
  * called word embeddings. These embeddings can capture semantic and syntactic properties of the words.<br>
  * For example, it has been noted that in the learned embedding spaces, similar words tend<br>
  * to be close to each other and dissimilar words far apart.<br>
  * <br>
  * For an input array of shape (d1, ..., dK),<br>
  * the shape of an output array is (d1, ..., dK, output_dim).<br>
  * All the input values should be integers in the range [0, input_dim).<br>
  * <br>
  * If the input_dim is ip0 and output_dim is op0, then shape of the embedding weight matrix must be<br>
  * (ip0, op0).<br>
  * <br>
  * By default, if any index mentioned is too large, it is replaced by the index that addresses<br>
  * the last vector in an embedding matrix.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   input_dim = 4<br>
  *   output_dim = 5<br>
  * <br>
  *   // Each row in weight matrix y represents a word. So, y = (w0,w1,w2,w3)<br>
  *   y = [[  0.,   1.,   2.,   3.,   4.],<br>
  *        [  5.,   6.,   7.,   8.,   9.],<br>
  *        [ 10.,  11.,  12.,  13.,  14.],<br>
  *        [ 15.,  16.,  17.,  18.,  19.]]<br>
  * <br>
  *   // Input array x represents n-grams(2-gram). So, x = [(w1,w3), (w0,w2)]<br>
  *   x = [[ 1.,  3.],<br>
  *        [ 0.,  2.]]<br>
  * <br>
  *   // Mapped input x to its vector representation y.<br>
  *   Embedding(x, y, 4, 5) = [[[  5.,   6.,   7.,   8.,   9.],<br>
  *                             [ 15.,  16.,  17.,  18.,  19.]],<br>
  * <br>
  *                            [[  0.,   1.,   2.,   3.,   4.],<br>
  *                             [ 10.,  11.,  12.,  13.,  14.]]]<br>
  * <br>
  * <br>
  * The storage type of weight can be either row_sparse or default, while<br>
  * the storage type of weight's grad depends on the value of "sparse_grad".<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/indexing_op.cc:L232<br>
  * @return org.apache.mxnet.Symbol
  */
def Embedding(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.<br>
  * <br>
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.<br>
  * <br>
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes<br>
  * the input array into an output array of shape ``(d1, d2*...*dk)``.<br>
  * <br>
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,<br>
  * which behaves similar to mxnet.ndarray.reshape((-1,)).<br>
  * <br>
  * Example::<br>
  * <br>
  *     x = [[<br>
  *         [1,2,3],<br>
  *         [4,5,6],<br>
  *         [7,8,9]<br>
  *     ],<br>
  *     [    [1,2,3],<br>
  *         [4,5,6],<br>
  *         [7,8,9]<br>
  *     ]],<br>
  * <br>
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],<br>
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L258<br>
  * @return org.apache.mxnet.Symbol
  */
def Flatten(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies a linear transformation: :math:`Y = XW^T + b`.<br>
  * <br>
  * If ``flatten`` is set to be true, then the shapes are:<br>
  * <br>
  * - **data**: `(batch_size, x1, x2, ..., xn)`<br>
  * - **weight**: `(num_hidden, x1 * x2 * ... * xn)`<br>
  * - **bias**: `(num_hidden,)`<br>
  * - **out**: `(batch_size, num_hidden)`<br>
  * <br>
  * If ``flatten`` is set to be false, then the shapes are:<br>
  * <br>
  * - **data**: `(x1, x2, ..., xn, input_dim)`<br>
  * - **weight**: `(num_hidden, input_dim)`<br>
  * - **bias**: `(num_hidden,)`<br>
  * - **out**: `(x1, x2, ..., xn, num_hidden)`<br>
  * <br>
  * The learnable parameters include both ``weight`` and ``bias``.<br>
  * <br>
  * If ``no_bias`` is set to be true, then the ``bias`` term is ignored.<br>
  * <br>
  * Note that the operator also supports forward computation with `row_sparse` weight and bias,<br>
  * where the length of `weight.indices` and `bias.indices` must be equal to `num_hidden`.<br>
  * This could be used for model inference with `row_sparse` weights trained with `SparseEmbedding`.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/fully_connected.cc:L254<br>
  * @return org.apache.mxnet.Symbol
  */
def FullyConnected(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Generates 2D sampling grid for bilinear sampling.<br>
  * @return org.apache.mxnet.Symbol
  */
def GridGenerator(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Apply a sparse regularization to the output a sigmoid activation function.<br>
  * @return org.apache.mxnet.Symbol
  */
def IdentityAttachKLSparseReg(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies instance normalization to the n-dimensional input array.<br>
  * <br>
  * This operator takes an n-dimensional input array where (n>2) and normalizes<br>
  * the input using the following formula:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   out = \frac{x - mean[data]}{ \sqrt{Var[data]} + \epsilon} * gamma + beta<br>
  * <br>
  * This layer is similar to batch normalization layer (`BatchNorm`)<br>
  * with two differences: first, the normalization is<br>
  * carried out per example (instance), not over a batch. Second, the<br>
  * same normalization is applied both at test and train time. This<br>
  * operation is also known as `contrast normalization`.<br>
  * <br>
  * If the input data is of shape [batch, channel, spacial_dim1, spacial_dim2, ...],<br>
  * `gamma` and `beta` parameters must be vectors of shape [channel].<br>
  * <br>
  * This implementation is based on paper:<br>
  * <br>
  * .. [1] Instance Normalization: The Missing Ingredient for Fast Stylization,<br>
  *    D. Ulyanov, A. Vedaldi, V. Lempitsky, 2016 (arXiv:1607.08022v2).<br>
  * <br>
  * Examples::<br>
  * <br>
  *   // Input of shape (2,1,2)<br>
  *   x = [[[ 1.1,  2.2]],<br>
  *        [[ 3.3,  4.4]]]<br>
  * <br>
  *   // gamma parameter of length 1<br>
  *   gamma = [1.5]<br>
  * <br>
  *   // beta parameter of length 1<br>
  *   beta = [0.5]<br>
  * <br>
  *   // Instance normalization is calculated with the above formula<br>
  *   InstanceNorm(x,gamma,beta) = [[[-0.997527  ,  1.99752665]],<br>
  *                                 [[-0.99752653,  1.99752724]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/instance_norm.cc:L95<br>
  * @return org.apache.mxnet.Symbol
  */
def InstanceNorm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Normalize the input array using the L2 norm.<br>
  * <br>
  * For 1-D NDArray, it computes::<br>
  * <br>
  *   out = data / sqrt(sum(data ** 2) + eps)<br>
  * <br>
  * For N-D NDArray, if the input array has shape (N, N, ..., N),<br>
  * <br>
  * with ``mode`` = ``instance``, it normalizes each instance in the multidimensional<br>
  * array by its L2 norm.::<br>
  * <br>
  *   for i in 0...N<br>
  *     out[i,:,:,...,:] = data[i,:,:,...,:] / sqrt(sum(data[i,:,:,...,:] ** 2) + eps)<br>
  * <br>
  * with ``mode`` = ``channel``, it normalizes each channel in the array by its L2 norm.::<br>
  * <br>
  *   for i in 0...N<br>
  *     out[:,i,:,...,:] = data[:,i,:,...,:] / sqrt(sum(data[:,i,:,...,:] ** 2) + eps)<br>
  * <br>
  * with ``mode`` = ``spatial``, it normalizes the cross channel norm for each position<br>
  * in the array by its L2 norm.::<br>
  * <br>
  *   for dim in 2...N<br>
  *     for i in 0...N<br>
  *       out[.....,i,...] = take(out, indices=i, axis=dim) / sqrt(sum(take(out, indices=i, axis=dim) ** 2) + eps)<br>
  *           -dim-<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[[1,2],<br>
  *         [3,4]],<br>
  *        [[2,2],<br>
  *         [5,6]]]<br>
  * <br>
  *   L2Normalization(x, mode='instance')<br>
  *   =[[[ 0.18257418  0.36514837]<br>
  *      [ 0.54772252  0.73029673]]<br>
  *     [[ 0.24077171  0.24077171]<br>
  *      [ 0.60192931  0.72231513]]]<br>
  * <br>
  *   L2Normalization(x, mode='channel')<br>
  *   =[[[ 0.31622776  0.44721359]<br>
  *      [ 0.94868326  0.89442718]]<br>
  *     [[ 0.37139067  0.31622776]<br>
  *      [ 0.92847669  0.94868326]]]<br>
  * <br>
  *   L2Normalization(x, mode='spatial')<br>
  *   =[[[ 0.44721359  0.89442718]<br>
  *      [ 0.60000002  0.80000001]]<br>
  *     [[ 0.70710677  0.70710677]<br>
  *      [ 0.6401844   0.76822126]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/l2_normalization.cc:L98<br>
  * @return org.apache.mxnet.Symbol
  */
def L2Normalization(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies local response normalization to the input.<br>
  * <br>
  * The local response normalization layer performs "lateral inhibition" by normalizing<br>
  * over local input regions.<br>
  * <br>
  * If :math:`a_{x,y}^{i}` is the activity of a neuron computed by applying kernel :math:`i` at position<br>
  * :math:`(x, y)` and then applying the ReLU nonlinearity, the response-normalized<br>
  * activity :math:`b_{x,y}^{i}` is given by the expression:<br>
  * <br>
  * .. math::<br>
  *    b_{x,y}^{i} = \frac{a_{x,y}^{i}}{\Bigg({k + \frac{\alpha}{n} \sum_{j=max(0, i-\frac{n}{2})}^{min(N-1, i+\frac{n}{2})} (a_{x,y}^{j})^{2}}\Bigg)^{\beta}}<br>
  * <br>
  * where the sum runs over :math:`n` "adjacent" kernel maps at the same spatial position, and :math:`N` is the total<br>
  * number of kernels in the layer.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/lrn.cc:L175<br>
  * @return org.apache.mxnet.Symbol
  */
def LRN(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Layer normalization.<br>
  * <br>
  * Normalizes the channels of the input tensor by mean and variance, and applies a scale ``gamma`` as<br>
  * well as offset ``beta``.<br>
  * <br>
  * Assume the input has more than one dimension and we normalize along axis 1.<br>
  * We first compute the mean and variance along this axis and then <br>
  * compute the normalized output, which has the same shape as input, as following:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   out = \frac{data - mean(data, axis)}{\sqrt{var(data, axis) + \epsilon}} * gamma + beta<br>
  * <br>
  * Both ``gamma`` and ``beta`` are learnable parameters.<br>
  * <br>
  * Unlike BatchNorm and InstanceNorm,  the *mean* and *var* are computed along the channel dimension.<br>
  * <br>
  * Assume the input has size *k* on axis 1, then both ``gamma`` and ``beta``<br>
  * have shape *(k,)*. If ``output_mean_var`` is set to be true, then outputs both ``data_mean`` and<br>
  * ``data_std``. Note that no gradient will be passed through these two outputs.<br>
  * <br>
  * The parameter ``axis`` specifies which axis of the input shape denotes<br>
  * the 'channel' (separately normalized groups).  The default is -1, which sets the channel<br>
  * axis to be the last item in the input shape.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/layer_norm.cc:L94<br>
  * @return org.apache.mxnet.Symbol
  */
def LayerNorm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies Leaky rectified linear unit activation element-wise to the input.<br>
  * <br>
  * Leaky ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope`<br>
  * when the input is negative and has a slope of one when input is positive.<br>
  * <br>
  * The following modified ReLU Activation functions are supported:<br>
  * <br>
  * - *elu*: Exponential Linear Unit. `y = x > 0 ? x : slope * (exp(x)-1)`<br>
  * - *leaky*: Leaky ReLU. `y = x > 0 ? x : slope * x`<br>
  * - *prelu*: Parametric ReLU. This is same as *leaky* except that `slope` is learnt during training.<br>
  * - *rrelu*: Randomized ReLU. same as *leaky* but the `slope` is uniformly and randomly chosen from<br>
  *   *[lower_bound, upper_bound)* for training, while fixed to be<br>
  *   *(lower_bound+upper_bound)/2* for inference.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/leaky_relu.cc:L63<br>
  * @return org.apache.mxnet.Symbol
  */
def LeakyReLU(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes and optimizes for squared loss during backward propagation.<br>
  * Just outputs ``data`` during forward propagation.<br>
  * <br>
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,<br>
  * then the squared loss estimated over :math:`n` samples is defined as<br>
  * <br>
  * :math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`<br>
  * <br>
  * .. note::<br>
  *    Use the LinearRegressionOutput as the final output layer of a net.<br>
  * <br>
  * The storage type of ``label`` can be ``default`` or ``csr``<br>
  * <br>
  * - LinearRegressionOutput(default, default) = default<br>
  * - LinearRegressionOutput(default, csr) = default<br>
  * <br>
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.<br>
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/regression_output.cc:L92<br>
  * @return org.apache.mxnet.Symbol
  */
def LinearRegressionOutput(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies a logistic function to the input.<br>
  * <br>
  * The logistic function, also known as the sigmoid function, is computed as<br>
  * :math:`\frac{1}{1+exp(-\textbf{x})}`.<br>
  * <br>
  * Commonly, the sigmoid is used to squash the real-valued output of a linear model<br>
  * :math:`wTx+b` into the [0,1] range so that it can be interpreted as a probability.<br>
  * It is suitable for binary classification or probability prediction tasks.<br>
  * <br>
  * .. note::<br>
  *    Use the LogisticRegressionOutput as the final output layer of a net.<br>
  * <br>
  * The storage type of ``label`` can be ``default`` or ``csr``<br>
  * <br>
  * - LogisticRegressionOutput(default, default) = default<br>
  * - LogisticRegressionOutput(default, csr) = default<br>
  * <br>
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.<br>
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/regression_output.cc:L148<br>
  * @return org.apache.mxnet.Symbol
  */
def LogisticRegressionOutput(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes mean absolute error of the input.<br>
  * <br>
  * MAE is a risk metric corresponding to the expected value of the absolute error.<br>
  * <br>
  * If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,<br>
  * then the mean absolute error (MAE) estimated over :math:`n` samples is defined as<br>
  * <br>
  * :math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`<br>
  * <br>
  * .. note::<br>
  *    Use the MAERegressionOutput as the final output layer of a net.<br>
  * <br>
  * The storage type of ``label`` can be ``default`` or ``csr``<br>
  * <br>
  * - MAERegressionOutput(default, default) = default<br>
  * - MAERegressionOutput(default, csr) = default<br>
  * <br>
  * By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of regression outputs of a training example.<br>
  * The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/regression_output.cc:L120<br>
  * @return org.apache.mxnet.Symbol
  */
def MAERegressionOutput(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Make your own loss function in network construction.<br>
  * <br>
  * This operator accepts a customized loss function symbol as a terminal loss and<br>
  * the symbol should be an operator with no backward dependency.<br>
  * The output of this function is the gradient of loss with respect to the input data.<br>
  * <br>
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the<br>
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::<br>
  * <br>
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)<br>
  *   loss = MakeLoss(cross_entropy)<br>
  * <br>
  * We will need to use ``MakeLoss`` when we are creating our own loss function or we want to<br>
  * combine multiple loss functions. Also we may want to stop some variables' gradients<br>
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.<br>
  * <br>
  * In addition, we can give a scale to the loss by setting ``grad_scale``,<br>
  * so that the gradient of the loss will be rescaled in the backpropagation.<br>
  * <br>
  * .. note:: This operator should be used as a Symbol instead of NDArray.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/make_loss.cc:L71<br>
  * @return org.apache.mxnet.Symbol
  */
def MakeLoss(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Pads an input array with a constant or edge values of the array.<br>
  * <br>
  * .. note:: `Pad` is deprecated. Use `pad` instead.<br>
  * <br>
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied<br>
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.<br>
  * <br>
  * This operation pads an input array with either a `constant_value` or edge values<br>
  * along each axis of the input array. The amount of padding is specified by `pad_width`.<br>
  * <br>
  * `pad_width` is a tuple of integer padding widths for each axis of the format<br>
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``<br>
  * where ``N`` is the number of dimensions of the array.<br>
  * <br>
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values<br>
  * to add before and after the elements of the array along dimension ``N``.<br>
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,<br>
  * ``after_2`` must be 0.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[[[  1.   2.   3.]<br>
  *           [  4.   5.   6.]]<br>
  * <br>
  *          [[  7.   8.   9.]<br>
  *           [ 10.  11.  12.]]]<br>
  * <br>
  * <br>
  *         [[[ 11.  12.  13.]<br>
  *           [ 14.  15.  16.]]<br>
  * <br>
  *          [[ 17.  18.  19.]<br>
  *           [ 20.  21.  22.]]]]<br>
  * <br>
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =<br>
  * <br>
  *          [[[[  1.   1.   2.   3.   3.]<br>
  *             [  1.   1.   2.   3.   3.]<br>
  *             [  4.   4.   5.   6.   6.]<br>
  *             [  4.   4.   5.   6.   6.]]<br>
  * <br>
  *            [[  7.   7.   8.   9.   9.]<br>
  *             [  7.   7.   8.   9.   9.]<br>
  *             [ 10.  10.  11.  12.  12.]<br>
  *             [ 10.  10.  11.  12.  12.]]]<br>
  * <br>
  * <br>
  *           [[[ 11.  11.  12.  13.  13.]<br>
  *             [ 11.  11.  12.  13.  13.]<br>
  *             [ 14.  14.  15.  16.  16.]<br>
  *             [ 14.  14.  15.  16.  16.]]<br>
  * <br>
  *            [[ 17.  17.  18.  19.  19.]<br>
  *             [ 17.  17.  18.  19.  19.]<br>
  *             [ 20.  20.  21.  22.  22.]<br>
  *             [ 20.  20.  21.  22.  22.]]]]<br>
  * <br>
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =<br>
  * <br>
  *          [[[[  0.   0.   0.   0.   0.]<br>
  *             [  0.   1.   2.   3.   0.]<br>
  *             [  0.   4.   5.   6.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]<br>
  * <br>
  *            [[  0.   0.   0.   0.   0.]<br>
  *             [  0.   7.   8.   9.   0.]<br>
  *             [  0.  10.  11.  12.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]]<br>
  * <br>
  * <br>
  *           [[[  0.   0.   0.   0.   0.]<br>
  *             [  0.  11.  12.  13.   0.]<br>
  *             [  0.  14.  15.  16.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]<br>
  * <br>
  *            [[  0.   0.   0.   0.   0.]<br>
  *             [  0.  17.  18.  19.   0.]<br>
  *             [  0.  20.  21.  22.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]]]<br>
  * <br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/pad.cc:L766<br>
  * @return org.apache.mxnet.Symbol
  */
def Pad(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs pooling on the input.<br>
  * <br>
  * The shapes for 1-D pooling are<br>
  * <br>
  * - **data**: *(batch_size, channel, width)*,<br>
  * - **out**: *(batch_size, num_filter, out_width)*.<br>
  * <br>
  * The shapes for 2-D pooling are<br>
  * <br>
  * - **data**: *(batch_size, channel, height, width)*<br>
  * - **out**: *(batch_size, num_filter, out_height, out_width)*, with::<br>
  * <br>
  *     out_height = f(height, kernel[0], pad[0], stride[0])<br>
  *     out_width = f(width, kernel[1], pad[1], stride[1])<br>
  * <br>
  * The definition of *f* depends on ``pooling_convention``, which has two options:<br>
  * <br>
  * - **valid** (default)::<br>
  * <br>
  *     f(x, k, p, s) = floor((x+2*p-k)/s)+1<br>
  * <br>
  * - **full**, which is compatible with Caffe::<br>
  * <br>
  *     f(x, k, p, s) = ceil((x+2*p-k)/s)+1<br>
  * <br>
  * But ``global_pool`` is set to be true, then do a global pooling, namely reset<br>
  * ``kernel=(height, width)``.<br>
  * <br>
  * Three pooling options are supported by ``pool_type``:<br>
  * <br>
  * - **avg**: average pooling<br>
  * - **max**: max pooling<br>
  * - **sum**: sum pooling<br>
  * - **lp**: Lp pooling<br>
  * <br>
  * For 3-D pooling, an additional *depth* dimension is added before<br>
  * *height*. Namely the input data will have shape *(batch_size, channel, depth,<br>
  * height, width)*.<br>
  * <br>
  * Notes on Lp pooling:<br>
  * <br>
  * Lp pooling was first introduced by this paper: https://arxiv.org/pdf/1204.3968.pdf.<br>
  * L-1 pooling is simply sum pooling, while L-inf pooling is simply max pooling.<br>
  * We can see that Lp pooling stands between those two, in practice the most common value for p is 2.<br>
  * <br>
  * For each window ``X``, the mathematical expression for Lp pooling is:<br>
  * <br>
  * ..math::<br>
  *   f(X) = \sqrt{p}{\sum\limits_{x \in X} x^p}<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/pooling.cc:L367<br>
  * @return org.apache.mxnet.Symbol
  */
def Pooling(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * This operator is DEPRECATED.<br>
  * Perform pooling on the input.<br>
  * <br>
  * The shapes for 2-D pooling is<br>
  * <br>
  * - **data**: *(batch_size, channel, height, width)*<br>
  * - **out**: *(batch_size, num_filter, out_height, out_width)*, with::<br>
  * <br>
  *     out_height = f(height, kernel[0], pad[0], stride[0])<br>
  *     out_width = f(width, kernel[1], pad[1], stride[1])<br>
  * <br>
  * The definition of *f* depends on ``pooling_convention``, which has two options:<br>
  * <br>
  * - **valid** (default)::<br>
  * <br>
  *     f(x, k, p, s) = floor((x+2*p-k)/s)+1<br>
  * <br>
  * - **full**, which is compatible with Caffe::<br>
  * <br>
  *     f(x, k, p, s) = ceil((x+2*p-k)/s)+1<br>
  * <br>
  * But ``global_pool`` is set to be true, then do a global pooling, namely reset<br>
  * ``kernel=(height, width)``.<br>
  * <br>
  * Three pooling options are supported by ``pool_type``:<br>
  * <br>
  * - **avg**: average pooling<br>
  * - **max**: max pooling<br>
  * - **sum**: sum pooling<br>
  * <br>
  * 1-D pooling is special case of 2-D pooling with *weight=1* and<br>
  * *kernel[1]=1*.<br>
  * <br>
  * For 3-D pooling, an additional *depth* dimension is added before<br>
  * *height*. Namely the input data will have shape *(batch_size, channel, depth,<br>
  * height, width)*.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/pooling_v1.cc:L104<br>
  * @return org.apache.mxnet.Symbol
  */
def Pooling_v1(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies recurrent layers to input data. Currently, vanilla RNN, LSTM and GRU are <br>
  * implemented, with both multi-layer and bidirectional support.<br>
  * <br>
  * **Vanilla RNN**<br>
  * <br>
  * Applies a single-gate recurrent layer to input X. Two kinds of activation function are supported: <br>
  * ReLU and Tanh.<br>
  * <br>
  * With ReLU activation function:<br>
  * <br>
  * .. math::<br>
  *     h_t = relu(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})<br>
  * <br>
  * With Tanh activtion function:<br>
  * <br>
  * .. math::<br>
  *     h_t = \tanh(W_{ih} * x_t + b_{ih}  +  W_{hh} * h_{(t-1)} + b_{hh})<br>
  * <br>
  * Reference paper: Finding structure in time - Elman, 1988. <br>
  * https://crl.ucsd.edu/~elman/Papers/fsit.pdf<br>
  * <br>
  * **LSTM**<br>
  * <br>
  * Long Short-Term Memory - Hochreiter, 1997. http://www.bioinf.jku.at/publications/older/2604.pdf<br>
  * <br>
  * .. math::<br>
  *   \begin{array}{ll}<br>
  *             i_t = \mathrm{sigmoid}(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi}) \\<br>
  *             f_t = \mathrm{sigmoid}(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf}) \\<br>
  *             g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg}) \\<br>
  *             o_t = \mathrm{sigmoid}(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho}) \\<br>
  *             c_t = f_t * c_{(t-1)} + i_t * g_t \\<br>
  *             h_t = o_t * \tanh(c_t)<br>
  *             \end{array}<br>
  * <br>
  * **GRU**<br>
  * <br>
  * Gated Recurrent Unit - Cho et al. 2014. http://arxiv.org/abs/1406.1078<br>
  * <br>
  * The definition of GRU here is slightly different from paper but compatible with CUDNN.<br>
  * <br>
  * .. math::<br>
  *   \begin{array}{ll}<br>
  *             r_t = \mathrm{sigmoid}(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\<br>
  *             z_t = \mathrm{sigmoid}(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\<br>
  *             n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\<br>
  *             h_t = (1 - z_t) * n_t + z_t * h_{(t-1)} \\<br>
  *             \end{array}<br>
  * @return org.apache.mxnet.Symbol
  */
def RNN(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs region of interest(ROI) pooling on the input array.<br>
  * <br>
  * ROI pooling is a variant of a max pooling layer, in which the output size is fixed and<br>
  * region of interest is a parameter. Its purpose is to perform max pooling on the inputs<br>
  * of non-uniform sizes to obtain fixed-size feature maps. ROI pooling is a neural-net<br>
  * layer mostly used in training a `Fast R-CNN` network for object detection.<br>
  * <br>
  * This operator takes a 4D feature map as an input array and region proposals as `rois`,<br>
  * then it pools over sub-regions of input and produces a fixed-sized output array<br>
  * regardless of the ROI size.<br>
  * <br>
  * To crop the feature map accordingly, you can resize the bounding box coordinates<br>
  * by changing the parameters `rois` and `spatial_scale`.<br>
  * <br>
  * The cropped feature maps are pooled by standard max pooling operation to a fixed size output<br>
  * indicated by a `pooled_size` parameter. batch_size will change to the number of region<br>
  * bounding boxes after `ROIPooling`.<br>
  * <br>
  * The size of each region of interest doesn't have to be perfectly divisible by<br>
  * the number of pooling sections(`pooled_size`).<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[[[  0.,   1.,   2.,   3.,   4.,   5.],<br>
  *          [  6.,   7.,   8.,   9.,  10.,  11.],<br>
  *          [ 12.,  13.,  14.,  15.,  16.,  17.],<br>
  *          [ 18.,  19.,  20.,  21.,  22.,  23.],<br>
  *          [ 24.,  25.,  26.,  27.,  28.,  29.],<br>
  *          [ 30.,  31.,  32.,  33.,  34.,  35.],<br>
  *          [ 36.,  37.,  38.,  39.,  40.,  41.],<br>
  *          [ 42.,  43.,  44.,  45.,  46.,  47.]]]]<br>
  * <br>
  *   // region of interest i.e. bounding box coordinates.<br>
  *   y = [[0,0,0,4,4]]<br>
  * <br>
  *   // returns array of shape (2,2) according to the given roi with max pooling.<br>
  *   ROIPooling(x, y, (2,2), 1.0) = [[[[ 14.,  16.],<br>
  *                                     [ 26.,  28.]]]]<br>
  * <br>
  *   // region of interest is changed due to the change in `spacial_scale` parameter.<br>
  *   ROIPooling(x, y, (2,2), 0.7) = [[[[  7.,   9.],<br>
  *                                     [ 19.,  21.]]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/roi_pooling.cc:L295<br>
  * @return org.apache.mxnet.Symbol
  */
def ROIPooling(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reshapes the input array.<br>
  * <br>
  * .. note:: ``Reshape`` is deprecated, use ``reshape``<br>
  * <br>
  * Given an array and a shape, this function returns a copy of the array in the new shape.<br>
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.<br>
  * <br>
  * Example::<br>
  * <br>
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]<br>
  * <br>
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:<br>
  * <br>
  * - ``0``  copy this dimension from the input to the output shape.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)<br>
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)<br>
  * <br>
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions<br>
  *   keeping the size of the new array same as that of the input array.<br>
  *   At most one dimension of shape can be -1.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)<br>
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)<br>
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)<br>
  * <br>
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)<br>
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)<br>
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)<br>
  * <br>
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)<br>
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)<br>
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)<br>
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)<br>
  * <br>
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)<br>
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)<br>
  * <br>
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)<br>
  *   - with reverse=1, output shape will be (50,4).<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L168<br>
  * @return org.apache.mxnet.Symbol
  */
def Reshape(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes support vector machine based transformation of the input.<br>
  * <br>
  * This tutorial demonstrates using SVM as output layer for classification instead of softmax:<br>
  * https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.<br>
  * @return org.apache.mxnet.Symbol
  */
def SVMOutput(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Takes the last element of a sequence.<br>
  * <br>
  * This function takes an n-dimensional input array of the form<br>
  * [max_sequence_length, batch_size, other_feature_dims] and returns a (n-1)-dimensional array<br>
  * of the form [batch_size, other_feature_dims].<br>
  * <br>
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length` should be<br>
  * an input array of positive ints of dimension [batch_size]. To use this parameter,<br>
  * set `use_sequence_length` to `True`, otherwise each example in the batch is assumed<br>
  * to have the max sequence length.<br>
  * <br>
  * .. note:: Alternatively, you can also use `take` operator.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[[  1.,   2.,   3.],<br>
  *          [  4.,   5.,   6.],<br>
  *          [  7.,   8.,   9.]],<br>
  * <br>
  *         [[ 10.,   11.,   12.],<br>
  *          [ 13.,   14.,   15.],<br>
  *          [ 16.,   17.,   18.]],<br>
  * <br>
  *         [[  19.,   20.,   21.],<br>
  *          [  22.,   23.,   24.],<br>
  *          [  25.,   26.,   27.]]]<br>
  * <br>
  *    // returns last sequence when sequence_length parameter is not used<br>
  *    SequenceLast(x) = [[  19.,   20.,   21.],<br>
  *                       [  22.,   23.,   24.],<br>
  *                       [  25.,   26.,   27.]]<br>
  * <br>
  *    // sequence_length is used<br>
  *    SequenceLast(x, sequence_length=[1,1,1], use_sequence_length=True) =<br>
  *             [[  1.,   2.,   3.],<br>
  *              [  4.,   5.,   6.],<br>
  *              [  7.,   8.,   9.]]<br>
  * <br>
  *    // sequence_length is used<br>
  *    SequenceLast(x, sequence_length=[1,2,3], use_sequence_length=True) =<br>
  *             [[  1.,    2.,   3.],<br>
  *              [  13.,  14.,  15.],<br>
  *              [  25.,  26.,  27.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/sequence_last.cc:L92<br>
  * @return org.apache.mxnet.Symbol
  */
def SequenceLast(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Sets all elements outside the sequence to a constant value.<br>
  * <br>
  * This function takes an n-dimensional input array of the form<br>
  * [max_sequence_length, batch_size, other_feature_dims] and returns an array of the same shape.<br>
  * <br>
  * Parameter `sequence_length` is used to handle variable-length sequences. `sequence_length`<br>
  * should be an input array of positive ints of dimension [batch_size].<br>
  * To use this parameter, set `use_sequence_length` to `True`,<br>
  * otherwise each example in the batch is assumed to have the max sequence length and<br>
  * this operator works as the `identity` operator.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[[  1.,   2.,   3.],<br>
  *          [  4.,   5.,   6.]],<br>
  * <br>
  *         [[  7.,   8.,   9.],<br>
  *          [ 10.,  11.,  12.]],<br>
  * <br>
  *         [[ 13.,  14.,   15.],<br>
  *          [ 16.,  17.,   18.]]]<br>
  * <br>
  *    // Batch 1<br>
  *    B1 = [[  1.,   2.,   3.],<br>
  *          [  7.,   8.,   9.],<br>
  *          [ 13.,  14.,  15.]]<br>
  * <br>
  *    // Batch 2<br>
  *    B2 = [[  4.,   5.,   6.],<br>
  *          [ 10.,  11.,  12.],<br>
  *          [ 16.,  17.,  18.]]<br>
  * <br>
  *    // works as identity operator when sequence_length parameter is not used<br>
  *    SequenceMask(x) = [[[  1.,   2.,   3.],<br>
  *                        [  4.,   5.,   6.]],<br>
  * <br>
  *                       [[  7.,   8.,   9.],<br>
  *                        [ 10.,  11.,  12.]],<br>
  * <br>
  *                       [[ 13.,  14.,   15.],<br>
  *                        [ 16.,  17.,   18.]]]<br>
  * <br>
  *    // sequence_length [1,1] means 1 of each batch will be kept<br>
  *    // and other rows are masked with default mask value = 0<br>
  *    SequenceMask(x, sequence_length=[1,1], use_sequence_length=True) =<br>
  *                 [[[  1.,   2.,   3.],<br>
  *                   [  4.,   5.,   6.]],<br>
  * <br>
  *                  [[  0.,   0.,   0.],<br>
  *                   [  0.,   0.,   0.]],<br>
  * <br>
  *                  [[  0.,   0.,   0.],<br>
  *                   [  0.,   0.,   0.]]]<br>
  * <br>
  *    // sequence_length [2,3] means 2 of batch B1 and 3 of batch B2 will be kept<br>
  *    // and other rows are masked with value = 1<br>
  *    SequenceMask(x, sequence_length=[2,3], use_sequence_length=True, value=1) =<br>
  *                 [[[  1.,   2.,   3.],<br>
  *                   [  4.,   5.,   6.]],<br>
  * <br>
  *                  [[  7.,   8.,   9.],<br>
  *                   [  10.,  11.,  12.]],<br>
  * <br>
  *                  [[   1.,   1.,   1.],<br>
  *                   [  16.,  17.,  18.]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/sequence_mask.cc:L114<br>
  * @return org.apache.mxnet.Symbol
  */
def SequenceMask(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reverses the elements of each sequence.<br>
  * <br>
  * This function takes an n-dimensional input array of the form [max_sequence_length, batch_size, other_feature_dims]<br>
  * and returns an array of the same shape.<br>
  * <br>
  * Parameter `sequence_length` is used to handle variable-length sequences.<br>
  * `sequence_length` should be an input array of positive ints of dimension [batch_size].<br>
  * To use this parameter, set `use_sequence_length` to `True`,<br>
  * otherwise each example in the batch is assumed to have the max sequence length.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[[  1.,   2.,   3.],<br>
  *          [  4.,   5.,   6.]],<br>
  * <br>
  *         [[  7.,   8.,   9.],<br>
  *          [ 10.,  11.,  12.]],<br>
  * <br>
  *         [[ 13.,  14.,   15.],<br>
  *          [ 16.,  17.,   18.]]]<br>
  * <br>
  *    // Batch 1<br>
  *    B1 = [[  1.,   2.,   3.],<br>
  *          [  7.,   8.,   9.],<br>
  *          [ 13.,  14.,  15.]]<br>
  * <br>
  *    // Batch 2<br>
  *    B2 = [[  4.,   5.,   6.],<br>
  *          [ 10.,  11.,  12.],<br>
  *          [ 16.,  17.,  18.]]<br>
  * <br>
  *    // returns reverse sequence when sequence_length parameter is not used<br>
  *    SequenceReverse(x) = [[[ 13.,  14.,   15.],<br>
  *                           [ 16.,  17.,   18.]],<br>
  * <br>
  *                          [[  7.,   8.,   9.],<br>
  *                           [ 10.,  11.,  12.]],<br>
  * <br>
  *                          [[  1.,   2.,   3.],<br>
  *                           [  4.,   5.,   6.]]]<br>
  * <br>
  *    // sequence_length [2,2] means 2 rows of<br>
  *    // both batch B1 and B2 will be reversed.<br>
  *    SequenceReverse(x, sequence_length=[2,2], use_sequence_length=True) =<br>
  *                      [[[  7.,   8.,   9.],<br>
  *                        [ 10.,  11.,  12.]],<br>
  * <br>
  *                       [[  1.,   2.,   3.],<br>
  *                        [  4.,   5.,   6.]],<br>
  * <br>
  *                       [[ 13.,  14.,   15.],<br>
  *                        [ 16.,  17.,   18.]]]<br>
  * <br>
  *    // sequence_length [2,3] means 2 of batch B2 and 3 of batch B3<br>
  *    // will be reversed.<br>
  *    SequenceReverse(x, sequence_length=[2,3], use_sequence_length=True) =<br>
  *                     [[[  7.,   8.,   9.],<br>
  *                       [ 16.,  17.,  18.]],<br>
  * <br>
  *                      [[  1.,   2.,   3.],<br>
  *                       [ 10.,  11.,  12.]],<br>
  * <br>
  *                      [[ 13.,  14,   15.],<br>
  *                       [  4.,   5.,   6.]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/sequence_reverse.cc:L113<br>
  * @return org.apache.mxnet.Symbol
  */
def SequenceReverse(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Splits an array along a particular axis into multiple sub-arrays.<br>
  * <br>
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.<br>
  * <br>
  * **Note** that `num_outputs` should evenly divide the length of the axis<br>
  * along which to split the array.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x  = [[[ 1.]<br>
  *           [ 2.]]<br>
  *          [[ 3.]<br>
  *           [ 4.]]<br>
  *          [[ 5.]<br>
  *           [ 6.]]]<br>
  *    x.shape = (3, 2, 1)<br>
  * <br>
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)<br>
  *    y = [[[ 1.]]<br>
  *         [[ 3.]]<br>
  *         [[ 5.]]]<br>
  * <br>
  *        [[[ 2.]]<br>
  *         [[ 4.]]<br>
  *         [[ 6.]]]<br>
  * <br>
  *    y[0].shape = (3, 1, 1)<br>
  * <br>
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)<br>
  *    z = [[[ 1.]<br>
  *          [ 2.]]]<br>
  * <br>
  *        [[[ 3.]<br>
  *          [ 4.]]]<br>
  * <br>
  *        [[[ 5.]<br>
  *          [ 6.]]]<br>
  * <br>
  *    z[0].shape = (1, 2, 1)<br>
  * <br>
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.<br>
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only<br>
  * along the `axis` which it is split.<br>
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.<br>
  * <br>
  * Example::<br>
  * <br>
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)<br>
  *    z = [[ 1.]<br>
  *         [ 2.]]<br>
  * <br>
  *        [[ 3.]<br>
  *         [ 4.]]<br>
  * <br>
  *        [[ 5.]<br>
  *         [ 6.]]<br>
  *    z[0].shape = (2 ,1 )<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/slice_channel.cc:L107<br>
  * @return org.apache.mxnet.Symbol
  */
def SliceChannel(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Please use `SoftmaxOutput`.<br>
  * <br>
  * .. note::<br>
  * <br>
  *   This operator has been renamed to `SoftmaxOutput`, which<br>
  *   computes the gradient of cross-entropy loss w.r.t softmax output.<br>
  *   To just compute softmax output, use the `softmax` operator.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/softmax_output.cc:L138<br>
  * @return org.apache.mxnet.Symbol
  */
def Softmax(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies softmax activation to input. This is intended for internal layers.<br>
  * <br>
  * .. note::<br>
  * <br>
  *   This operator has been deprecated, please use `softmax`.<br>
  * <br>
  * If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.<br>
  * This is the default mode.<br>
  * <br>
  * If `mode` = ``channel``, this operator will compute a k-class softmax at each position<br>
  * of each instance, where `k` = ``num_channel``. This mode can only be used when the input array<br>
  * has at least 3 dimensions.<br>
  * This can be used for `fully convolutional network`, `image segmentation`, etc.<br>
  * <br>
  * Example::<br>
  * <br>
  *   >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],<br>
  *   >>>                            [2., -.4, 7.,   3., 0.2]])<br>
  *   >>> softmax_act = mx.nd.SoftmaxActivation(input_array)<br>
  *   >>> print softmax_act.asnumpy()<br>
  *   [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]<br>
  *    [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/softmax_activation.cc:L59<br>
  * @return org.apache.mxnet.Symbol
  */
def SoftmaxActivation(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the gradient of cross entropy loss with respect to softmax output.<br>
  * <br>
  * - This operator computes the gradient in two steps.<br>
  *   The cross entropy loss does not actually need to be computed.<br>
  * <br>
  *   - Applies softmax function on the input array.<br>
  *   - Computes and returns the gradient of cross entropy loss w.r.t. the softmax output.<br>
  * <br>
  * - The softmax function, cross entropy loss and gradient is given by:<br>
  * <br>
  *   - Softmax Function:<br>
  * <br>
  *     .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}<br>
  * <br>
  *   - Cross Entropy Function:<br>
  * <br>
  *     .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)<br>
  * <br>
  *   - The gradient of cross entropy loss w.r.t softmax output:<br>
  * <br>
  *     .. math:: \text{gradient} = \text{output} - \text{label}<br>
  * <br>
  * - During forward propagation, the softmax function is computed for each instance in the input array.<br>
  * <br>
  *   For general *N*-D input arrays with shape :math:`(d_1, d_2, ..., d_n)`. The size is<br>
  *   :math:`s=d_1 \cdot d_2 \cdot \cdot \cdot d_n`. We can use the parameters `preserve_shape`<br>
  *   and `multi_output` to specify the way to compute softmax:<br>
  * <br>
  *   - By default, `preserve_shape` is ``false``. This operator will reshape the input array<br>
  *     into a 2-D array with shape :math:`(d_1, \frac{s}{d_1})` and then compute the softmax function for<br>
  *     each row in the reshaped array, and afterwards reshape it back to the original shape<br>
  *     :math:`(d_1, d_2, ..., d_n)`.<br>
  *   - If `preserve_shape` is ``true``, the softmax function will be computed along<br>
  *     the last axis (`axis` = ``-1``).<br>
  *   - If `multi_output` is ``true``, the softmax function will be computed along<br>
  *     the second axis (`axis` = ``1``).<br>
  * <br>
  * - During backward propagation, the gradient of cross-entropy loss w.r.t softmax output array is computed.<br>
  *   The provided label can be a one-hot label array or a probability label array.<br>
  * <br>
  *   - If the parameter `use_ignore` is ``true``, `ignore_label` can specify input instances<br>
  *     with a particular label to be ignored during backward propagation. **This has no effect when<br>
  *     softmax `output` has same shape as `label`**.<br>
  * <br>
  *     Example::<br>
  * <br>
  *       data = [[1,2,3,4],[2,2,2,2],[3,3,3,3],[4,4,4,4]]<br>
  *       label = [1,0,2,3]<br>
  *       ignore_label = 1<br>
  *       SoftmaxOutput(data=data, label = label,\<br>
  *                     multi_output=true, use_ignore=true,\<br>
  *                     ignore_label=ignore_label)<br>
  *       ## forward softmax output<br>
  *       [[ 0.0320586   0.08714432  0.23688284  0.64391428]<br>
  *        [ 0.25        0.25        0.25        0.25      ]<br>
  *        [ 0.25        0.25        0.25        0.25      ]<br>
  *        [ 0.25        0.25        0.25        0.25      ]]<br>
  *       ## backward gradient output<br>
  *       [[ 0.    0.    0.    0.  ]<br>
  *        [-0.75  0.25  0.25  0.25]<br>
  *        [ 0.25  0.25 -0.75  0.25]<br>
  *        [ 0.25  0.25  0.25 -0.75]]<br>
  *       ## notice that the first row is all 0 because label[0] is 1, which is equal to ignore_label.<br>
  * <br>
  *   - The parameter `grad_scale` can be used to rescale the gradient, which is often used to<br>
  *     give each loss function different weights.<br>
  * <br>
  *   - This operator also supports various ways to normalize the gradient by `normalization`,<br>
  *     The `normalization` is applied if softmax output has different shape than the labels.<br>
  *     The `normalization` mode can be set to the followings:<br>
  * <br>
  *     - ``'null'``: do nothing.<br>
  *     - ``'batch'``: divide the gradient by the batch size.<br>
  *     - ``'valid'``: divide the gradient by the number of instances which are not ignored.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/softmax_output.cc:L123<br>
  * @return org.apache.mxnet.Symbol
  */
def SoftmaxOutput(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies a spatial transformer to input feature map.<br>
  * @return org.apache.mxnet.Symbol
  */
def SpatialTransformer(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Interchanges two axes of an array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[1, 2, 3]])<br>
  *   swapaxes(x, 0, 1) = [[ 1],<br>
  *                        [ 2],<br>
  *                        [ 3]]<br>
  * <br>
  *   x = [[[ 0, 1],<br>
  *         [ 2, 3]],<br>
  *        [[ 4, 5],<br>
  *         [ 6, 7]]]  // (2,2,2) array<br>
  * <br>
  *  swapaxes(x, 0, 2) = [[[ 0, 4],<br>
  *                        [ 2, 6]],<br>
  *                       [[ 1, 5],<br>
  *                        [ 3, 7]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/swapaxis.cc:L70<br>
  * @return org.apache.mxnet.Symbol
  */
def SwapAxis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs nearest neighbor/bilinear up sampling to inputs.<br>
  * @return org.apache.mxnet.Symbol
  */
def UpSampling(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise absolute value of the input.<br>
  * <br>
  * Example::<br>
  * <br>
  *    abs([-2, 0, 3]) = [2, 0, 3]<br>
  * <br>
  * The storage type of ``abs`` output depends upon the input storage type:<br>
  * <br>
  *    - abs(default) = default<br>
  *    - abs(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L490<br>
  * @return org.apache.mxnet.Symbol
  */
def abs(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for Adam optimizer. Adam is seen as a generalization<br>
  * of AdaGrad.<br>
  * <br>
  * Adam update consists of the following steps, where g represents gradient and m, v<br>
  * are 1st and 2nd order moment estimates (mean and variance).<br>
  * <br>
  * .. math::<br>
  * <br>
  *  g_t = \nabla J(W_{t-1})\\<br>
  *  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t\\<br>
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\<br>
  *  W_t = W_{t-1} - \alpha \frac{ m_t }{ \sqrt{ v_t } + \epsilon }<br>
  * <br>
  * It updates the weights using::<br>
  * <br>
  *  m = beta1*m + (1-beta1)*grad<br>
  *  v = beta2*v + (1-beta2)*(grad**2)<br>
  *  w += - learning_rate * m / (sqrt(v) + epsilon)<br>
  * <br>
  * However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and the storage<br>
  * type of weight is the same as those of m and v,<br>
  * only the row slices whose indices appear in grad.indices are updated (for w, m and v)::<br>
  * <br>
  *  for row in grad.indices:<br>
  *      m[row] = beta1*m[row] + (1-beta1)*grad[row]<br>
  *      v[row] = beta2*v[row] + (1-beta2)*(grad[row]**2)<br>
  *      w[row] += - learning_rate * m[row] / (sqrt(v[row]) + epsilon)<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L495<br>
  * @return org.apache.mxnet.Symbol
  */
def adam_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Adds all input arguments element-wise.<br>
  * <br>
  * .. math::<br>
  *    add\_n(a_1, a_2, ..., a_n) = a_1 + a_2 + ... + a_n<br>
  * <br>
  * ``add_n`` is potentially more efficient than calling ``add`` by `n` times.<br>
  * <br>
  * The storage type of ``add_n`` output depends on storage types of inputs<br>
  * <br>
  * - add_n(row_sparse, row_sparse, ..) = row_sparse<br>
  * - otherwise, ``add_n`` generates output with default storage<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_sum.cc:L150<br>
  * @return org.apache.mxnet.Symbol
  */
def add_n(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise inverse cosine of the input array.<br>
  * <br>
  * The input should be in range `[-1, 1]`.<br>
  * The output is in the closed interval :math:`[0, \pi]`<br>
  * <br>
  * .. math::<br>
  *    arccos([-1, -.707, 0, .707, 1]) = [\pi, 3\pi/4, \pi/2, \pi/4, 0]<br>
  * <br>
  * The storage type of ``arccos`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L123<br>
  * @return org.apache.mxnet.Symbol
  */
def arccos(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the element-wise inverse hyperbolic cosine of the input array, \<br>
  * computed element-wise.<br>
  * <br>
  * The storage type of ``arccosh`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L264<br>
  * @return org.apache.mxnet.Symbol
  */
def arccosh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise inverse sine of the input array.<br>
  * <br>
  * The input should be in the range `[-1, 1]`.<br>
  * The output is in the closed interval of [:math:`-\pi/2`, :math:`\pi/2`].<br>
  * <br>
  * .. math::<br>
  *    arcsin([-1, -.707, 0, .707, 1]) = [-\pi/2, -\pi/4, 0, \pi/4, \pi/2]<br>
  * <br>
  * The storage type of ``arcsin`` output depends upon the input storage type:<br>
  * <br>
  *    - arcsin(default) = default<br>
  *    - arcsin(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L104<br>
  * @return org.apache.mxnet.Symbol
  */
def arcsin(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the element-wise inverse hyperbolic sine of the input array, \<br>
  * computed element-wise.<br>
  * <br>
  * The storage type of ``arcsinh`` output depends upon the input storage type:<br>
  * <br>
  *    - arcsinh(default) = default<br>
  *    - arcsinh(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L250<br>
  * @return org.apache.mxnet.Symbol
  */
def arcsinh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise inverse tangent of the input array.<br>
  * <br>
  * The output is in the closed interval :math:`[-\pi/2, \pi/2]`<br>
  * <br>
  * .. math::<br>
  *    arctan([-1, 0, 1]) = [-\pi/4, 0, \pi/4]<br>
  * <br>
  * The storage type of ``arctan`` output depends upon the input storage type:<br>
  * <br>
  *    - arctan(default) = default<br>
  *    - arctan(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L144<br>
  * @return org.apache.mxnet.Symbol
  */
def arctan(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the element-wise inverse hyperbolic tangent of the input array, \<br>
  * computed element-wise.<br>
  * <br>
  * The storage type of ``arctanh`` output depends upon the input storage type:<br>
  * <br>
  *    - arctanh(default) = default<br>
  *    - arctanh(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L281<br>
  * @return org.apache.mxnet.Symbol
  */
def arctanh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns indices of the maximum values along an axis.<br>
  * <br>
  * In the case of multiple occurrences of maximum values, the indices corresponding to the first occurrence<br>
  * are returned.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  1.,  2.],<br>
  *        [ 3.,  4.,  5.]]<br>
  * <br>
  *   // argmax along axis 0<br>
  *   argmax(x, axis=0) = [ 1.,  1.,  1.]<br>
  * <br>
  *   // argmax along axis 1<br>
  *   argmax(x, axis=1) = [ 2.,  2.]<br>
  * <br>
  *   // argmax along axis 1 keeping same dims as an input array<br>
  *   argmax(x, axis=1, keepdims=True) = [[ 2.],<br>
  *                                       [ 2.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L52<br>
  * @return org.apache.mxnet.Symbol
  */
def argmax(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns argmax indices of each channel from the input array.<br>
  * <br>
  * The result will be an NDArray of shape (num_channel,).<br>
  * <br>
  * In case of multiple occurrences of the maximum values, the indices corresponding to the first occurrence<br>
  * are returned.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  1.,  2.],<br>
  *        [ 3.,  4.,  5.]]<br>
  * <br>
  *   argmax_channel(x) = [ 2.,  2.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L97<br>
  * @return org.apache.mxnet.Symbol
  */
def argmax_channel(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns indices of the minimum values along an axis.<br>
  * <br>
  * In the case of multiple occurrences of minimum values, the indices corresponding to the first occurrence<br>
  * are returned.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  1.,  2.],<br>
  *        [ 3.,  4.,  5.]]<br>
  * <br>
  *   // argmin along axis 0<br>
  *   argmin(x, axis=0) = [ 0.,  0.,  0.]<br>
  * <br>
  *   // argmin along axis 1<br>
  *   argmin(x, axis=1) = [ 0.,  0.]<br>
  * <br>
  *   // argmin along axis 1 keeping same dims as an input array<br>
  *   argmin(x, axis=1, keepdims=True) = [[ 0.],<br>
  *                                       [ 0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L77<br>
  * @return org.apache.mxnet.Symbol
  */
def argmin(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the indices that would sort an input array along the given axis.<br>
  * <br>
  * This function performs sorting along the given axis and returns an array of indices having same shape<br>
  * as an input array that index data in sorted order.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.3,  0.2,  0.4],<br>
  *        [ 0.1,  0.3,  0.2]]<br>
  * <br>
  *   // sort along axis -1<br>
  *   argsort(x) = [[ 1.,  0.,  2.],<br>
  *                 [ 0.,  2.,  1.]]<br>
  * <br>
  *   // sort along axis 0<br>
  *   argsort(x, axis=0) = [[ 1.,  0.,  1.]<br>
  *                         [ 0.,  1.,  0.]]<br>
  * <br>
  *   // flatten and then sort<br>
  *   argsort(x) = [ 3.,  1.,  5.,  0.,  4.,  2.]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/ordering_op.cc:L176<br>
  * @return org.apache.mxnet.Symbol
  */
def argsort(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Batchwise dot product.<br>
  * <br>
  * ``batch_dot`` is used to compute dot product of ``x`` and ``y`` when ``x`` and<br>
  * ``y`` are data in batch, namely 3D arrays in shape of `(batch_size, :, :)`.<br>
  * <br>
  * For example, given ``x`` with shape `(batch_size, n, m)` and ``y`` with shape<br>
  * `(batch_size, m, k)`, the result array will have shape `(batch_size, n, k)`,<br>
  * which is computed by::<br>
  * <br>
  *    batch_dot(x,y)[i,:,:] = dot(x[i,:,:], y[i,:,:])<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/dot.cc:L117<br>
  * @return org.apache.mxnet.Symbol
  */
def batch_dot(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Takes elements from a data batch.<br>
  * <br>
  * .. note::<br>
  *   `batch_take` is deprecated. Use `pick` instead.<br>
  * <br>
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be<br>
  * an output array of shape ``(i0,)`` with::<br>
  * <br>
  *   output[i] = input[i, indices[i]]<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 1.,  2.],<br>
  *        [ 3.,  4.],<br>
  *        [ 5.,  6.]]<br>
  * <br>
  *   // takes elements with specified indices<br>
  *   batch_take(x, [0,1,0]) = [ 1.  4.  5.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/indexing_op.cc:L444<br>
  * @return org.apache.mxnet.Symbol
  */
def batch_take(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise sum of the input arrays with broadcasting.<br>
  * <br>
  * `broadcast_plus` is an alias to the function `broadcast_add`.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],<br>
  *                           [ 2.,  2.,  2.]]<br>
  * <br>
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],<br>
  *                            [ 2.,  2.,  2.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_add(csr, dense(1D)) = dense<br>
  *    broadcast_add(dense(1D), csr) = dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_add(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Broadcasts the input array over particular axes.<br>
  * <br>
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to<br>
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.<br>
  * <br>
  * Example::<br>
  * <br>
  *    // given x of shape (1,2,1)<br>
  *    x = [[[ 1.],<br>
  *          [ 2.]]]<br>
  * <br>
  *    // broadcast x on on axis 2<br>
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],<br>
  *                                          [ 2.,  2.,  2.]]]<br>
  *    // broadcast x on on axes 0 and 2<br>
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],<br>
  *                                                  [ 2.,  2.,  2.]],<br>
  *                                                 [[ 1.,  1.,  1.],<br>
  *                                                  [ 2.,  2.,  2.]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L237<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_axes(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Broadcasts the input array over particular axes.<br>
  * <br>
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to<br>
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.<br>
  * <br>
  * Example::<br>
  * <br>
  *    // given x of shape (1,2,1)<br>
  *    x = [[[ 1.],<br>
  *          [ 2.]]]<br>
  * <br>
  *    // broadcast x on on axis 2<br>
  *    broadcast_axis(x, axis=2, size=3) = [[[ 1.,  1.,  1.],<br>
  *                                          [ 2.,  2.,  2.]]]<br>
  *    // broadcast x on on axes 0 and 2<br>
  *    broadcast_axis(x, axis=(0,2), size=(2,3)) = [[[ 1.,  1.,  1.],<br>
  *                                                  [ 2.,  2.,  2.]],<br>
  *                                                 [[ 1.,  1.,  1.],<br>
  *                                                  [ 2.,  2.,  2.]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L237<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_axis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise division of the input arrays with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 6.,  6.,  6.],<br>
  *         [ 6.,  6.,  6.]]<br>
  * <br>
  *    y = [[ 2.],<br>
  *         [ 3.]]<br>
  * <br>
  *    broadcast_div(x, y) = [[ 3.,  3.,  3.],<br>
  *                           [ 2.,  2.,  2.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_div(csr, dense(1D)) = csr<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L187<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_div(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **equal to** (==) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_equal(x, y) = [[ 0.,  0.,  0.],<br>
  *                             [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L46<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_equal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **greater than** (>) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_greater(x, y) = [[ 1.,  1.,  1.],<br>
  *                               [ 0.,  0.,  0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L82<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_greater(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **greater than or equal to** (>=) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_greater_equal(x, y) = [[ 1.,  1.,  1.],<br>
  *                                     [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L100<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_greater_equal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  *  Returns the hypotenuse of a right angled triangle, given its "legs"<br>
  * with broadcasting.<br>
  * <br>
  * It is equivalent to doing :math:`sqrt(x_1^2 + x_2^2)`.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 3.,  3.,  3.]]<br>
  * <br>
  *    y = [[ 4.],<br>
  *         [ 4.]]<br>
  * <br>
  *    broadcast_hypot(x, y) = [[ 5.,  5.,  5.],<br>
  *                             [ 5.,  5.,  5.]]<br>
  * <br>
  *    z = [[ 0.],<br>
  *         [ 4.]]<br>
  * <br>
  *    broadcast_hypot(x, z) = [[ 3.,  3.,  3.],<br>
  *                             [ 5.,  5.,  5.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L156<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_hypot(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **lesser than** (<) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_lesser(x, y) = [[ 0.,  0.,  0.],<br>
  *                              [ 0.,  0.,  0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L118<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_lesser(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **lesser than or equal to** (<=) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_lesser_equal(x, y) = [[ 0.,  0.,  0.],<br>
  *                                    [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L136<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_lesser_equal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **logical and** with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_logical_and(x, y) = [[ 0.,  0.,  0.],<br>
  *                                   [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L154<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_logical_and(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **logical or** with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  0.],<br>
  *         [ 1.,  1.,  0.]]<br>
  * <br>
  *    y = [[ 1.],<br>
  *         [ 0.]]<br>
  * <br>
  *    broadcast_logical_or(x, y) = [[ 1.,  1.,  1.],<br>
  *                                  [ 1.,  1.,  0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L172<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_logical_or(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **logical xor** with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  0.],<br>
  *         [ 1.,  1.,  0.]]<br>
  * <br>
  *    y = [[ 1.],<br>
  *         [ 0.]]<br>
  * <br>
  *    broadcast_logical_xor(x, y) = [[ 0.,  0.,  1.],<br>
  *                                   [ 1.,  1.,  0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L190<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_logical_xor(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise maximum of the input arrays with broadcasting.<br>
  * <br>
  * This function compares two input arrays and returns a new array having the element-wise maxima.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_maximum(x, y) = [[ 1.,  1.,  1.],<br>
  *                               [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L80<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_maximum(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise minimum of the input arrays with broadcasting.<br>
  * <br>
  * This function compares two input arrays and returns a new array having the element-wise minima.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_maximum(x, y) = [[ 0.,  0.,  0.],<br>
  *                               [ 1.,  1.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L115<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_minimum(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise difference of the input arrays with broadcasting.<br>
  * <br>
  * `broadcast_minus` is an alias to the function `broadcast_sub`.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],<br>
  *                           [ 0.,  0.,  0.]]<br>
  * <br>
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],<br>
  *                             [ 0.,  0.,  0.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_sub/minus(csr, dense(1D)) = dense<br>
  *    broadcast_sub/minus(dense(1D), csr) = dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_minus(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise modulo of the input arrays with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 8.,  8.,  8.],<br>
  *         [ 8.,  8.,  8.]]<br>
  * <br>
  *    y = [[ 2.],<br>
  *         [ 3.]]<br>
  * <br>
  *    broadcast_mod(x, y) = [[ 0.,  0.,  0.],<br>
  *                           [ 2.,  2.,  2.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L222<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_mod(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise product of the input arrays with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_mul(x, y) = [[ 0.,  0.,  0.],<br>
  *                           [ 1.,  1.,  1.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_mul(csr, dense(1D)) = csr<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L146<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_mul(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of element-wise **not equal to** (!=) comparison operation with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_not_equal(x, y) = [[ 1.,  1.,  1.],<br>
  *                                 [ 0.,  0.,  0.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_logic.cc:L64<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_not_equal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise sum of the input arrays with broadcasting.<br>
  * <br>
  * `broadcast_plus` is an alias to the function `broadcast_add`.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_add(x, y) = [[ 1.,  1.,  1.],<br>
  *                           [ 2.,  2.,  2.]]<br>
  * <br>
  *    broadcast_plus(x, y) = [[ 1.,  1.,  1.],<br>
  *                            [ 2.,  2.,  2.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_add(csr, dense(1D)) = dense<br>
  *    broadcast_add(dense(1D), csr) = dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L58<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_plus(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns result of first array elements raised to powers from second array, element-wise with broadcasting.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_power(x, y) = [[ 2.,  2.,  2.],<br>
  *                             [ 4.,  4.,  4.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_extended.cc:L45<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_power(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise difference of the input arrays with broadcasting.<br>
  * <br>
  * `broadcast_minus` is an alias to the function `broadcast_sub`.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[ 1.,  1.,  1.],<br>
  *         [ 1.,  1.,  1.]]<br>
  * <br>
  *    y = [[ 0.],<br>
  *         [ 1.]]<br>
  * <br>
  *    broadcast_sub(x, y) = [[ 1.,  1.,  1.],<br>
  *                           [ 0.,  0.,  0.]]<br>
  * <br>
  *    broadcast_minus(x, y) = [[ 1.,  1.,  1.],<br>
  *                             [ 0.,  0.,  0.]]<br>
  * <br>
  * Supported sparse operations:<br>
  * <br>
  *    broadcast_sub/minus(csr, dense(1D)) = dense<br>
  *    broadcast_sub/minus(dense(1D), csr) = dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_broadcast_op_basic.cc:L106<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_sub(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Broadcasts the input array to a new shape.<br>
  * <br>
  * Broadcasting is a mechanism that allows NDArrays to perform arithmetic operations<br>
  * with arrays of different shapes efficiently without creating multiple copies of arrays.<br>
  * Also see, `Broadcasting <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_ for more explanation.<br>
  * <br>
  * Broadcasting is allowed on axes with size 1, such as from `(2,1,3,1)` to<br>
  * `(2,8,3,9)`. Elements will be duplicated on the broadcasted axes.<br>
  * <br>
  * For example::<br>
  * <br>
  *    broadcast_to([[1,2,3]], shape=(2,3)) = [[ 1.,  2.,  3.],<br>
  *                                            [ 1.,  2.,  3.]])<br>
  * <br>
  * The dimension which you do not want to change can also be kept as `0` which means copy the original value.<br>
  * So with `shape=(2,0)`, we will obtain the same result as in the above example.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L261<br>
  * @return org.apache.mxnet.Symbol
  */
def broadcast_to(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Casts all elements of the input to a new type.<br>
  * <br>
  * .. note:: ``Cast`` is deprecated. Use ``cast`` instead.<br>
  * <br>
  * Example::<br>
  * <br>
  *    cast([0.9, 1.3], dtype='int32') = [0, 1]<br>
  *    cast([1e20, 11.1], dtype='float16') = [inf, 11.09375]<br>
  *    cast([300, 11.1, 10.9, -1, -3], dtype='uint8') = [44, 11, 10, 255, 253]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L415<br>
  * @return org.apache.mxnet.Symbol
  */
def cast(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Casts tensor storage type to the new type.<br>
  * <br>
  * When an NDArray with default storage type is cast to csr or row_sparse storage,<br>
  * the result is compact, which means:<br>
  * <br>
  * - for csr, zero values will not be retained<br>
  * - for row_sparse, row slices of all zeros will not be retained<br>
  * <br>
  * The storage type of ``cast_storage`` output depends on stype parameter:<br>
  * <br>
  * - cast_storage(csr, 'default') = default<br>
  * - cast_storage(row_sparse, 'default') = default<br>
  * - cast_storage(default, 'csr') = csr<br>
  * - cast_storage(default, 'row_sparse') = row_sparse<br>
  * - cast_storage(csr, 'csr') = csr<br>
  * - cast_storage(row_sparse, 'row_sparse') = row_sparse<br>
  * <br>
  * Example::<br>
  * <br>
  *     dense = [[ 0.,  1.,  0.],<br>
  *              [ 2.,  0.,  3.],<br>
  *              [ 0.,  0.,  0.],<br>
  *              [ 0.,  0.,  0.]]<br>
  * <br>
  *     # cast to row_sparse storage type<br>
  *     rsp = cast_storage(dense, 'row_sparse')<br>
  *     rsp.indices = [0, 1]<br>
  *     rsp.values = [[ 0.,  1.,  0.],<br>
  *                   [ 2.,  0.,  3.]]<br>
  * <br>
  *     # cast to csr storage type<br>
  *     csr = cast_storage(dense, 'csr')<br>
  *     csr.indices = [1, 0, 2]<br>
  *     csr.values = [ 1.,  2.,  3.]<br>
  *     csr.indptr = [0, 1, 3, 3, 3]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/cast_storage.cc:L71<br>
  * @return org.apache.mxnet.Symbol
  */
def cast_storage(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise cube-root value of the input.<br>
  * <br>
  * .. math::<br>
  *    cbrt(x) = \sqrt[3]{x}<br>
  * <br>
  * Example::<br>
  * <br>
  *    cbrt([1, 8, -125]) = [1, 2, -5]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L706<br>
  * @return org.apache.mxnet.Symbol
  */
def cbrt(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise ceiling of the input.<br>
  * <br>
  * The ceil of the scalar x is the smallest integer i, such that i >= x.<br>
  * <br>
  * Example::<br>
  * <br>
  *    ceil([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  2.,  2.,  3.]<br>
  * <br>
  * The storage type of ``ceil`` output depends upon the input storage type:<br>
  * <br>
  *    - ceil(default) = default<br>
  *    - ceil(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L568<br>
  * @return org.apache.mxnet.Symbol
  */
def ceil(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Choose one element from each line(row for python, column for R/Julia) in lhs according to index indicated by rhs. This function assume rhs uses 0-based index.<br>
  * @return org.apache.mxnet.Symbol
  */
def choose_element_0index(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Clips (limits) the values in an array.<br>
  * <br>
  * Given an interval, values outside the interval are clipped to the interval edges.<br>
  * Clipping ``x`` between `a_min` and `a_x` would be::<br>
  * <br>
  *    clip(x, a_min, a_max) = max(min(x, a_max), a_min))<br>
  * <br>
  * Example::<br>
  * <br>
  *     x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]<br>
  * <br>
  *     clip(x,1,8) = [ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  8.]<br>
  * <br>
  * The storage type of ``clip`` output depends on storage types of inputs and the a_min, a_max \<br>
  * parameter values:<br>
  * <br>
  *    - clip(default) = default<br>
  *    - clip(row_sparse, a_min <= 0, a_max >= 0) = row_sparse<br>
  *    - clip(csr, a_min <= 0, a_max >= 0) = csr<br>
  *    - clip(row_sparse, a_min < 0, a_max < 0) = default<br>
  *    - clip(row_sparse, a_min > 0, a_max > 0) = default<br>
  *    - clip(csr, a_min < 0, a_max < 0) = csr<br>
  *    - clip(csr, a_min > 0, a_max > 0) = csr<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L617<br>
  * @return org.apache.mxnet.Symbol
  */
def clip(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Joins input arrays along a given axis.<br>
  * <br>
  * .. note:: `Concat` is deprecated. Use `concat` instead.<br>
  * <br>
  * The dimensions of the input arrays should be the same except the axis along<br>
  * which they will be concatenated.<br>
  * The dimension of the output array along the concatenated axis will be equal<br>
  * to the sum of the corresponding dimensions of the input arrays.<br>
  * <br>
  * The storage type of ``concat`` output depends on storage types of inputs<br>
  * <br>
  * - concat(csr, csr, ..., csr, dim=0) = csr<br>
  * - otherwise, ``concat`` generates output with default storage<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[1,1],[2,2]]<br>
  *    y = [[3,3],[4,4],[5,5]]<br>
  *    z = [[6,6], [7,7],[8,8]]<br>
  * <br>
  *    concat(x,y,z,dim=0) = [[ 1.,  1.],<br>
  *                           [ 2.,  2.],<br>
  *                           [ 3.,  3.],<br>
  *                           [ 4.,  4.],<br>
  *                           [ 5.,  5.],<br>
  *                           [ 6.,  6.],<br>
  *                           [ 7.,  7.],<br>
  *                           [ 8.,  8.]]<br>
  * <br>
  *    Note that you cannot concat x,y,z along dimension 1 since dimension<br>
  *    0 is not the same for all the input arrays.<br>
  * <br>
  *    concat(y,z,dim=1) = [[ 3.,  3.,  6.,  6.],<br>
  *                          [ 4.,  4.,  7.,  7.],<br>
  *                          [ 5.,  5.,  8.,  8.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/concat.cc:L260<br>
  * @return org.apache.mxnet.Symbol
  */
def concat(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the element-wise cosine of the input array.<br>
  * <br>
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).<br>
  * <br>
  * .. math::<br>
  *    cos([0, \pi/4, \pi/2]) = [1, 0.707, 0]<br>
  * <br>
  * The storage type of ``cos`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L63<br>
  * @return org.apache.mxnet.Symbol
  */
def cos(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the hyperbolic cosine  of the input array, computed element-wise.<br>
  * <br>
  * .. math::<br>
  *    cosh(x) = 0.5\times(exp(x) + exp(-x))<br>
  * <br>
  * The storage type of ``cosh`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L216<br>
  * @return org.apache.mxnet.Symbol
  */
def cosh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Slices a region of the array.<br>
  * <br>
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.<br>
  * <br>
  * This function returns a sliced array between the indices given<br>
  * by `begin` and `end` with the corresponding `step`.<br>
  * <br>
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,<br>
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,<br>
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,<br>
  * where m <= n, results in an array with the shape<br>
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.<br>
  * <br>
  * The resulting array's *k*-th dimension contains elements<br>
  * from the *k*-th dimension of the input array starting<br>
  * from index ``b_k`` (inclusive) with step ``s_k``<br>
  * until reaching ``e_k`` (exclusive).<br>
  * <br>
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,<br>
  * and `step`, the following rule will be used to set default values.<br>
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;<br>
  * else, set `b_k=d_k-1`, `e_k=-1`.<br>
  * <br>
  * The storage type of ``slice`` output depends on storage types of inputs<br>
  * <br>
  * - slice(csr) = csr<br>
  * - otherwise, ``slice`` generates output with default storage<br>
  * <br>
  * .. note:: When input data storage type is csr, it only supports<br>
  * step=(), or step=(None,), or step=(1,) to generate a csr output.<br>
  * For other step parameter values, it falls back to slicing<br>
  * a dense tensor.<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[  1.,   2.,   3.,   4.],<br>
  *        [  5.,   6.,   7.,   8.],<br>
  *        [  9.,  10.,  11.,  12.]]<br>
  * <br>
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],<br>
  *                                      [ 6.,  7.,  8.]]<br>
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],<br>
  *                                                             [5.,  7.],<br>
  *                                                             [1.,  3.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L412<br>
  * @return org.apache.mxnet.Symbol
  */
def crop(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Converts each element of the input array from radians to degrees.<br>
  * <br>
  * .. math::<br>
  *    degrees([0, \pi/2, \pi, 3\pi/2, 2\pi]) = [0, 90, 180, 270, 360]<br>
  * <br>
  * The storage type of ``degrees`` output depends upon the input storage type:<br>
  * <br>
  *    - degrees(default) = default<br>
  *    - degrees(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L163<br>
  * @return org.apache.mxnet.Symbol
  */
def degrees(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Dot product of two arrays.<br>
  * <br>
  * ``dot``'s behavior depends on the input array dimensions:<br>
  * <br>
  * - 1-D arrays: inner product of vectors<br>
  * - 2-D arrays: matrix multiplication<br>
  * - N-D arrays: a sum product over the last axis of the first input and the first<br>
  *   axis of the second input<br>
  * <br>
  *   For example, given 3-D ``x`` with shape `(n,m,k)` and ``y`` with shape `(k,r,s)`, the<br>
  *   result array will have shape `(n,m,r,s)`. It is computed by::<br>
  * <br>
  *     dot(x,y)[i,j,a,b] = sum(x[i,j,:]*y[:,a,b])<br>
  * <br>
  *   Example::<br>
  * <br>
  *     x = reshape([0,1,2,3,4,5,6,7], shape=(2,2,2))<br>
  *     y = reshape([7,6,5,4,3,2,1,0], shape=(2,2,2))<br>
  *     dot(x,y)[0,0,1,1] = 0<br>
  *     sum(x[0,0,:]*y[:,1,1]) = 0<br>
  * <br>
  * The storage type of ``dot`` output depends on storage types of inputs, transpose option and<br>
  * forward_stype option for output storage type. Implemented sparse operations include:<br>
  * <br>
  * - dot(default, default, transpose_a=True/False, transpose_b=True/False) = default<br>
  * - dot(csr, default, transpose_a=True) = default<br>
  * - dot(csr, default, transpose_a=True) = row_sparse<br>
  * - dot(csr, default) = default<br>
  * - dot(csr, row_sparse) = default<br>
  * - dot(default, csr) = csr (CPU only)<br>
  * - dot(default, csr, forward_stype='default') = default<br>
  * - dot(default, csr, transpose_b=True, forward_stype='default') = default<br>
  * <br>
  * If the combination of input storage types and forward_stype does not match any of the<br>
  * above patterns, ``dot`` will fallback and generate output with default storage.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/dot.cc:L69<br>
  * @return org.apache.mxnet.Symbol
  */
def dot(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Adds arguments element-wise.<br>
  * <br>
  * The storage type of ``elemwise_add`` output depends on storage types of inputs<br>
  * <br>
  *    - elemwise_add(row_sparse, row_sparse) = row_sparse<br>
  *    - elemwise_add(csr, csr) = csr<br>
  *    - elemwise_add(default, csr) = default<br>
  *    - elemwise_add(csr, default) = default<br>
  *    - elemwise_add(default, rsp) = default<br>
  *    - elemwise_add(rsp, default) = default<br>
  *    - otherwise, ``elemwise_add`` generates output with default storage<br>
  * @return org.apache.mxnet.Symbol
  */
def elemwise_add(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Divides arguments element-wise.<br>
  * <br>
  * The storage type of ``elemwise_div`` output is always dense<br>
  * @return org.apache.mxnet.Symbol
  */
def elemwise_div(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Multiplies arguments element-wise.<br>
  * <br>
  * The storage type of ``elemwise_mul`` output depends on storage types of inputs<br>
  * <br>
  *    - elemwise_mul(default, default) = default<br>
  *    - elemwise_mul(row_sparse, row_sparse) = row_sparse<br>
  *    - elemwise_mul(default, row_sparse) = row_sparse<br>
  *    - elemwise_mul(row_sparse, default) = row_sparse<br>
  *    - elemwise_mul(csr, csr) = csr<br>
  *    - otherwise, ``elemwise_mul`` generates output with default storage<br>
  * @return org.apache.mxnet.Symbol
  */
def elemwise_mul(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Subtracts arguments element-wise.<br>
  * <br>
  * The storage type of ``elemwise_sub`` output depends on storage types of inputs<br>
  * <br>
  *    - elemwise_sub(row_sparse, row_sparse) = row_sparse<br>
  *    - elemwise_sub(csr, csr) = csr<br>
  *    - elemwise_sub(default, csr) = default<br>
  *    - elemwise_sub(csr, default) = default<br>
  *    - elemwise_sub(default, rsp) = default<br>
  *    - elemwise_sub(rsp, default) = default<br>
  *    - otherwise, ``elemwise_sub`` generates output with default storage<br>
  * @return org.apache.mxnet.Symbol
  */
def elemwise_sub(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise exponential value of the input.<br>
  * <br>
  * .. math::<br>
  *    exp(x) = e^x \approx 2.718^x<br>
  * <br>
  * Example::<br>
  * <br>
  *    exp([0, 1, 2]) = [1., 2.71828175, 7.38905621]<br>
  * <br>
  * The storage type of ``exp`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L746<br>
  * @return org.apache.mxnet.Symbol
  */
def exp(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Inserts a new axis of size 1 into the array shape<br>
  * <br>
  * For example, given ``x`` with shape ``(2,3,4)``, then ``expand_dims(x, axis=1)``<br>
  * will return a new array with shape ``(2,1,3,4)``.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L346<br>
  * @return org.apache.mxnet.Symbol
  */
def expand_dims(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns ``exp(x) - 1`` computed element-wise on the input.<br>
  * <br>
  * This function provides greater precision than ``exp(x) - 1`` for small values of ``x``.<br>
  * <br>
  * The storage type of ``expm1`` output depends upon the input storage type:<br>
  * <br>
  *    - expm1(default) = default<br>
  *    - expm1(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L825<br>
  * @return org.apache.mxnet.Symbol
  */
def expm1(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Fill one element of each line(row for python, column for R/Julia) in lhs according to index indicated by rhs and values indicated by mhs. This function assume rhs uses 0-based index.<br>
  * @return org.apache.mxnet.Symbol
  */
def fill_element_0index(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise rounded value to the nearest \<br>
  * integer towards zero of the input.<br>
  * <br>
  * Example::<br>
  * <br>
  *    fix([-2.1, -1.9, 1.9, 2.1]) = [-2., -1.,  1., 2.]<br>
  * <br>
  * The storage type of ``fix`` output depends upon the input storage type:<br>
  * <br>
  *    - fix(default) = default<br>
  *    - fix(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L625<br>
  * @return org.apache.mxnet.Symbol
  */
def fix(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Flattens the input array into a 2-D array by collapsing the higher dimensions.<br>
  * <br>
  * .. note:: `Flatten` is deprecated. Use `flatten` instead.<br>
  * <br>
  * For an input array with shape ``(d1, d2, ..., dk)``, `flatten` operation reshapes<br>
  * the input array into an output array of shape ``(d1, d2*...*dk)``.<br>
  * <br>
  * Note that the bahavior of this function is different from numpy.ndarray.flatten,<br>
  * which behaves similar to mxnet.ndarray.reshape((-1,)).<br>
  * <br>
  * Example::<br>
  * <br>
  *     x = [[<br>
  *         [1,2,3],<br>
  *         [4,5,6],<br>
  *         [7,8,9]<br>
  *     ],<br>
  *     [    [1,2,3],<br>
  *         [4,5,6],<br>
  *         [7,8,9]<br>
  *     ]],<br>
  * <br>
  *     flatten(x) = [[ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.],<br>
  *        [ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L258<br>
  * @return org.apache.mxnet.Symbol
  */
def flatten(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reverses the order of elements along given axis while preserving array shape.<br>
  * <br>
  * Note: reverse and flip are equivalent. We use reverse in the following examples.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  1.,  2.,  3.,  4.],<br>
  *        [ 5.,  6.,  7.,  8.,  9.]]<br>
  * <br>
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],<br>
  *                         [ 0.,  1.,  2.,  3.,  4.]]<br>
  * <br>
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],<br>
  *                         [ 9.,  8.,  7.,  6.,  5.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L792<br>
  * @return org.apache.mxnet.Symbol
  */
def flip(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise floor of the input.<br>
  * <br>
  * The floor of the scalar x is the largest integer i, such that i <= x.<br>
  * <br>
  * Example::<br>
  * <br>
  *    floor([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-3., -2.,  1.,  1.,  2.]<br>
  * <br>
  * The storage type of ``floor`` output depends upon the input storage type:<br>
  * <br>
  *    - floor(default) = default<br>
  *    - floor(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L587<br>
  * @return org.apache.mxnet.Symbol
  */
def floor(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * The FTML optimizer described in<br>
  * *FTML - Follow the Moving Leader in Deep Learning*,<br>
  * available at http://proceedings.mlr.press/v70/zheng17a/zheng17a.pdf.<br>
  * <br>
  * .. math::<br>
  * <br>
  *  g_t = \nabla J(W_{t-1})\\<br>
  *  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\<br>
  *  d_t = \frac{ 1 - \beta_1^t }{ \eta_t } (\sqrt{ \frac{ v_t }{ 1 - \beta_2^t } } + \epsilon)<br>
  *  \sigma_t = d_t - \beta_1 d_{t-1}<br>
  *  z_t = \beta_1 z_{ t-1 } + (1 - \beta_1^t) g_t - \sigma_t W_{t-1}<br>
  *  W_t = - \frac{ z_t }{ d_t }<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L447<br>
  * @return org.apache.mxnet.Symbol
  */
def ftml_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for Ftrl optimizer.<br>
  * Referenced from *Ad Click Prediction: a View from the Trenches*, available at<br>
  * http://dl.acm.org/citation.cfm?id=2488200.<br>
  * <br>
  * It updates the weights using::<br>
  * <br>
  *  rescaled_grad = clip(grad * rescale_grad, clip_gradient)<br>
  *  z += rescaled_grad - (sqrt(n + rescaled_grad**2) - sqrt(n)) * weight / learning_rate<br>
  *  n += rescaled_grad**2<br>
  *  w = (sign(z) * lamda1 - z) / ((beta + sqrt(n)) / learning_rate + wd) * (abs(z) > lamda1)<br>
  * <br>
  * If w, z and n are all of ``row_sparse`` storage type,<br>
  * only the row slices whose indices appear in grad.indices are updated (for w, z and n)::<br>
  * <br>
  *  for row in grad.indices:<br>
  *      rescaled_grad[row] = clip(grad[row] * rescale_grad, clip_gradient)<br>
  *      z[row] += rescaled_grad[row] - (sqrt(n[row] + rescaled_grad[row]**2) - sqrt(n[row])) * weight[row] / learning_rate<br>
  *      n[row] += rescaled_grad[row]**2<br>
  *      w[row] = (sign(z[row]) * lamda1 - z[row]) / ((beta + sqrt(n[row])) / learning_rate + wd) * (abs(z[row]) > lamda1)<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L632<br>
  * @return org.apache.mxnet.Symbol
  */
def ftrl_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the gamma function (extension of the factorial function \<br>
  * to the reals), computed element-wise on the input array.<br>
  * <br>
  * The storage type of ``gamma`` output is always dense<br>
  * @return org.apache.mxnet.Symbol
  */
def gamma(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise log of the absolute value of the gamma function \<br>
  * of the input.<br>
  * <br>
  * The storage type of ``gammaln`` output is always dense<br>
  * @return org.apache.mxnet.Symbol
  */
def gammaln(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Gather elements or slices from `data` and store to a tensor whose<br>
  * shape is defined by `indices`.<br>
  * <br>
  * Given `data` with shape `(X_0, X_1, ..., X_{N-1})` and indices with shape<br>
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})`,<br>
  * where `M <= N`. If `M == N`, output shape will simply be `(Y_0, ..., Y_{K-1})`.<br>
  * <br>
  * The elements in output is defined as follows::<br>
  * <br>
  *   output[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}] = data[indices[0, y_0, ..., y_{K-1}],<br>
  *                                                       ...,<br>
  *                                                       indices[M-1, y_0, ..., y_{K-1}],<br>
  *                                                       x_M, ..., x_{N-1}]<br>
  * <br>
  * Examples::<br>
  * <br>
  *   data = [[0, 1], [2, 3]]<br>
  *   indices = [[1, 1, 0], [0, 1, 0]]<br>
  *   gather_nd(data, indices) = [2, 3, 0]<br>
  * @return org.apache.mxnet.Symbol
  */
def gather_nd(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes hard sigmoid of x element-wise.<br>
  * <br>
  * .. math::<br>
  *    y = max(0, min(1, alpha * x + beta))<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L118<br>
  * @return org.apache.mxnet.Symbol
  */
def hard_sigmoid(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns a copy of the input.<br>
  * <br>
  * From:src/operator/tensor/elemwise_unary_op_basic.cc:205<br>
  * @return org.apache.mxnet.Symbol
  */
def identity(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the Khatri-Rao product of the input matrices.<br>
  * <br>
  * Given a collection of :math:`n` input matrices,<br>
  * <br>
  * .. math::<br>
  *    A_1 \in \mathbb{R}^{M_1 \times M}, \ldots, A_n \in \mathbb{R}^{M_n \times N},<br>
  * <br>
  * the (column-wise) Khatri-Rao product is defined as the matrix,<br>
  * <br>
  * .. math::<br>
  *    X = A_1 \otimes \cdots \otimes A_n \in \mathbb{R}^{(M_1 \cdots M_n) \times N},<br>
  * <br>
  * where the :math:`k` th column is equal to the column-wise outer product<br>
  * :math:`{A_1}_k \otimes \cdots \otimes {A_n}_k` where :math:`{A_i}_k` is the kth<br>
  * column of the ith matrix.<br>
  * <br>
  * Example::<br>
  * <br>
  *   >>> A = mx.nd.array([[1, -1],<br>
  *   >>>                  [2, -3]])<br>
  *   >>> B = mx.nd.array([[1, 4],<br>
  *   >>>                  [2, 5],<br>
  *   >>>                  [3, 6]])<br>
  *   >>> C = mx.nd.khatri_rao(A, B)<br>
  *   >>> print(C.asnumpy())<br>
  *   [[  1.  -4.]<br>
  *    [  2.  -5.]<br>
  *    [  3.  -6.]<br>
  *    [  2. -12.]<br>
  *    [  4. -15.]<br>
  *    [  6. -18.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/contrib/krprod.cc:L108<br>
  * @return org.apache.mxnet.Symbol
  */
def khatri_rao(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * LQ factorization for general matrix.<br>
  * Input is a tensor *A* of dimension *n >= 2*.<br>
  * <br>
  * If *n=2*, we compute the LQ factorization (LAPACK *gelqf*, followed by *orglq*). *A*<br>
  * must have shape *(x, y)* with *x <= y*, and must have full rank *=x*. The LQ<br>
  * factorization consists of *L* with shape *(x, x)* and *Q* with shape *(x, y)*, so<br>
  * that:<br>
  * <br>
  *    *A* = *L* \* *Q*<br>
  * <br>
  * Here, *L* is lower triangular (upper triangle equal to zero) with nonzero diagonal,<br>
  * and *Q* is row-orthonormal, meaning that<br>
  * <br>
  *    *Q* \* *Q*\ :sup:`T`<br>
  * <br>
  * is equal to the identity matrix of shape *(x, x)*.<br>
  * <br>
  * If *n>2*, *gelqf* is performed separately on the trailing two dimensions for all<br>
  * inputs (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single LQ factorization<br>
  *    A = [[1., 2., 3.], [4., 5., 6.]]<br>
  *    Q, L = gelqf(A)<br>
  *    Q = [[-0.26726124, -0.53452248, -0.80178373],<br>
  *         [0.87287156, 0.21821789, -0.43643578]]<br>
  *    L = [[-3.74165739, 0.],<br>
  *         [-8.55235974, 1.96396101]]<br>
  * <br>
  *    // Batch LQ factorization<br>
  *    A = [[[1., 2., 3.], [4., 5., 6.]],<br>
  *         [[7., 8., 9.], [10., 11., 12.]]]<br>
  *    Q, L = gelqf(A)<br>
  *    Q = [[[-0.26726124, -0.53452248, -0.80178373],<br>
  *          [0.87287156, 0.21821789, -0.43643578]],<br>
  *         [[-0.50257071, -0.57436653, -0.64616234],<br>
  *          [0.7620735, 0.05862104, -0.64483142]]]<br>
  *    L = [[[-3.74165739, 0.],<br>
  *          [-8.55235974, 1.96396101]],<br>
  *         [[-13.92838828, 0.],<br>
  *          [-19.09768702, 0.52758934]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L552<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_gelqf(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs general matrix multiplication and accumulation.<br>
  * Input are tensors *A*, *B*, *C*, each of dimension *n >= 2* and having the same shape<br>
  * on the leading *n-2* dimensions.<br>
  * <br>
  * If *n=2*, the BLAS3 function *gemm* is performed:<br>
  * <br>
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*) + *beta* \* *C*<br>
  * <br>
  * Here, *alpha* and *beta* are scalar parameters, and *op()* is either the identity or<br>
  * matrix transposition (depending on *transpose_a*, *transpose_b*).<br>
  * <br>
  * If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices<br>
  * are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis* <br>
  * parameter. By default, the trailing two dimensions will be used for matrix encoding.<br>
  * <br>
  * For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes<br>
  * calls. For example let *A*, *B*, *C* be 5 dimensional tensors. Then gemm(*A*, *B*, *C*, axis=1) is equivalent to<br>
  * <br>
  *     A1 = swapaxes(A, dim1=1, dim2=3)<br>
  *     B1 = swapaxes(B, dim1=1, dim2=3)<br>
  *     C = swapaxes(C, dim1=1, dim2=3)<br>
  *     C = gemm(A1, B1, C)<br>
  *     C = swapaxis(C, dim1=1, dim2=3)<br>
  * <br>
  * without the overhead of the additional swapaxis operations.<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix multiply-add<br>
  *    A = [[1.0, 1.0], [1.0, 1.0]]<br>
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]<br>
  *    C = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]<br>
  *    gemm(A, B, C, transpose_b=True, alpha=2.0, beta=10.0)<br>
  *            = [[14.0, 14.0, 14.0], [14.0, 14.0, 14.0]]<br>
  * <br>
  *    // Batch matrix multiply-add<br>
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]<br>
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]<br>
  *    C = [[[10.0]], [[0.01]]]<br>
  *    gemm(A, B, C, transpose_b=True, alpha=2.0 , beta=10.0)<br>
  *            = [[[104.0]], [[0.14]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L81<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_gemm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs general matrix multiplication.<br>
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape<br>
  * on the leading *n-2* dimensions.<br>
  * <br>
  * If *n=2*, the BLAS3 function *gemm* is performed:<br>
  * <br>
  *    *out* = *alpha* \* *op*\ (*A*) \* *op*\ (*B*)<br>
  * <br>
  * Here *alpha* is a scalar parameter and *op()* is either the identity or the matrix<br>
  * transposition (depending on *transpose_a*, *transpose_b*).<br>
  * <br>
  * If *n>2*, *gemm* is performed separately for a batch of matrices. The column indices of the matrices<br>
  * are given by the last dimensions of the tensors, the row indices by the axis specified with the *axis* <br>
  * parameter. By default, the trailing two dimensions will be used for matrix encoding.<br>
  * <br>
  * For a non-default axis parameter, the operation performed is equivalent to a series of swapaxes/gemm/swapaxes<br>
  * calls. For example let *A*, *B* be 5 dimensional tensors. Then gemm(*A*, *B*, axis=1) is equivalent to<br>
  * <br>
  *     A1 = swapaxes(A, dim1=1, dim2=3)<br>
  *     B1 = swapaxes(B, dim1=1, dim2=3)<br>
  *     C = gemm2(A1, B1)<br>
  *     C = swapaxis(C, dim1=1, dim2=3)<br>
  * <br>
  * without the overhead of the additional swapaxis operations.<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix multiply<br>
  *    A = [[1.0, 1.0], [1.0, 1.0]]<br>
  *    B = [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]<br>
  *    gemm2(A, B, transpose_b=True, alpha=2.0)<br>
  *             = [[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]<br>
  * <br>
  *    // Batch matrix multiply<br>
  *    A = [[[1.0, 1.0]], [[0.1, 0.1]]]<br>
  *    B = [[[1.0, 1.0]], [[0.1, 0.1]]]<br>
  *    gemm2(A, B, transpose_b=True, alpha=2.0)<br>
  *            = [[[4.0]], [[0.04 ]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L151<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_gemm2(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs Cholesky factorization of a symmetric positive-definite matrix.<br>
  * Input is a tensor *A* of dimension *n >= 2*.<br>
  * <br>
  * If *n=2*, the Cholesky factor *L* of the symmetric, positive definite matrix *A* is<br>
  * computed. *L* is lower triangular (entries of upper triangle are all zero), has<br>
  * positive diagonal entries, and:<br>
  * <br>
  *   *A* = *L* \* *L*\ :sup:`T`<br>
  * <br>
  * If *n>2*, *potrf* is performed separately on the trailing two dimensions for all inputs<br>
  * (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix factorization<br>
  *    A = [[4.0, 1.0], [1.0, 4.25]]<br>
  *    potrf(A) = [[2.0, 0], [0.5, 2.0]]<br>
  * <br>
  *    // Batch matrix factorization<br>
  *    A = [[[4.0, 1.0], [1.0, 4.25]], [[16.0, 4.0], [4.0, 17.0]]]<br>
  *    potrf(A) = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L201<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_potrf(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs matrix inversion from a Cholesky factorization.<br>
  * Input is a tensor *A* of dimension *n >= 2*.<br>
  * <br>
  * If *n=2*, *A* is a lower triangular matrix (entries of upper triangle are all zero)<br>
  * with positive diagonal. We compute:<br>
  * <br>
  *   *out* = *A*\ :sup:`-T` \* *A*\ :sup:`-1`<br>
  * <br>
  * In other words, if *A* is the Cholesky factor of a symmetric positive definite matrix<br>
  * *B* (obtained by *potrf*), then<br>
  * <br>
  *   *out* = *B*\ :sup:`-1`<br>
  * <br>
  * If *n>2*, *potri* is performed separately on the trailing two dimensions for all inputs<br>
  * (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * .. note:: Use this operator only if you are certain you need the inverse of *B*, and<br>
  *           cannot use the Cholesky factor *A* (*potrf*), together with backsubstitution<br>
  *           (*trsm*). The latter is numerically much safer, and also cheaper.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix inverse<br>
  *    A = [[2.0, 0], [0.5, 2.0]]<br>
  *    potri(A) = [[0.26563, -0.0625], [-0.0625, 0.25]]<br>
  * <br>
  *    // Batch matrix inverse<br>
  *    A = [[[2.0, 0], [0.5, 2.0]], [[4.0, 0], [1.0, 4.0]]]<br>
  *    potri(A) = [[[0.26563, -0.0625], [-0.0625, 0.25]],<br>
  *                [[0.06641, -0.01562], [-0.01562, 0,0625]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L259<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_potri(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the sum of the logarithms of the diagonal elements of a square matrix.<br>
  * Input is a tensor *A* of dimension *n >= 2*.<br>
  * <br>
  * If *n=2*, *A* must be square with positive diagonal entries. We sum the natural<br>
  * logarithms of the diagonal elements, the result has shape (1,).<br>
  * <br>
  * If *n>2*, *sumlogdiag* is performed separately on the trailing two dimensions for all<br>
  * inputs (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix reduction<br>
  *    A = [[1.0, 1.0], [1.0, 7.0]]<br>
  *    sumlogdiag(A) = [1.9459]<br>
  * <br>
  *    // Batch matrix reduction<br>
  *    A = [[[1.0, 1.0], [1.0, 7.0]], [[3.0, 0], [0, 17.0]]]<br>
  *    sumlogdiag(A) = [1.9459, 3.9318]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L428<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_sumlogdiag(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Multiplication of matrix with its transpose.<br>
  * Input is a tensor *A* of dimension *n >= 2*.<br>
  * <br>
  * If *n=2*, the operator performs the BLAS3 function *syrk*:<br>
  * <br>
  *   *out* = *alpha* \* *A* \* *A*\ :sup:`T`<br>
  * <br>
  * if *transpose=False*, or<br>
  * <br>
  *   *out* = *alpha* \* *A*\ :sup:`T` \ \* *A*<br>
  * <br>
  * if *transpose=True*.<br>
  * <br>
  * If *n>2*, *syrk* is performed separately on the trailing two dimensions for all<br>
  * inputs (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix multiply<br>
  *    A = [[1., 2., 3.], [4., 5., 6.]]<br>
  *    syrk(A, alpha=1., transpose=False)<br>
  *             = [[14., 32.],<br>
  *                [32., 77.]]<br>
  *    syrk(A, alpha=1., transpose=True)<br>
  *             = [[17., 22., 27.],<br>
  *                [22., 29., 36.],<br>
  *                [27., 36., 45.]]<br>
  * <br>
  *    // Batch matrix multiply<br>
  *    A = [[[1., 1.]], [[0.1, 0.1]]]<br>
  *    syrk(A, alpha=2., transpose=False) = [[[4.]], [[0.04]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L484<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_syrk(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Performs multiplication with a lower triangular matrix.<br>
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape<br>
  * on the leading *n-2* dimensions.<br>
  * <br>
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function<br>
  * *trmm*:<br>
  * <br>
  *    *out* = *alpha* \* *op*\ (*A*) \* *B*<br>
  * <br>
  * if *rightside=False*, or<br>
  * <br>
  *    *out* = *alpha* \* *B* \* *op*\ (*A*)<br>
  * <br>
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the<br>
  * identity or the matrix transposition (depending on *transpose*).<br>
  * <br>
  * If *n>2*, *trmm* is performed separately on the trailing two dimensions for all inputs<br>
  * (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single triangular matrix multiply<br>
  *    A = [[1.0, 0], [1.0, 1.0]]<br>
  *    B = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]<br>
  *    trmm(A, B, alpha=2.0) = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]<br>
  * <br>
  *    // Batch triangular matrix multiply<br>
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]<br>
  *    B = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]<br>
  *    trmm(A, B, alpha=2.0) = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],<br>
  *                             [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L316<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_trmm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Solves matrix equation involving a lower triangular matrix.<br>
  * Input are tensors *A*, *B*, each of dimension *n >= 2* and having the same shape<br>
  * on the leading *n-2* dimensions.<br>
  * <br>
  * If *n=2*, *A* must be lower triangular. The operator performs the BLAS3 function<br>
  * *trsm*, solving for *out* in:<br>
  * <br>
  *    *op*\ (*A*) \* *out* = *alpha* \* *B*<br>
  * <br>
  * if *rightside=False*, or<br>
  * <br>
  *    *out* \* *op*\ (*A*) = *alpha* \* *B*<br>
  * <br>
  * if *rightside=True*. Here, *alpha* is a scalar parameter, and *op()* is either the<br>
  * identity or the matrix transposition (depending on *transpose*).<br>
  * <br>
  * If *n>2*, *trsm* is performed separately on the trailing two dimensions for all inputs<br>
  * (batch mode).<br>
  * <br>
  * .. note:: The operator supports float32 and float64 data types only.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    // Single matrix solve<br>
  *    A = [[1.0, 0], [1.0, 1.0]]<br>
  *    B = [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]]<br>
  *    trsm(A, B, alpha=0.5) = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]<br>
  * <br>
  *    // Batch matrix solve<br>
  *    A = [[[1.0, 0], [1.0, 1.0]], [[1.0, 0], [1.0, 1.0]]]<br>
  *    B = [[[2.0, 2.0, 2.0], [4.0, 4.0, 4.0]],<br>
  *         [[4.0, 4.0, 4.0], [8.0, 8.0, 8.0]]]<br>
  *    trsm(A, B, alpha=0.5) = [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],<br>
  *                             [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/la_op.cc:L379<br>
  * @return org.apache.mxnet.Symbol
  */
def linalg_trsm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise Natural logarithmic value of the input.<br>
  * <br>
  * The natural logarithm is logarithm in base *e*, so that ``log(exp(x)) = x``<br>
  * <br>
  * The storage type of ``log`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L758<br>
  * @return org.apache.mxnet.Symbol
  */
def log(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise Base-10 logarithmic value of the input.<br>
  * <br>
  * ``10**log10(x) = x``<br>
  * <br>
  * The storage type of ``log10`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L770<br>
  * @return org.apache.mxnet.Symbol
  */
def log10(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise ``log(1 + x)`` value of the input.<br>
  * <br>
  * This function is more accurate than ``log(1 + x)``  for small ``x`` so that<br>
  * :math:`1+x\approx 1`<br>
  * <br>
  * The storage type of ``log1p`` output depends upon the input storage type:<br>
  * <br>
  *    - log1p(default) = default<br>
  *    - log1p(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L807<br>
  * @return org.apache.mxnet.Symbol
  */
def log1p(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise Base-2 logarithmic value of the input.<br>
  * <br>
  * ``2**log2(x) = x``<br>
  * <br>
  * The storage type of ``log2`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L782<br>
  * @return org.apache.mxnet.Symbol
  */
def log2(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the log softmax of the input.<br>
  * This is equivalent to computing softmax followed by log.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   >>> x = mx.nd.array([1, 2, .1])<br>
  *   >>> mx.nd.log_softmax(x).asnumpy()<br>
  *   array([-1.41702998, -0.41702995, -2.31702995], dtype=float32)<br>
  * <br>
  *   >>> x = mx.nd.array( [[1, 2, .1],[.1, 2, 1]] )<br>
  *   >>> mx.nd.log_softmax(x, axis=0).asnumpy()<br>
  *   array([[-0.34115392, -0.69314718, -1.24115396],<br>
  *          [-1.24115396, -0.69314718, -0.34115392]], dtype=float32)<br>
  * @return org.apache.mxnet.Symbol
  */
def log_softmax(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the result of logical NOT (!) function<br>
  * <br>
  * Example:<br>
  *   logical_not([-2., 0., 1.]) = [0., 1., 0.]<br>
  * @return org.apache.mxnet.Symbol
  */
def logical_not(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Make your own loss function in network construction.<br>
  * <br>
  * This operator accepts a customized loss function symbol as a terminal loss and<br>
  * the symbol should be an operator with no backward dependency.<br>
  * The output of this function is the gradient of loss with respect to the input data.<br>
  * <br>
  * For example, if you are a making a cross entropy loss function. Assume ``out`` is the<br>
  * predicted output and ``label`` is the true label, then the cross entropy can be defined as::<br>
  * <br>
  *   cross_entropy = label * log(out) + (1 - label) * log(1 - out)<br>
  *   loss = make_loss(cross_entropy)<br>
  * <br>
  * We will need to use ``make_loss`` when we are creating our own loss function or we want to<br>
  * combine multiple loss functions. Also we may want to stop some variables' gradients<br>
  * from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.<br>
  * <br>
  * The storage type of ``make_loss`` output depends upon the input storage type:<br>
  * <br>
  *    - make_loss(default) = default<br>
  *    - make_loss(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L303<br>
  * @return org.apache.mxnet.Symbol
  */
def make_loss(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the max of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L190<br>
  * @return org.apache.mxnet.Symbol
  */
def max(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the max of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L190<br>
  * @return org.apache.mxnet.Symbol
  */
def max_axis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the mean of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L131<br>
  * @return org.apache.mxnet.Symbol
  */
def mean(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the min of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L204<br>
  * @return org.apache.mxnet.Symbol
  */
def min(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the min of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L204<br>
  * @return org.apache.mxnet.Symbol
  */
def min_axis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Updater function for multi-precision sgd optimizer<br>
  * @return org.apache.mxnet.Symbol
  */
def mp_sgd_mom_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Updater function for multi-precision sgd optimizer<br>
  * @return org.apache.mxnet.Symbol
  */
def mp_sgd_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the product of array elements over given axes treating Not a Numbers (``NaN``) as one.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L176<br>
  * @return org.apache.mxnet.Symbol
  */
def nanprod(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the sum of array elements over given axes treating Not a Numbers (``NaN``) as zero.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L161<br>
  * @return org.apache.mxnet.Symbol
  */
def nansum(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Numerical negative of the argument, element-wise.<br>
  * <br>
  * The storage type of ``negative`` output depends upon the input storage type:<br>
  * <br>
  *    - negative(default) = default<br>
  *    - negative(row_sparse) = row_sparse<br>
  *    - negative(csr) = csr<br>
  * @return org.apache.mxnet.Symbol
  */
def negative(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the norm on an NDArray.<br>
  * <br>
  * This operator computes the norm on an NDArray with the specified axis, depending<br>
  * on the value of the ord parameter. By default, it computes the L2 norm on the entire<br>
  * array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[1, 2],<br>
  *        [3, 4]]<br>
  * <br>
  *   norm(x) = [5.47722578]<br>
  * <br>
  *   rsp = x.cast_storage('row_sparse')<br>
  * <br>
  *   norm(rsp) = [5.47722578]<br>
  * <br>
  *   csr = x.cast_storage('csr')<br>
  * <br>
  *   norm(csr) = [5.47722578]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L300<br>
  * @return org.apache.mxnet.Symbol
  */
def norm(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a normal (Gaussian) distribution.<br>
  * <br>
  * .. note:: The existing alias ``normal`` is deprecated.<br>
  * <br>
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).<br>
  * <br>
  * Example::<br>
  * <br>
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],<br>
  *                                           [-1.23474145,  1.55807114]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L85<br>
  * @return org.apache.mxnet.Symbol
  */
def normal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns a one-hot array.<br>
  * <br>
  * The locations represented by `indices` take value `on_value`, while all<br>
  * other locations take value `off_value`.<br>
  * <br>
  * `one_hot` operation with `indices` of shape ``(i0, i1)`` and `depth`  of ``d`` would result<br>
  * in an output array of shape ``(i0, i1, d)`` with::<br>
  * <br>
  *   output[i,j,:] = off_value<br>
  *   output[i,j,indices[i,j]] = on_value<br>
  * <br>
  * Examples::<br>
  * <br>
  *   one_hot([1,0,2,0], 3) = [[ 0.  1.  0.]<br>
  *                            [ 1.  0.  0.]<br>
  *                            [ 0.  0.  1.]<br>
  *                            [ 1.  0.  0.]]<br>
  * <br>
  *   one_hot([1,0,2,0], 3, on_value=8, off_value=1,<br>
  *           dtype='int32') = [[1 8 1]<br>
  *                             [8 1 1]<br>
  *                             [1 1 8]<br>
  *                             [8 1 1]]<br>
  * <br>
  *   one_hot([[1,0],[1,0],[2,0]], 3) = [[[ 0.  1.  0.]<br>
  *                                       [ 1.  0.  0.]]<br>
  * <br>
  *                                      [[ 0.  1.  0.]<br>
  *                                       [ 1.  0.  0.]]<br>
  * <br>
  *                                      [[ 0.  0.  1.]<br>
  *                                       [ 1.  0.  0.]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/indexing_op.cc:L490<br>
  * @return org.apache.mxnet.Symbol
  */
def one_hot(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Return an array of ones with the same shape and type<br>
  * as the input array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  0.,  0.],<br>
  *        [ 0.,  0.,  0.]]<br>
  * <br>
  *   ones_like(x) = [[ 1.,  1.,  1.],<br>
  *                   [ 1.,  1.,  1.]]<br>
  * @return org.apache.mxnet.Symbol
  */
def ones_like(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Pads an input array with a constant or edge values of the array.<br>
  * <br>
  * .. note:: `Pad` is deprecated. Use `pad` instead.<br>
  * <br>
  * .. note:: Current implementation only supports 4D and 5D input arrays with padding applied<br>
  *    only on axes 1, 2 and 3. Expects axes 4 and 5 in `pad_width` to be zero.<br>
  * <br>
  * This operation pads an input array with either a `constant_value` or edge values<br>
  * along each axis of the input array. The amount of padding is specified by `pad_width`.<br>
  * <br>
  * `pad_width` is a tuple of integer padding widths for each axis of the format<br>
  * ``(before_1, after_1, ... , before_N, after_N)``. The `pad_width` should be of length ``2*N``<br>
  * where ``N`` is the number of dimensions of the array.<br>
  * <br>
  * For dimension ``N`` of the input array, ``before_N`` and ``after_N`` indicates how many values<br>
  * to add before and after the elements of the array along dimension ``N``.<br>
  * The widths of the higher two dimensions ``before_1``, ``after_1``, ``before_2``,<br>
  * ``after_2`` must be 0.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x = [[[[  1.   2.   3.]<br>
  *           [  4.   5.   6.]]<br>
  * <br>
  *          [[  7.   8.   9.]<br>
  *           [ 10.  11.  12.]]]<br>
  * <br>
  * <br>
  *         [[[ 11.  12.  13.]<br>
  *           [ 14.  15.  16.]]<br>
  * <br>
  *          [[ 17.  18.  19.]<br>
  *           [ 20.  21.  22.]]]]<br>
  * <br>
  *    pad(x,mode="edge", pad_width=(0,0,0,0,1,1,1,1)) =<br>
  * <br>
  *          [[[[  1.   1.   2.   3.   3.]<br>
  *             [  1.   1.   2.   3.   3.]<br>
  *             [  4.   4.   5.   6.   6.]<br>
  *             [  4.   4.   5.   6.   6.]]<br>
  * <br>
  *            [[  7.   7.   8.   9.   9.]<br>
  *             [  7.   7.   8.   9.   9.]<br>
  *             [ 10.  10.  11.  12.  12.]<br>
  *             [ 10.  10.  11.  12.  12.]]]<br>
  * <br>
  * <br>
  *           [[[ 11.  11.  12.  13.  13.]<br>
  *             [ 11.  11.  12.  13.  13.]<br>
  *             [ 14.  14.  15.  16.  16.]<br>
  *             [ 14.  14.  15.  16.  16.]]<br>
  * <br>
  *            [[ 17.  17.  18.  19.  19.]<br>
  *             [ 17.  17.  18.  19.  19.]<br>
  *             [ 20.  20.  21.  22.  22.]<br>
  *             [ 20.  20.  21.  22.  22.]]]]<br>
  * <br>
  *    pad(x, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,1,1,1)) =<br>
  * <br>
  *          [[[[  0.   0.   0.   0.   0.]<br>
  *             [  0.   1.   2.   3.   0.]<br>
  *             [  0.   4.   5.   6.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]<br>
  * <br>
  *            [[  0.   0.   0.   0.   0.]<br>
  *             [  0.   7.   8.   9.   0.]<br>
  *             [  0.  10.  11.  12.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]]<br>
  * <br>
  * <br>
  *           [[[  0.   0.   0.   0.   0.]<br>
  *             [  0.  11.  12.  13.   0.]<br>
  *             [  0.  14.  15.  16.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]<br>
  * <br>
  *            [[  0.   0.   0.   0.   0.]<br>
  *             [  0.  17.  18.  19.   0.]<br>
  *             [  0.  20.  21.  22.   0.]<br>
  *             [  0.   0.   0.   0.   0.]]]]<br>
  * <br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/pad.cc:L766<br>
  * @return org.apache.mxnet.Symbol
  */
def pad(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Picks elements from an input array according to the input indices along the given axis.<br>
  * <br>
  * Given an input array of shape ``(d0, d1)`` and indices of shape ``(i0,)``, the result will be<br>
  * an output array of shape ``(i0,)`` with::<br>
  * <br>
  *   output[i] = input[i, indices[i]]<br>
  * <br>
  * By default, if any index mentioned is too large, it is replaced by the index that addresses<br>
  * the last element along an axis (the `clip` mode).<br>
  * <br>
  * This function supports n-dimensional input and (n-1)-dimensional indices arrays.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 1.,  2.],<br>
  *        [ 3.,  4.],<br>
  *        [ 5.,  6.]]<br>
  * <br>
  *   // picks elements with specified indices along axis 0<br>
  *   pick(x, y=[0,1], 0) = [ 1.,  4.]<br>
  * <br>
  *   // picks elements with specified indices along axis 1<br>
  *   pick(x, y=[0,1,0], 1) = [ 1.,  4.,  5.]<br>
  * <br>
  *   y = [[ 1.],<br>
  *        [ 0.],<br>
  *        [ 2.]]<br>
  * <br>
  *   // picks elements with specified indices along axis 1 and dims are maintained<br>
  *   pick(x,y, 1, keepdims=True) = [[ 2.],<br>
  *                                  [ 3.],<br>
  *                                  [ 6.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_index.cc:L145<br>
  * @return org.apache.mxnet.Symbol
  */
def pick(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the product of array elements over given axes.<br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L146<br>
  * @return org.apache.mxnet.Symbol
  */
def prod(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Converts each element of the input array from degrees to radians.<br>
  * <br>
  * .. math::<br>
  *    radians([0, 90, 180, 270, 360]) = [0, \pi/2, \pi, 3\pi/2, 2\pi]<br>
  * <br>
  * The storage type of ``radians`` output depends upon the input storage type:<br>
  * <br>
  *    - radians(default) = default<br>
  *    - radians(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L182<br>
  * @return org.apache.mxnet.Symbol
  */
def radians(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from an exponential distribution.<br>
  * <br>
  * Samples are distributed according to an exponential distribution parametrized by *lambda* (rate).<br>
  * <br>
  * Example::<br>
  * <br>
  *    exponential(lam=4, shape=(2,2)) = [[ 0.0097189 ,  0.08999364],<br>
  *                                       [ 0.04146638,  0.31715935]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L115<br>
  * @return org.apache.mxnet.Symbol
  */
def random_exponential(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a gamma distribution.<br>
  * <br>
  * Samples are distributed according to a gamma distribution parametrized by *alpha* (shape) and *beta* (scale).<br>
  * <br>
  * Example::<br>
  * <br>
  *    gamma(alpha=9, beta=0.5, shape=(2,2)) = [[ 7.10486984,  3.37695289],<br>
  *                                             [ 3.91697288,  3.65933681]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L100<br>
  * @return org.apache.mxnet.Symbol
  */
def random_gamma(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a generalized negative binomial distribution.<br>
  * <br>
  * Samples are distributed according to a generalized negative binomial distribution parametrized by<br>
  * *mu* (mean) and *alpha* (dispersion). *alpha* is defined as *1/k* where *k* is the failure limit of the<br>
  * number of unsuccessful experiments (generalized to real numbers).<br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Example::<br>
  * <br>
  *    generalized_negative_binomial(mu=2.0, alpha=0.3, shape=(2,2)) = [[ 2.,  1.],<br>
  *                                                                     [ 6.,  4.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L168<br>
  * @return org.apache.mxnet.Symbol
  */
def random_generalized_negative_binomial(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a negative binomial distribution.<br>
  * <br>
  * Samples are distributed according to a negative binomial distribution parametrized by<br>
  * *k* (limit of unsuccessful experiments) and *p* (failure probability in each experiment).<br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Example::<br>
  * <br>
  *    negative_binomial(k=3, p=0.4, shape=(2,2)) = [[ 4.,  7.],<br>
  *                                                  [ 2.,  5.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L149<br>
  * @return org.apache.mxnet.Symbol
  */
def random_negative_binomial(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a normal (Gaussian) distribution.<br>
  * <br>
  * .. note:: The existing alias ``normal`` is deprecated.<br>
  * <br>
  * Samples are distributed according to a normal distribution parametrized by *loc* (mean) and *scale* (standard deviation).<br>
  * <br>
  * Example::<br>
  * <br>
  *    normal(loc=0, scale=1, shape=(2,2)) = [[ 1.89171135, -1.16881478],<br>
  *                                           [-1.23474145,  1.55807114]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L85<br>
  * @return org.apache.mxnet.Symbol
  */
def random_normal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a Poisson distribution.<br>
  * <br>
  * Samples are distributed according to a Poisson distribution parametrized by *lambda* (rate).<br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Example::<br>
  * <br>
  *    poisson(lam=4, shape=(2,2)) = [[ 5.,  2.],<br>
  *                                   [ 4.,  6.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L132<br>
  * @return org.apache.mxnet.Symbol
  */
def random_poisson(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a uniform distribution.<br>
  * <br>
  * .. note:: The existing alias ``uniform`` is deprecated.<br>
  * <br>
  * Samples are uniformly distributed over the half-open interval *[low, high)*<br>
  * (includes *low*, but excludes *high*).<br>
  * <br>
  * Example::<br>
  * <br>
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],<br>
  *                                           [ 0.54488319,  0.84725171]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L66<br>
  * @return org.apache.mxnet.Symbol
  */
def random_uniform(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a single multi index is given by a column of the input matrix. <br>
  * <br>
  * Examples::<br>
  *    <br>
  *    A = [[3,6,6],[4,5,1]]<br>
  *    ravel(A, shape=(7,6)) = [22,41,37]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/ravel.cc:L41<br>
  * @return org.apache.mxnet.Symbol
  */
def ravel_multi_index(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise inverse cube-root value of the input.<br>
  * <br>
  * .. math::<br>
  *    rcbrt(x) = 1/\sqrt[3]{x}<br>
  * <br>
  * Example::<br>
  * <br>
  *    rcbrt([1,8,-125]) = [1.0, 0.5, -0.2]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L723<br>
  * @return org.apache.mxnet.Symbol
  */
def rcbrt(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the reciprocal of the argument, element-wise.<br>
  * <br>
  * Calculates 1/x.<br>
  * <br>
  * Example::<br>
  * <br>
  *     reciprocal([-2, 1, 3, 1.6, 0.2]) = [-0.5, 1.0, 0.33333334, 0.625, 5.0]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L468<br>
  * @return org.apache.mxnet.Symbol
  */
def reciprocal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes rectified linear.<br>
  * <br>
  * .. math::<br>
  *    max(features, 0)<br>
  * <br>
  * The storage type of ``relu`` output depends upon the input storage type:<br>
  * <br>
  *    - relu(default) = default<br>
  *    - relu(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L85<br>
  * @return org.apache.mxnet.Symbol
  */
def relu(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Repeats elements of an array.<br>
  * <br>
  * By default, ``repeat`` flattens the input array into 1-D and then repeats the<br>
  * elements::<br>
  * <br>
  *   x = [[ 1, 2],<br>
  *        [ 3, 4]]<br>
  * <br>
  *   repeat(x, repeats=2) = [ 1.,  1.,  2.,  2.,  3.,  3.,  4.,  4.]<br>
  * <br>
  * The parameter ``axis`` specifies the axis along which to perform repeat::<br>
  * <br>
  *   repeat(x, repeats=2, axis=1) = [[ 1.,  1.,  2.,  2.],<br>
  *                                   [ 3.,  3.,  4.,  4.]]<br>
  * <br>
  *   repeat(x, repeats=2, axis=0) = [[ 1.,  2.],<br>
  *                                   [ 1.,  2.],<br>
  *                                   [ 3.,  4.],<br>
  *                                   [ 3.,  4.]]<br>
  * <br>
  *   repeat(x, repeats=2, axis=-1) = [[ 1.,  1.,  2.,  2.],<br>
  *                                    [ 3.,  3.,  4.,  4.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L690<br>
  * @return org.apache.mxnet.Symbol
  */
def repeat(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reshapes the input array.<br>
  * <br>
  * .. note:: ``Reshape`` is deprecated, use ``reshape``<br>
  * <br>
  * Given an array and a shape, this function returns a copy of the array in the new shape.<br>
  * The shape is a tuple of integers such as (2,3,4). The size of the new shape should be same as the size of the input array.<br>
  * <br>
  * Example::<br>
  * <br>
  *   reshape([1,2,3,4], shape=(2,2)) = [[1,2], [3,4]]<br>
  * <br>
  * Some dimensions of the shape can take special values from the set {0, -1, -2, -3, -4}. The significance of each is explained below:<br>
  * <br>
  * - ``0``  copy this dimension from the input to the output shape.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (4,0,2), output shape = (4,3,2)<br>
  *   - input shape = (2,3,4), shape = (2,0,0), output shape = (2,3,4)<br>
  * <br>
  * - ``-1`` infers the dimension of the output shape by using the remainder of the input dimensions<br>
  *   keeping the size of the new array same as that of the input array.<br>
  *   At most one dimension of shape can be -1.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (6,1,-1), output shape = (6,1,4)<br>
  *   - input shape = (2,3,4), shape = (3,-1,8), output shape = (3,1,8)<br>
  *   - input shape = (2,3,4), shape=(-1,), output shape = (24,)<br>
  * <br>
  * - ``-2`` copy all/remainder of the input dimensions to the output shape.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-2,), output shape = (2,3,4)<br>
  *   - input shape = (2,3,4), shape = (2,-2), output shape = (2,3,4)<br>
  *   - input shape = (2,3,4), shape = (-2,1,1), output shape = (2,3,4,1,1)<br>
  * <br>
  * - ``-3`` use the product of two consecutive dimensions of the input shape as the output dimension.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-3,4), output shape = (6,4)<br>
  *   - input shape = (2,3,4,5), shape = (-3,-3), output shape = (6,20)<br>
  *   - input shape = (2,3,4), shape = (0,-3), output shape = (2,12)<br>
  *   - input shape = (2,3,4), shape = (-3,-2), output shape = (6,4)<br>
  * <br>
  * - ``-4`` split one dimension of the input into two dimensions passed subsequent to -4 in shape (can contain -1).<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - input shape = (2,3,4), shape = (-4,1,2,-2), output shape =(1,2,3,4)<br>
  *   - input shape = (2,3,4), shape = (2,-4,-1,3,-2), output shape = (2,1,3,4)<br>
  * <br>
  * If the argument `reverse` is set to 1, then the special values are inferred from right to left.<br>
  * <br>
  *   Example::<br>
  * <br>
  *   - without reverse=1, for input shape = (10,5,4), shape = (-1,0), output shape would be (40,5)<br>
  *   - with reverse=1, output shape will be (50,4).<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L168<br>
  * @return org.apache.mxnet.Symbol
  */
def reshape(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reshape lhs to have the same shape as rhs.<br>
  * @return org.apache.mxnet.Symbol
  */
def reshape_like(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Reverses the order of elements along given axis while preserving array shape.<br>
  * <br>
  * Note: reverse and flip are equivalent. We use reverse in the following examples.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.,  1.,  2.,  3.,  4.],<br>
  *        [ 5.,  6.,  7.,  8.,  9.]]<br>
  * <br>
  *   reverse(x, axis=0) = [[ 5.,  6.,  7.,  8.,  9.],<br>
  *                         [ 0.,  1.,  2.,  3.,  4.]]<br>
  * <br>
  *   reverse(x, axis=1) = [[ 4.,  3.,  2.,  1.,  0.],<br>
  *                         [ 9.,  8.,  7.,  6.,  5.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L792<br>
  * @return org.apache.mxnet.Symbol
  */
def reverse(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise rounded value to the nearest integer of the input.<br>
  * <br>
  * .. note::<br>
  *    - For input ``n.5`` ``rint`` returns ``n`` while ``round`` returns ``n+1``.<br>
  *    - For input ``-n.5`` both ``rint`` and ``round`` returns ``-n-1``.<br>
  * <br>
  * Example::<br>
  * <br>
  *    rint([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  1., -2.,  2.,  2.]<br>
  * <br>
  * The storage type of ``rint`` output depends upon the input storage type:<br>
  * <br>
  *    - rint(default) = default<br>
  *    - rint(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L549<br>
  * @return org.apache.mxnet.Symbol
  */
def rint(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for `RMSProp` optimizer.<br>
  * <br>
  * `RMSprop` is a variant of stochastic gradient descent where the gradients are<br>
  * divided by a cache which grows with the sum of squares of recent gradients?<br>
  * <br>
  * `RMSProp` is similar to `AdaGrad`, a popular variant of `SGD` which adaptively<br>
  * tunes the learning rate of each parameter. `AdaGrad` lowers the learning rate for<br>
  * each parameter monotonically over the course of training.<br>
  * While this is analytically motivated for convex optimizations, it may not be ideal<br>
  * for non-convex problems. `RMSProp` deals with this heuristically by allowing the<br>
  * learning rates to rebound as the denominator decays over time.<br>
  * <br>
  * Define the Root Mean Square (RMS) error criterion of the gradient as<br>
  * :math:`RMS[g]_t = \sqrt{E[g^2]_t + \epsilon}`, where :math:`g` represents<br>
  * gradient and :math:`E[g^2]_t` is the decaying average over past squared gradient.<br>
  * <br>
  * The :math:`E[g^2]_t` is given by:<br>
  * <br>
  * .. math::<br>
  *   E[g^2]_t = \gamma * E[g^2]_{t-1} + (1-\gamma) * g_t^2<br>
  * <br>
  * The update step is<br>
  * <br>
  * .. math::<br>
  *   \theta_{t+1} = \theta_t - \frac{\eta}{RMS[g]_t} g_t<br>
  * <br>
  * The RMSProp code follows the version in<br>
  * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf<br>
  * Tieleman & Hinton, 2012.<br>
  * <br>
  * Hinton suggests the momentum term :math:`\gamma` to be 0.9 and the learning rate<br>
  * :math:`\eta` to be 0.001.<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L553<br>
  * @return org.apache.mxnet.Symbol
  */
def rmsprop_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for RMSPropAlex optimizer.<br>
  * <br>
  * `RMSPropAlex` is non-centered version of `RMSProp`.<br>
  * <br>
  * Define :math:`E[g^2]_t` is the decaying average over past squared gradient and<br>
  * :math:`E[g]_t` is the decaying average over past gradient.<br>
  * <br>
  * .. math::<br>
  *   E[g^2]_t = \gamma_1 * E[g^2]_{t-1} + (1 - \gamma_1) * g_t^2\\<br>
  *   E[g]_t = \gamma_1 * E[g]_{t-1} + (1 - \gamma_1) * g_t\\<br>
  *   \Delta_t = \gamma_2 * \Delta_{t-1} - \frac{\eta}{\sqrt{E[g^2]_t - E[g]_t^2 + \epsilon}} g_t\\<br>
  * <br>
  * The update step is<br>
  * <br>
  * .. math::<br>
  *   \theta_{t+1} = \theta_t + \Delta_t<br>
  * <br>
  * The RMSPropAlex code follows the version in<br>
  * http://arxiv.org/pdf/1308.0850v5.pdf Eq(38) - Eq(45) by Alex Graves, 2013.<br>
  * <br>
  * Graves suggests the momentum term :math:`\gamma_1` to be 0.95, :math:`\gamma_2`<br>
  * to be 0.9 and the learning rate :math:`\eta` to be 0.0001.<br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L592<br>
  * @return org.apache.mxnet.Symbol
  */
def rmspropalex_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise rounded value to the nearest integer of the input.<br>
  * <br>
  * Example::<br>
  * <br>
  *    round([-1.5, 1.5, -1.9, 1.9, 2.1]) = [-2.,  2., -2.,  2.,  2.]<br>
  * <br>
  * The storage type of ``round`` output depends upon the input storage type:<br>
  * <br>
  *   - round(default) = default<br>
  *   - round(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L528<br>
  * @return org.apache.mxnet.Symbol
  */
def round(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise inverse square-root value of the input.<br>
  * <br>
  * .. math::<br>
  *    rsqrt(x) = 1/\sqrt{x}<br>
  * <br>
  * Example::<br>
  * <br>
  *    rsqrt([4,9,16]) = [0.5, 0.33333334, 0.25]<br>
  * <br>
  * The storage type of ``rsqrt`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L689<br>
  * @return org.apache.mxnet.Symbol
  */
def rsqrt(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * exponential distributions with parameters lambda (rate).<br>
  * <br>
  * The parameters of the distributions are provided as an input array.<br>
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input value at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    lam = [ 1.0, 8.5 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_exponential(lam) = [ 0.51837951,  0.09994757]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_exponential(lam, shape=(2)) = [[ 0.51837951,  0.19866663],<br>
  *                                          [ 0.09994757,  0.50447971]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L284<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_exponential(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * gamma distributions with parameters *alpha* (shape) and *beta* (scale).<br>
  * <br>
  * The parameters of the distributions are provided as input arrays.<br>
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input values at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input arrays.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    alpha = [ 0.0, 2.5 ]<br>
  *    beta = [ 1.0, 0.7 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_gamma(alpha, beta) = [ 0.        ,  2.25797319]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_gamma(alpha, beta, shape=(2)) = [[ 0.        ,  0.        ],<br>
  *                                            [ 2.25797319,  1.70734084]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L282<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_gamma(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * generalized negative binomial distributions with parameters *mu* (mean) and *alpha* (dispersion).<br>
  * <br>
  * The parameters of the distributions are provided as input arrays.<br>
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input values at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input arrays.<br>
  * <br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    mu = [ 2.0, 2.5 ]<br>
  *    alpha = [ 1.0, 0.1 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_generalized_negative_binomial(mu, alpha) = [ 0.,  3.]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_generalized_negative_binomial(mu, alpha, shape=(2)) = [[ 0.,  3.],<br>
  *                                                                  [ 3.,  1.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L293<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_generalized_negative_binomial(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple multinomial distributions.<br>
  * <br>
  * *data* is an *n* dimensional array whose last dimension has length *k*, where<br>
  * *k* is the number of possible outcomes of each multinomial distribution. This<br>
  * operator will draw *shape* samples from each distribution. If shape is empty<br>
  * one sample will be drawn from each distribution.<br>
  * <br>
  * If *get_prob* is true, a second array containing log likelihood of the drawn<br>
  * samples will also be returned. This is usually used for reinforcement learning<br>
  * where you can provide reward as head gradient for this array to estimate<br>
  * gradient.<br>
  * <br>
  * Note that the input distribution must be normalized, i.e. *data* must sum to<br>
  * 1 along its last axis.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    probs = [[0, 0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1, 0]]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_multinomial(probs) = [3, 0]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_multinomial(probs, shape=(2)) = [[4, 2],<br>
  *                                            [0, 0]]<br>
  * <br>
  *    // requests log likelihood<br>
  *    sample_multinomial(probs, get_prob=True) = [2, 1], [0.2, 0.3]<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_multinomial(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * negative binomial distributions with parameters *k* (failure limit) and *p* (failure probability).<br>
  * <br>
  * The parameters of the distributions are provided as input arrays.<br>
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input values at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input arrays.<br>
  * <br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    k = [ 20, 49 ]<br>
  *    p = [ 0.4 , 0.77 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_negative_binomial(k, p) = [ 15.,  16.]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_negative_binomial(k, p, shape=(2)) = [[ 15.,  50.],<br>
  *                                                 [ 16.,  12.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L289<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_negative_binomial(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * normal distributions with parameters *mu* (mean) and *sigma* (standard deviation).<br>
  * <br>
  * The parameters of the distributions are provided as input arrays.<br>
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input values at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input arrays.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    mu = [ 0.0, 2.5 ]<br>
  *    sigma = [ 1.0, 3.7 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_normal(mu, sigma) = [-0.56410581,  0.95934606]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_normal(mu, sigma, shape=(2)) = [[-0.56410581,  0.2928229 ],<br>
  *                                           [ 0.95934606,  4.48287058]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L279<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_normal(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * Poisson distributions with parameters lambda (rate).<br>
  * <br>
  * The parameters of the distributions are provided as an input array.<br>
  * Let *[s]* be the shape of the input array, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input array, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input value at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input array.<br>
  * <br>
  * Samples will always be returned as a floating point data type.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    lam = [ 1.0, 8.5 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_poisson(lam) = [  0.,  13.]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_poisson(lam, shape=(2)) = [[  0.,   4.],<br>
  *                                      [ 13.,   8.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L286<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_poisson(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Concurrent sampling from multiple<br>
  * uniform distributions on the intervals given by *[low,high)*.<br>
  * <br>
  * The parameters of the distributions are provided as input arrays.<br>
  * Let *[s]* be the shape of the input arrays, *n* be the dimension of *[s]*, *[t]*<br>
  * be the shape specified as the parameter of the operator, and *m* be the dimension<br>
  * of *[t]*. Then the output will be a *(n+m)*-dimensional array with shape *[s]x[t]*.<br>
  * <br>
  * For any valid *n*-dimensional index *i* with respect to the input arrays, *output[i]*<br>
  * will be an *m*-dimensional array that holds randomly drawn samples from the distribution<br>
  * which is parameterized by the input values at index *i*. If the shape parameter of the<br>
  * operator is not set, then one sample will be drawn per distribution and the output array<br>
  * has the same shape as the input arrays.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    low = [ 0.0, 2.5 ]<br>
  *    high = [ 1.0, 3.7 ]<br>
  * <br>
  *    // Draw a single sample for each distribution<br>
  *    sample_uniform(low, high) = [ 0.40451524,  3.18687344]<br>
  * <br>
  *    // Draw a vector containing two samples for each distribution<br>
  *    sample_uniform(low, high, shape=(2)) = [[ 0.40451524,  0.18017688],<br>
  *                                            [ 3.18687344,  3.68352246]]<br>
  * <br>
  * <br>
  * Defined in src/operator/random/multisample_op.cc:L277<br>
  * @return org.apache.mxnet.Symbol
  */
def sample_uniform(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Scatters data into a new tensor according to indices.<br>
  * <br>
  * Given `data` with shape `(Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1})` and indices with shape<br>
  * `(M, Y_0, ..., Y_{K-1})`, the output will have shape `(X_0, X_1, ..., X_{N-1})`,<br>
  * where `M <= N`. If `M == N`, data shape should simply be `(Y_0, ..., Y_{K-1})`.<br>
  * <br>
  * The elements in output is defined as follows::<br>
  * <br>
  *   output[indices[0, y_0, ..., y_{K-1}],<br>
  *          ...,<br>
  *          indices[M-1, y_0, ..., y_{K-1}],<br>
  *          x_M, ..., x_{N-1}] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]<br>
  * <br>
  * all other entries in output are 0.<br>
  * <br>
  * .. warning::<br>
  * <br>
  *     If the indices have duplicates, the result will be non-deterministic and<br>
  *     the gradient of `scatter_nd` will not be correct!!<br>
  * <br>
  * <br>
  * Examples::<br>
  * <br>
  *   data = [2, 3, 0]<br>
  *   indices = [[1, 1, 0], [0, 1, 0]]<br>
  *   shape = (2, 2)<br>
  *   scatter_nd(data, indices, shape) = [[0, 0], [2, 3]]<br>
  * @return org.apache.mxnet.Symbol
  */
def scatter_nd(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Momentum update function for Stochastic Gradient Descent (SGD) optimizer.<br>
  * <br>
  * Momentum update has better convergence rates on neural networks. Mathematically it looks<br>
  * like below:<br>
  * <br>
  * .. math::<br>
  * <br>
  *   v_1 = \alpha * \nabla J(W_0)\\<br>
  *   v_t = \gamma v_{t-1} - \alpha * \nabla J(W_{t-1})\\<br>
  *   W_t = W_{t-1} + v_t<br>
  * <br>
  * It updates the weights using::<br>
  * <br>
  *   v = momentum * v - learning_rate * gradient<br>
  *   weight += v<br>
  * <br>
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.<br>
  * <br>
  * However, if grad's storage type is ``row_sparse``, ``lazy_update`` is True and weight's storage<br>
  * type is the same as momentum's storage type,<br>
  * only the row slices whose indices appear in grad.indices are updated (for both weight and momentum)::<br>
  * <br>
  *   for row in gradient.indices:<br>
  *       v[row] = momentum[row] * v[row] - learning_rate * gradient[row]<br>
  *       weight[row] += v[row]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L372<br>
  * @return org.apache.mxnet.Symbol
  */
def sgd_mom_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for Stochastic Gradient Descent (SDG) optimizer.<br>
  * <br>
  * It updates the weights using::<br>
  * <br>
  *  weight = weight - learning_rate * (gradient + wd * weight)<br>
  * <br>
  * However, if gradient is of ``row_sparse`` storage type and ``lazy_update`` is True,<br>
  * only the row slices whose indices appear in grad.indices are updated::<br>
  * <br>
  *  for row in gradient.indices:<br>
  *      weight[row] = weight[row] - learning_rate * (gradient[row] + wd * weight[row])<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L331<br>
  * @return org.apache.mxnet.Symbol
  */
def sgd_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Randomly shuffle the elements.<br>
  * <br>
  * This shuffles the array along the first axis.<br>
  * The order of the elements in each subarray does not change.<br>
  * For example, if a 2D array is given, the order of the rows randomly changes,<br>
  * but the order of the elements in each row does not change.<br>
  * @return org.apache.mxnet.Symbol
  */
def shuffle(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes sigmoid of x element-wise.<br>
  * <br>
  * .. math::<br>
  *    y = 1 / (1 + exp(-x))<br>
  * <br>
  * The storage type of ``sigmoid`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L104<br>
  * @return org.apache.mxnet.Symbol
  */
def sigmoid(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise sign of the input.<br>
  * <br>
  * Example::<br>
  * <br>
  *    sign([-2, 0, 3]) = [-1, 0, 1]<br>
  * <br>
  * The storage type of ``sign`` output depends upon the input storage type:<br>
  * <br>
  *    - sign(default) = default<br>
  *    - sign(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L509<br>
  * @return org.apache.mxnet.Symbol
  */
def sign(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Update function for SignSGD optimizer.<br>
  * <br>
  * .. math::<br>
  * <br>
  *  g_t = \nabla J(W_{t-1})\\<br>
  *  W_t = W_{t-1} - \eta_t \text{sign}(g_t)<br>
  * <br>
  * It updates the weights using::<br>
  * <br>
  *  weight = weight - learning_rate * sign(gradient)<br>
  * <br>
  * .. note:: <br>
  *    - sparse ndarray not supported for this optimizer yet.<br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L57<br>
  * @return org.apache.mxnet.Symbol
  */
def signsgd_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * SIGN momentUM (Signum) optimizer.<br>
  * <br>
  * .. math::<br>
  * <br>
  *  g_t = \nabla J(W_{t-1})\\<br>
  *  m_t = \beta m_{t-1} + (1 - \beta) g_t\\<br>
  *  W_t = W_{t-1} - \eta_t \text{sign}(m_t)<br>
  * <br>
  * It updates the weights using::<br>
  *  state = momentum * state + (1-momentum) * gradient<br>
  *  weight = weight - learning_rate * sign(state)<br>
  * <br>
  * Where the parameter ``momentum`` is the decay rate of momentum estimates at each epoch.<br>
  * <br>
  * .. note:: <br>
  *    - sparse ndarray not supported for this optimizer yet.<br>
  * <br>
  * <br>
  * Defined in src/operator/optimizer_op.cc:L86<br>
  * @return org.apache.mxnet.Symbol
  */
def signum_update(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the element-wise sine of the input array.<br>
  * <br>
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).<br>
  * <br>
  * .. math::<br>
  *    sin([0, \pi/4, \pi/2]) = [0, 0.707, 1]<br>
  * <br>
  * The storage type of ``sin`` output depends upon the input storage type:<br>
  * <br>
  *    - sin(default) = default<br>
  *    - sin(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L46<br>
  * @return org.apache.mxnet.Symbol
  */
def sin(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the hyperbolic sine of the input array, computed element-wise.<br>
  * <br>
  * .. math::<br>
  *    sinh(x) = 0.5\times(exp(x) - exp(-x))<br>
  * <br>
  * The storage type of ``sinh`` output depends upon the input storage type:<br>
  * <br>
  *    - sinh(default) = default<br>
  *    - sinh(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L201<br>
  * @return org.apache.mxnet.Symbol
  */
def sinh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Slices a region of the array.<br>
  * <br>
  * .. note:: ``crop`` is deprecated. Use ``slice`` instead.<br>
  * <br>
  * This function returns a sliced array between the indices given<br>
  * by `begin` and `end` with the corresponding `step`.<br>
  * <br>
  * For an input array of ``shape=(d_0, d_1, ..., d_n-1)``,<br>
  * slice operation with ``begin=(b_0, b_1...b_m-1)``,<br>
  * ``end=(e_0, e_1, ..., e_m-1)``, and ``step=(s_0, s_1, ..., s_m-1)``,<br>
  * where m <= n, results in an array with the shape<br>
  * ``(|e_0-b_0|/|s_0|, ..., |e_m-1-b_m-1|/|s_m-1|, d_m, ..., d_n-1)``.<br>
  * <br>
  * The resulting array's *k*-th dimension contains elements<br>
  * from the *k*-th dimension of the input array starting<br>
  * from index ``b_k`` (inclusive) with step ``s_k``<br>
  * until reaching ``e_k`` (exclusive).<br>
  * <br>
  * If the *k*-th elements are `None` in the sequence of `begin`, `end`,<br>
  * and `step`, the following rule will be used to set default values.<br>
  * If `s_k` is `None`, set `s_k=1`. If `s_k > 0`, set `b_k=0`, `e_k=d_k`;<br>
  * else, set `b_k=d_k-1`, `e_k=-1`.<br>
  * <br>
  * The storage type of ``slice`` output depends on storage types of inputs<br>
  * <br>
  * - slice(csr) = csr<br>
  * - otherwise, ``slice`` generates output with default storage<br>
  * <br>
  * .. note:: When input data storage type is csr, it only supports<br>
  * step=(), or step=(None,), or step=(1,) to generate a csr output.<br>
  * For other step parameter values, it falls back to slicing<br>
  * a dense tensor.<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[  1.,   2.,   3.,   4.],<br>
  *        [  5.,   6.,   7.,   8.],<br>
  *        [  9.,  10.,  11.,  12.]]<br>
  * <br>
  *   slice(x, begin=(0,1), end=(2,4)) = [[ 2.,  3.,  4.],<br>
  *                                      [ 6.,  7.,  8.]]<br>
  *   slice(x, begin=(None, 0), end=(None, 3), step=(-1, 2)) = [[9., 11.],<br>
  *                                                             [5.,  7.],<br>
  *                                                             [1.,  3.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L412<br>
  * @return org.apache.mxnet.Symbol
  */
def slice(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Slices along a given axis.<br>
  * <br>
  * Returns an array slice along a given `axis` starting from the `begin` index<br>
  * to the `end` index.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[  1.,   2.,   3.,   4.],<br>
  *        [  5.,   6.,   7.,   8.],<br>
  *        [  9.,  10.,  11.,  12.]]<br>
  * <br>
  *   slice_axis(x, axis=0, begin=1, end=3) = [[  5.,   6.,   7.,   8.],<br>
  *                                            [  9.,  10.,  11.,  12.]]<br>
  * <br>
  *   slice_axis(x, axis=1, begin=0, end=2) = [[  1.,   2.],<br>
  *                                            [  5.,   6.],<br>
  *                                            [  9.,  10.]]<br>
  * <br>
  *   slice_axis(x, axis=1, begin=-3, end=-1) = [[  2.,   3.],<br>
  *                                              [  6.,   7.],<br>
  *                                              [ 10.,  11.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L499<br>
  * @return org.apache.mxnet.Symbol
  */
def slice_axis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Slices a region of the array like the shape of another array.<br>
  * <br>
  * This function is similar to ``slice``, however, the `begin` are always `0`s<br>
  * and `end` of specific axes are inferred from the second input `shape_like`.<br>
  * <br>
  * Given the second `shape_like` input of ``shape=(d_0, d_1, ..., d_n-1)``,<br>
  * a ``slice_like`` operator with default empty `axes`, it performs the<br>
  * following operation:<br>
  * <br>
  * `` out = slice(input, begin=(0, 0, ..., 0), end=(d_0, d_1, ..., d_n-1))``.<br>
  * <br>
  * When `axes` is not empty, it is used to speficy which axes are being sliced.<br>
  * <br>
  * Given a 4-d input data, ``slice_like`` operator with ``axes=(0, 2, -1)``<br>
  * will perform the following operation:<br>
  * <br>
  * `` out = slice(input, begin=(0, 0, 0, 0), end=(d_0, None, d_2, d_3))``.<br>
  * <br>
  * Note that it is allowed to have first and second input with different dimensions,<br>
  * however, you have to make sure the `axes` are specified and not exceeding the<br>
  * dimension limits.<br>
  * <br>
  * For example, given `input_1` with ``shape=(2,3,4,5)`` and `input_2` with<br>
  * ``shape=(1,2,3)``, it is not allowed to use:<br>
  * <br>
  * `` out = slice_like(a, b)`` because ndim of `input_1` is 4, and ndim of `input_2`<br>
  * is 3.<br>
  * <br>
  * The following is allowed in this situation:<br>
  * <br>
  * `` out = slice_like(a, b, axes=(0, 2))``<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[  1.,   2.,   3.,   4.],<br>
  *        [  5.,   6.,   7.,   8.],<br>
  *        [  9.,  10.,  11.,  12.]]<br>
  * <br>
  *   y = [[  0.,   0.,   0.],<br>
  *        [  0.,   0.,   0.]]<br>
  * <br>
  *   slice_like(x, y) = [[ 1.,  2.,  3.]<br>
  *                       [ 5.,  6.,  7.]]<br>
  *   slice_like(x, y, axes=(0, 1)) = [[ 1.,  2.,  3.]<br>
  *                                    [ 5.,  6.,  7.]]<br>
  *   slice_like(x, y, axes=(0)) = [[ 1.,  2.,  3.,  4.]<br>
  *                                 [ 5.,  6.,  7.,  8.]]<br>
  *   slice_like(x, y, axes=(-1)) = [[  1.,   2.,   3.]<br>
  *                                  [  5.,   6.,   7.]<br>
  *                                  [  9.,  10.,  11.]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L568<br>
  * @return org.apache.mxnet.Symbol
  */
def slice_like(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Calculate Smooth L1 Loss(lhs, scalar) by summing<br>
  * <br>
  * .. math::<br>
  * <br>
  *     f(x) =<br>
  *     \begin{cases}<br>
  *     (\sigma x)^2/2,& \text{if }x < 1/\sigma^2\\<br>
  *     |x|-0.5/\sigma^2,& \text{otherwise}<br>
  *     \end{cases}<br>
  * <br>
  * where :math:`x` is an element of the tensor *lhs* and :math:`\sigma` is the scalar.<br>
  * <br>
  * Example::<br>
  * <br>
  *   smooth_l1([1, 2, 3, 4], scalar=1) = [0.5, 1.5, 2.5, 3.5]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_binary_scalar_op_extended.cc:L103<br>
  * @return org.apache.mxnet.Symbol
  */
def smooth_l1(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Applies the softmax function.<br>
  * <br>
  * The resulting array contains elements in the range (0,1) and the elements along the given axis sum up to 1.<br>
  * <br>
  * .. math::<br>
  *    softmax(\mathbf{z})_j = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}<br>
  * <br>
  * for :math:`j = 1, ..., K`<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[ 1.  1.  1.]<br>
  *        [ 1.  1.  1.]]<br>
  * <br>
  *   softmax(x,axis=0) = [[ 0.5  0.5  0.5]<br>
  *                        [ 0.5  0.5  0.5]]<br>
  * <br>
  *   softmax(x,axis=1) = [[ 0.33333334,  0.33333334,  0.33333334],<br>
  *                        [ 0.33333334,  0.33333334,  0.33333334]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/nn/softmax.cc:L95<br>
  * @return org.apache.mxnet.Symbol
  */
def softmax(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Calculate cross entropy of softmax output and one-hot label.<br>
  * <br>
  * - This operator computes the cross entropy in two steps:<br>
  *   - Applies softmax function on the input array.<br>
  *   - Computes and returns the cross entropy loss between the softmax output and the labels.<br>
  * <br>
  * - The softmax function and cross entropy loss is given by:<br>
  * <br>
  *   - Softmax Function:<br>
  * <br>
  *   .. math:: \text{softmax}(x)_i = \frac{exp(x_i)}{\sum_j exp(x_j)}<br>
  * <br>
  *   - Cross Entropy Function:<br>
  * <br>
  *   .. math:: \text{CE(label, output)} = - \sum_i \text{label}_i \log(\text{output}_i)<br>
  * <br>
  * Example::<br>
  * <br>
  *   x = [[1, 2, 3],<br>
  *        [11, 7, 5]]<br>
  * <br>
  *   label = [2, 0]<br>
  * <br>
  *   softmax(x) = [[0.09003057, 0.24472848, 0.66524094],<br>
  *                 [0.97962922, 0.01794253, 0.00242826]]<br>
  * <br>
  *   softmax_cross_entropy(data, label) = - log(0.66524084) - log(0.97962922) = 0.4281871<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/loss_binary_op.cc:L59<br>
  * @return org.apache.mxnet.Symbol
  */
def softmax_cross_entropy(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes softsign of x element-wise.<br>
  * <br>
  * .. math::<br>
  *    y = x / (1 + abs(x))<br>
  * <br>
  * The storage type of ``softsign`` output is always dense<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L148<br>
  * @return org.apache.mxnet.Symbol
  */
def softsign(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns a sorted copy of an input array along the given axis.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 1, 4],<br>
  *        [ 3, 1]]<br>
  * <br>
  *   // sorts along the last axis<br>
  *   sort(x) = [[ 1.,  4.],<br>
  *              [ 1.,  3.]]<br>
  * <br>
  *   // flattens and then sorts<br>
  *   sort(x) = [ 1.,  1.,  3.,  4.]<br>
  * <br>
  *   // sorts along the first axis<br>
  *   sort(x, axis=0) = [[ 1.,  1.],<br>
  *                      [ 3.,  4.]]<br>
  * <br>
  *   // in a descend order<br>
  *   sort(x, is_ascend=0) = [[ 4.,  1.],<br>
  *                           [ 3.,  1.]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/ordering_op.cc:L126<br>
  * @return org.apache.mxnet.Symbol
  */
def sort(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Splits an array along a particular axis into multiple sub-arrays.<br>
  * <br>
  * .. note:: ``SliceChannel`` is deprecated. Use ``split`` instead.<br>
  * <br>
  * **Note** that `num_outputs` should evenly divide the length of the axis<br>
  * along which to split the array.<br>
  * <br>
  * Example::<br>
  * <br>
  *    x  = [[[ 1.]<br>
  *           [ 2.]]<br>
  *          [[ 3.]<br>
  *           [ 4.]]<br>
  *          [[ 5.]<br>
  *           [ 6.]]]<br>
  *    x.shape = (3, 2, 1)<br>
  * <br>
  *    y = split(x, axis=1, num_outputs=2) // a list of 2 arrays with shape (3, 1, 1)<br>
  *    y = [[[ 1.]]<br>
  *         [[ 3.]]<br>
  *         [[ 5.]]]<br>
  * <br>
  *        [[[ 2.]]<br>
  *         [[ 4.]]<br>
  *         [[ 6.]]]<br>
  * <br>
  *    y[0].shape = (3, 1, 1)<br>
  * <br>
  *    z = split(x, axis=0, num_outputs=3) // a list of 3 arrays with shape (1, 2, 1)<br>
  *    z = [[[ 1.]<br>
  *          [ 2.]]]<br>
  * <br>
  *        [[[ 3.]<br>
  *          [ 4.]]]<br>
  * <br>
  *        [[[ 5.]<br>
  *          [ 6.]]]<br>
  * <br>
  *    z[0].shape = (1, 2, 1)<br>
  * <br>
  * `squeeze_axis=1` removes the axis with length 1 from the shapes of the output arrays.<br>
  * **Note** that setting `squeeze_axis` to ``1`` removes axis with length 1 only<br>
  * along the `axis` which it is split.<br>
  * Also `squeeze_axis` can be set to true only if ``input.shape[axis] == num_outputs``.<br>
  * <br>
  * Example::<br>
  * <br>
  *    z = split(x, axis=0, num_outputs=3, squeeze_axis=1) // a list of 3 arrays with shape (2, 1)<br>
  *    z = [[ 1.]<br>
  *         [ 2.]]<br>
  * <br>
  *        [[ 3.]<br>
  *         [ 4.]]<br>
  * <br>
  *        [[ 5.]<br>
  *         [ 6.]]<br>
  *    z[0].shape = (2 ,1 )<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/slice_channel.cc:L107<br>
  * @return org.apache.mxnet.Symbol
  */
def split(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise square-root value of the input.<br>
  * <br>
  * .. math::<br>
  *    \textrm{sqrt}(x) = \sqrt{x}<br>
  * <br>
  * Example::<br>
  * <br>
  *    sqrt([4, 9, 16]) = [2, 3, 4]<br>
  * <br>
  * The storage type of ``sqrt`` output depends upon the input storage type:<br>
  * <br>
  *    - sqrt(default) = default<br>
  *    - sqrt(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L669<br>
  * @return org.apache.mxnet.Symbol
  */
def sqrt(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns element-wise squared value of the input.<br>
  * <br>
  * .. math::<br>
  *    square(x) = x^2<br>
  * <br>
  * Example::<br>
  * <br>
  *    square([2, 3, 4]) = [4, 9, 16]<br>
  * <br>
  * The storage type of ``square`` output depends upon the input storage type:<br>
  * <br>
  *    - square(default) = default<br>
  *    - square(row_sparse) = row_sparse<br>
  *    - square(csr) = csr<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L646<br>
  * @return org.apache.mxnet.Symbol
  */
def square(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Remove single-dimensional entries from the shape of an array.<br>
  * Same behavior of defining the output tensor shape as numpy.squeeze for the most of cases.<br>
  * See the following note for exception.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   data = [[[0], [1], [2]]]<br>
  *   squeeze(data) = [0, 1, 2]<br>
  *   squeeze(data, axis=0) = [[0], [1], [2]]<br>
  *   squeeze(data, axis=2) = [[0, 1, 2]]<br>
  *   squeeze(data, axis=(0, 2)) = [0, 1, 2]<br>
  * <br>
  * .. Note::<br>
  *   The output of this operator will keep at least one dimension not removed. For example,<br>
  *   squeeze([[[4]]]) = [4], while in numpy.squeeze, the output will become a scalar.<br>
  * @return org.apache.mxnet.Symbol
  */
def squeeze(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Join a sequence of arrays along a new axis.<br>
  * <br>
  * The axis parameter specifies the index of the new axis in the dimensions of the<br>
  * result. For example, if axis=0 it will be the first dimension and if axis=-1 it<br>
  * will be the last dimension.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [1, 2]<br>
  *   y = [3, 4]<br>
  * <br>
  *   stack(x, y) = [[1, 2],<br>
  *                  [3, 4]]<br>
  *   stack(x, y, axis=1) = [[1, 3],<br>
  *                          [2, 4]]<br>
  * @return org.apache.mxnet.Symbol
  */
def stack(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Stops gradient computation.<br>
  * <br>
  * Stops the accumulated gradient of the inputs from flowing through this operator<br>
  * in the backward direction. In other words, this operator prevents the contribution<br>
  * of its inputs to be taken into account for computing gradients.<br>
  * <br>
  * Example::<br>
  * <br>
  *   v1 = [1, 2]<br>
  *   v2 = [0, 1]<br>
  *   a = Variable('a')<br>
  *   b = Variable('b')<br>
  *   b_stop_grad = stop_gradient(3 * b)<br>
  *   loss = MakeLoss(b_stop_grad + a)<br>
  * <br>
  *   executor = loss.simple_bind(ctx=cpu(), a=(1,2), b=(1,2))<br>
  *   executor.forward(is_train=True, a=v1, b=v2)<br>
  *   executor.outputs<br>
  *   [ 1.  5.]<br>
  * <br>
  *   executor.backward()<br>
  *   executor.grad_arrays<br>
  *   [ 0.  0.]<br>
  *   [ 1.  1.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L270<br>
  * @return org.apache.mxnet.Symbol
  */
def stop_gradient(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the sum of array elements over given axes.<br>
  * <br>
  * .. Note::<br>
  * <br>
  *   `sum` and `sum_axis` are equivalent.<br>
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.<br>
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.<br>
  * <br>
  * Example::<br>
  * <br>
  *   data = [[[1,2],[2,3],[1,3]],<br>
  *           [[1,4],[4,3],[5,2]],<br>
  *           [[7,1],[7,2],[7,3]]]<br>
  * <br>
  *   sum(data, axis=1)<br>
  *   [[  4.   8.]<br>
  *    [ 10.   9.]<br>
  *    [ 21.   6.]]<br>
  * <br>
  *   sum(data, axis=[1,2])<br>
  *   [ 12.  19.  27.]<br>
  * <br>
  *   data = [[1,2,0],<br>
  *           [3,0,1],<br>
  *           [4,1,0]]<br>
  * <br>
  *   csr = cast_storage(data, 'csr')<br>
  * <br>
  *   sum(csr, axis=0)<br>
  *   [ 8.  3.  1.]<br>
  * <br>
  *   sum(csr, axis=1)<br>
  *   [ 3.  4.  5.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L115<br>
  * @return org.apache.mxnet.Symbol
  */
def sum(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the sum of array elements over given axes.<br>
  * <br>
  * .. Note::<br>
  * <br>
  *   `sum` and `sum_axis` are equivalent.<br>
  *   For ndarray of csr storage type summation along axis 0 and axis 1 is supported.<br>
  *   Setting keepdims or exclude to True will cause a fallback to dense operator.<br>
  * <br>
  * Example::<br>
  * <br>
  *   data = [[[1,2],[2,3],[1,3]],<br>
  *           [[1,4],[4,3],[5,2]],<br>
  *           [[7,1],[7,2],[7,3]]]<br>
  * <br>
  *   sum(data, axis=1)<br>
  *   [[  4.   8.]<br>
  *    [ 10.   9.]<br>
  *    [ 21.   6.]]<br>
  * <br>
  *   sum(data, axis=[1,2])<br>
  *   [ 12.  19.  27.]<br>
  * <br>
  *   data = [[1,2,0],<br>
  *           [3,0,1],<br>
  *           [4,1,0]]<br>
  * <br>
  *   csr = cast_storage(data, 'csr')<br>
  * <br>
  *   sum(csr, axis=0)<br>
  *   [ 8.  3.  1.]<br>
  * <br>
  *   sum(csr, axis=1)<br>
  *   [ 3.  4.  5.]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/broadcast_reduce_op_value.cc:L115<br>
  * @return org.apache.mxnet.Symbol
  */
def sum_axis(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Interchanges two axes of an array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[1, 2, 3]])<br>
  *   swapaxes(x, 0, 1) = [[ 1],<br>
  *                        [ 2],<br>
  *                        [ 3]]<br>
  * <br>
  *   x = [[[ 0, 1],<br>
  *         [ 2, 3]],<br>
  *        [[ 4, 5],<br>
  *         [ 6, 7]]]  // (2,2,2) array<br>
  * <br>
  *  swapaxes(x, 0, 2) = [[[ 0, 4],<br>
  *                        [ 2, 6]],<br>
  *                       [[ 1, 5],<br>
  *                        [ 3, 7]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/swapaxis.cc:L70<br>
  * @return org.apache.mxnet.Symbol
  */
def swapaxes(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Takes elements from an input array along the given axis.<br>
  * <br>
  * This function slices the input array along a particular axis with the provided indices.<br>
  * <br>
  * Given an input array with shape ``(d0, d1, d2)`` and indices with shape ``(i0, i1)``, the output<br>
  * will have shape ``(i0, i1, d1, d2)``, computed by::<br>
  * <br>
  *   output[i,j,:,:] = input[indices[i,j],:,:]<br>
  * <br>
  * .. note::<br>
  *    - `axis`- Only slicing along axis 0 is supported for now.<br>
  *    - `mode`- Only `clip` mode is supported for now.<br>
  * <br>
  * Examples::<br>
  *   x = [4.  5.  6.]<br>
  * <br>
  *   // Trivial case, take the second element along the first axis.<br>
  *   take(x, [1]) = [ 5. ]<br>
  * <br>
  *   x = [[ 1.,  2.],<br>
  *        [ 3.,  4.],<br>
  *        [ 5.,  6.]]<br>
  * <br>
  *   // In this case we will get rows 0 and 1, then 1 and 2. Along axis 0<br>
  *   take(x, [[0,1],[1,2]]) = [[[ 1.,  2.],<br>
  *                              [ 3.,  4.]],<br>
  * <br>
  *                             [[ 3.,  4.],<br>
  *                              [ 5.,  6.]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/indexing_op.cc:L389<br>
  * @return org.apache.mxnet.Symbol
  */
def take(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Computes the element-wise tangent of the input array.<br>
  * <br>
  * The input should be in radians (:math:`2\pi` rad equals 360 degrees).<br>
  * <br>
  * .. math::<br>
  *    tan([0, \pi/4, \pi/2]) = [0, 1, -inf]<br>
  * <br>
  * The storage type of ``tan`` output depends upon the input storage type:<br>
  * <br>
  *    - tan(default) = default<br>
  *    - tan(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L83<br>
  * @return org.apache.mxnet.Symbol
  */
def tan(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the hyperbolic tangent of the input array, computed element-wise.<br>
  * <br>
  * .. math::<br>
  *    tanh(x) = sinh(x) / cosh(x)<br>
  * <br>
  * The storage type of ``tanh`` output depends upon the input storage type:<br>
  * <br>
  *    - tanh(default) = default<br>
  *    - tanh(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_trig.cc:L234<br>
  * @return org.apache.mxnet.Symbol
  */
def tanh(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Repeats the whole array multiple times.<br>
  * <br>
  * If ``reps`` has length *d*, and input array has dimension of *n*. There are<br>
  * three cases:<br>
  * <br>
  * - **n=d**. Repeat *i*-th dimension of the input by ``reps[i]`` times::<br>
  * <br>
  *     x = [[1, 2],<br>
  *          [3, 4]]<br>
  * <br>
  *     tile(x, reps=(2,3)) = [[ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.],<br>
  *                            [ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                            [ 3.,  4.,  3.,  4.,  3.,  4.]]<br>
  * <br>
  * - **n>d**. ``reps`` is promoted to length *n* by pre-pending 1's to it. Thus for<br>
  *   an input shape ``(2,3)``, ``repos=(2,)`` is treated as ``(1,2)``::<br>
  * <br>
  * <br>
  *     tile(x, reps=(2,)) = [[ 1.,  2.,  1.,  2.],<br>
  *                           [ 3.,  4.,  3.,  4.]]<br>
  * <br>
  * - **n<d**. The input is promoted to be d-dimensional by prepending new axes. So a<br>
  *   shape ``(2,2)`` array is promoted to ``(1,2,2)`` for 3-D replication::<br>
  * <br>
  *     tile(x, reps=(2,2,3)) = [[[ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],<br>
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]],<br>
  * <br>
  *                              [[ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.],<br>
  *                               [ 1.,  2.,  1.,  2.,  1.,  2.],<br>
  *                               [ 3.,  4.,  3.,  4.,  3.,  4.]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L751<br>
  * @return org.apache.mxnet.Symbol
  */
def tile(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Returns the top *k* elements in an input array along the given axis.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 0.3,  0.2,  0.4],<br>
  *        [ 0.1,  0.3,  0.2]]<br>
  * <br>
  *   // returns an index of the largest element on last axis<br>
  *   topk(x) = [[ 2.],<br>
  *              [ 1.]]<br>
  * <br>
  *   // returns the value of top-2 largest elements on last axis<br>
  *   topk(x, ret_typ='value', k=2) = [[ 0.4,  0.3],<br>
  *                                    [ 0.3,  0.2]]<br>
  * <br>
  *   // returns the value of top-2 smallest elements on last axis<br>
  *   topk(x, ret_typ='value', k=2, is_ascend=1) = [[ 0.2 ,  0.3],<br>
  *                                                [ 0.1 ,  0.2]]<br>
  * <br>
  *   // returns the value of top-2 largest elements on axis 0<br>
  *   topk(x, axis=0, ret_typ='value', k=2) = [[ 0.3,  0.3,  0.4],<br>
  *                                            [ 0.1,  0.2,  0.2]]<br>
  * <br>
  *   // flattens and then returns list of both values and indices<br>
  *   topk(x, ret_typ='both', k=2) = [[[ 0.4,  0.3], [ 0.3,  0.2]] ,  [[ 2.,  0.], [ 1.,  2.]]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/ordering_op.cc:L63<br>
  * @return org.apache.mxnet.Symbol
  */
def topk(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Permutes the dimensions of an array.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 1, 2],<br>
  *        [ 3, 4]]<br>
  * <br>
  *   transpose(x) = [[ 1.,  3.],<br>
  *                   [ 2.,  4.]]<br>
  * <br>
  *   x = [[[ 1.,  2.],<br>
  *         [ 3.,  4.]],<br>
  * <br>
  *        [[ 5.,  6.],<br>
  *         [ 7.,  8.]]]<br>
  * <br>
  *   transpose(x) = [[[ 1.,  5.],<br>
  *                    [ 3.,  7.]],<br>
  * <br>
  *                   [[ 2.,  6.],<br>
  *                    [ 4.,  8.]]]<br>
  * <br>
  *   transpose(x, axes=(1,0,2)) = [[[ 1.,  2.],<br>
  *                                  [ 5.,  6.]],<br>
  * <br>
  *                                 [[ 3.,  4.],<br>
  *                                  [ 7.,  8.]]]<br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/matrix_op.cc:L310<br>
  * @return org.apache.mxnet.Symbol
  */
def transpose(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Return the element-wise truncated value of the input.<br>
  * <br>
  * The truncated value of the scalar x is the nearest integer i which is closer to<br>
  * zero than x is. In short, the fractional part of the signed number x is discarded.<br>
  * <br>
  * Example::<br>
  * <br>
  *    trunc([-2.1, -1.9, 1.5, 1.9, 2.1]) = [-2., -1.,  1.,  1.,  2.]<br>
  * <br>
  * The storage type of ``trunc`` output depends upon the input storage type:<br>
  * <br>
  *    - trunc(default) = default<br>
  *    - trunc(row_sparse) = row_sparse<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/elemwise_unary_op_basic.cc:L607<br>
  * @return org.apache.mxnet.Symbol
  */
def trunc(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Draw random samples from a uniform distribution.<br>
  * <br>
  * .. note:: The existing alias ``uniform`` is deprecated.<br>
  * <br>
  * Samples are uniformly distributed over the half-open interval *[low, high)*<br>
  * (includes *low*, but excludes *high*).<br>
  * <br>
  * Example::<br>
  * <br>
  *    uniform(low=0, high=1, shape=(2,2)) = [[ 0.60276335,  0.85794562],<br>
  *                                           [ 0.54488319,  0.84725171]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/random/sample_op.cc:L66<br>
  * @return org.apache.mxnet.Symbol
  */
def uniform(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix.<br>
  * <br>
  * Examples::<br>
  * <br>
  *    A = [22,41,37]<br>
  *    unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/ravel.cc:L65<br>
  * @return org.apache.mxnet.Symbol
  */
def unravel_index(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Return the elements, either from x or y, depending on the condition.<br>
  * <br>
  * Given three ndarrays, condition, x, and y, return an ndarray with the elements from x or y,<br>
  * depending on the elements from condition are true or false. x and y must have the same shape.<br>
  * If condition has the same shape as x, each element in the output array is from x if the<br>
  * corresponding element in the condition is true, and from y if false.<br>
  * <br>
  * If condition does not have the same shape as x, it must be a 1D array whose size is<br>
  * the same as x's first dimension size. Each row of the output array is from x's row<br>
  * if the corresponding element from condition is true, and from y's row if false.<br>
  * <br>
  * Note that all non-zero values are interpreted as ``True`` in condition.<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[1, 2], [3, 4]]<br>
  *   y = [[5, 6], [7, 8]]<br>
  *   cond = [[0, 1], [-1, 0]]<br>
  * <br>
  *   where(cond, x, y) = [[5, 2], [3, 8]]<br>
  * <br>
  *   csr_cond = cast_storage(cond, 'csr')<br>
  * <br>
  *   where(csr_cond, x, y) = [[5, 2], [3, 8]]<br>
  * <br>
  * <br>
  * <br>
  * Defined in src/operator/tensor/control_flow_op.cc:L57<br>
  * @return org.apache.mxnet.Symbol
  */
def where(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
  /**
  * Return an array of zeros with the same shape, type and storage type<br>
  * as the input array.<br>
  * <br>
  * The storage type of ``zeros_like`` output depends on the storage type of the input<br>
  * <br>
  * - zeros_like(row_sparse) = row_sparse<br>
  * - zeros_like(csr) = csr<br>
  * - zeros_like(default) = default<br>
  * <br>
  * Examples::<br>
  * <br>
  *   x = [[ 1.,  1.,  1.],<br>
  *        [ 1.,  1.,  1.]]<br>
  * <br>
  *   zeros_like(x) = [[ 0.,  0.,  0.],<br>
  *                    [ 0.,  0.,  0.]]<br>
  * @return org.apache.mxnet.Symbol
  */
def zeros_like(name : String = null, attr : Map[String, String] = null)(args : org.apache.mxnet.Symbol*)(kwargs : Map[String, Any] = null): org.apache.mxnet.Symbol
}
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2017 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./convolution-inl.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(ConvolutionParam);

template<>
Operator* CreateOp<cpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  // If 1D convolution, use MXNet implementation
  if (param.kernel.ndim() == 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new ConvolutionOp<cpu, DType>(param);
    })
    return op;
  }
#if MXNET_USE_MKL2017 == 1
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new MKLConvolutionOp<cpu, float>(param);
    case mshadow::kFloat64:
      return new MKLConvolutionOp<cpu, double>(param);
    default:
      break;
    }
  }
#endif
#if MXNET_USE_NNPACK == 1
  const size_t batch_size = (*in_shape)[0][0];
  if ((param.dilate[0] == 1 && param.dilate[1] == 1)
      && param.kernel.ndim() == 2 && (!param.no_bias)
      && param.num_group == 1 && (batch_size == 1 ||
      ((batch_size > 1) && (param.stride[0] == 1) &&
      (param.stride[1] == 1)))) {
    switch (dtype) {
    case mshadow::kFloat32:
      return new NNPACKConvolutionOp<cpu, float>(param);
    default:
      break;
    }
  }
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(Convolution, ConvolutionProp)
.describe(R"code(Compute *N*-D convolution on *(N+2)*-D input.

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
width)*. We can choose other layouts such as *NHWC*.

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

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(ConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

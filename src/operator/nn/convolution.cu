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
 * \file convolution.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/

#include "./convolution-inl.h"
#include <vector>
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_convolution-inl.h"
#endif  // MXNET_USE_CUDNN

#include "./depthwise_convolution-inl.h"

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(ConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;

  // depth wise conv
  if (param.num_filter == param.num_group &&
      param.layout.value() == mshadow::kNCHW &&
      param.num_filter == (*in_shape)[conv::kData][1] &&
      param.kernel.ndim() == 2 &&
      param.dilate == mshadow::Shape2(1, 1) &&
      dtype == mshadow::kFloat32) {
    op = new DepthwiseConvolutionOp<float>(param, *in_shape, *out_shape);
    return op;
  }

#if MXNET_USE_CUDNN == 1
  // On fp16-I/O instances, use fp32 compute (i.e. pseudo-fp16).
  int compute_type = (dtype == mshadow::kFloat16) ? mshadow::kFloat32 : dtype;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.cudnn_off) {
      op = new ConvolutionOp<gpu, DType>(param);
    } else if (!CuDNNConvolutionOp<DType>::Supports(param, compute_type, compute_type, ctx)) {
      LOG(WARNING) << "This convolution is not supported by cudnn, MXNET convolution is applied.";
      op = new ConvolutionOp<gpu, DType>(param);
    } else {
      op = new CuDNNConvolutionOp<DType>(param, compute_type, compute_type,
                                         *in_shape, *out_shape, ctx);
    }
  })
#else
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ConvolutionOp<gpu, DType>(param);
  })
#endif  // MXNET_USE_CUDNN
  return op;
}

}  // namespace op
}  // namespace mxnet


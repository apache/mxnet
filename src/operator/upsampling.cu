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
 * Copyright (c) 2015 by Contributors
 * \file upsampling_nearest.cc
 * \brief
 * \author Bing Xu
*/

#include "./deconvolution-inl.h"
#include "./upsampling-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(UpSamplingParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (param.sample_type == up_enum::kNearest) {
      op = new UpSamplingNearestOp<gpu, DType>(param);
    } else if (param.sample_type == up_enum::kBilinear) {
      DeconvolutionParam p = DeconvolutionParam();
      int kernel = 2 * param.scale - param.scale % 2;
      int stride = param.scale;
      int pad = static_cast<int>(ceil((param.scale - 1) / 2.));
      p.workspace = param.workspace;
      p.num_group = param.num_filter;
      p.num_filter = param.num_filter;
      p.no_bias =  true;
      int shape[] = {1, 1};
      p.dilate = TShape(shape, shape + 2);
      shape[0] = shape[1] = kernel;
      p.kernel = TShape(shape, shape + 2);
      shape[0] = shape[1] = stride;
      p.stride = TShape(shape, shape + 2);
      shape[0] = shape[1] = pad;
      p.pad = TShape(shape, shape + 2);
      op = new DeconvolutionOp<gpu, DType>(p);
    } else {
      LOG(FATAL) << "Unknown sample type";
    }
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

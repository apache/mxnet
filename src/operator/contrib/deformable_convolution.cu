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
 * Copyright (c) 2017 Microsoft
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file deformable_convolution.cu
 * \brief
 * \author Yuwen Xiong, Haozhi Qi, Jifeng Dai
*/

#include "./deformable_convolution-inl.h"
#include <vector>

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(DeformableConvolutionParam param, int dtype,
    std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new DeformableConvolutionOp<gpu, DType>(param);
    })
      return op;
  }

}  // namespace op
}  // namespace mxnet


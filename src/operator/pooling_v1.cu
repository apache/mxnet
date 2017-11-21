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
 * \file pooling_v1.cu
 * \brief
 * \author Bing Xu
*/
#include <vector>
#include "./pooling_v1-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(PoolingV1Param param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.pool_type) {
      case pool_v1_enum::kMaxPooling:
        op = new PoolingV1Op<gpu, mshadow::red::maximum, DType>(param);
        break;
      case pool_v1_enum::kAvgPooling:
        op = new PoolingV1Op<gpu, mshadow::red::sum, DType>(param);
        break;
      case pool_v1_enum::kSumPooling:
        op = new PoolingV1Op<gpu, mshadow::red::sum, DType>(param);
        break;
      default:
        LOG(FATAL) << "unknown pooling type";
        return NULL;
    }
  });
  return op;
}

}  // namespace op
}  // namespace mxnet


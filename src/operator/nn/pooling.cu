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
 * \file pooling.cu
 * \brief
 * \author Bing Xu, Jun Wu
*/
#include <vector>
#include "./pooling-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_pooling-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(PoolingParam param, int dtype) {
  Operator *op = NULL;
#if MXNET_USE_CUDNN == 1
  if (!param.cudnn_off && param.kernel.ndim() > 1) {
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      switch (param.pool_type) {
        case pool_enum::kMaxPooling:
          op = new CuDNNPoolingOp<DType>(param);
          break;
        case pool_enum::kAvgPooling:
          op = new CuDNNPoolingOp<DType>(param);
          break;
        case pool_enum::kSumPooling:
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
          break;
      }
    });
  }
  if (op) return op;
#endif  // MXNET_USE_CUDNN
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      op = new PoolingOp<gpu, DType>(param);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
  return op;
}

}  // namespace op
}  // namespace mxnet

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
 * \author Bing Xu, Jun Wu, Da Zheng
*/
#include <vector>
#include "./pooling-inl.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_pooling-inl.h"
#endif  // MXNET_USE_CUDNN

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1
template<typename DType>
static CuDNNPoolingOp<DType> &GetCuDNNPoolingOp(const PoolingParam &param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuDNNPoolingOp<DType> op;
#else
  static MX_THREAD_LOCAL CuDNNPoolingOp<DType> op;
#endif
  op.Init(param);
  return op;
}
#endif

template<>
void PoolingCompute<gpu>(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);

#if MXNET_USE_CUDNN == 1
  if (!param.cudnn_off && param.kernel.ndim() > 1) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      switch (param.pool_type) {
        case pool_enum::kMaxPooling:
        case pool_enum::kAvgPooling:
          GetCuDNNPoolingOp<DType>(param).Forward(ctx, inputs[0], req[0], outputs[0]);
          return;
        case pool_enum::kSumPooling:
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
          break;
      }
    });
  }
#endif  // MXNET_USE_CUDNN

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      PoolingForward<gpu, DType>(ctx, param, inputs[0], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
}

template<>
void PoolingGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);

#if MXNET_USE_CUDNN == 1
  if (!param.cudnn_off && param.kernel.ndim() > 1) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      switch (param.pool_type) {
        case pool_enum::kMaxPooling:
        case pool_enum::kAvgPooling:
          GetCuDNNPoolingOp<DType>(param).Backward(ctx,
              inputs[0], inputs[1], inputs[2], req[0], outputs[0]);
          return;
        case pool_enum::kSumPooling:
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
          break;
      }
    });
  }
#endif  // MXNET_USE_CUDNN

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      PoolingBackward<gpu, DType>(ctx, param, inputs[0],
          inputs[1], inputs[2], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
}

NNVM_REGISTER_OP(Pooling)
.set_attr<FCompute>("FCompute<gpu>", PoolingCompute<gpu>);

NNVM_REGISTER_OP(_backward_Pooling)
.set_attr<FCompute>("FCompute<gpu>", PoolingGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

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
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), GetNumOutputs(param));

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
        case pool_enum::kLpPooling:
          LOG(WARNING) << "Lp pooling is not supported by cudnn, MXNet lp pooling is applied.";
          break;
      }
    });
  }
#endif  // MXNET_USE_CUDNN

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type
        || pool_enum::kLpPooling == param.pool_type) {
      PoolingOp<gpu, DType> op;
      op.Init(param);
      op.Forward(ctx, inputs[0], req[0], outputs[0]);
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
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), GetNumBackInputs(param));
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  off_t ograd_idx, in_data_idx, out_data_idx;
  // When MKLDNN is enabled, the input data may contains arrays for workspace.
  if (GetNumBackInputs(param) == 5) {
    ograd_idx = 0;
    in_data_idx = 2;
    out_data_idx = 3;
  } else {
    ograd_idx = 0;
    in_data_idx = 1;
    out_data_idx = 2;
  }

#if MXNET_USE_CUDNN == 1
  if (!param.cudnn_off && param.kernel.ndim() > 1) {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      switch (param.pool_type) {
        case pool_enum::kMaxPooling:
        case pool_enum::kAvgPooling:
          GetCuDNNPoolingOp<DType>(param).Backward(ctx, inputs[ograd_idx],
                                                   inputs[in_data_idx], inputs[out_data_idx],
                                                   req[0], outputs[0]);
          return;
        case pool_enum::kSumPooling:
          LOG(WARNING) << "Sum pooling is not supported by cudnn, MXNet sum pooling is applied.";
          break;
        case pool_enum::kLpPooling:
          LOG(WARNING) << "Lp pooling is not supported by cudnn, MXNet Lp pooling is applied.";
          break;
      }
    });
  }
#endif  // MXNET_USE_CUDNN

  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type
        || pool_enum::kLpPooling == param.pool_type) {
      PoolingOp<gpu, DType> op;
      op.Init(param);
      op.Backward(ctx, inputs[ograd_idx], inputs[in_data_idx],
                  inputs[out_data_idx], req[0], outputs[0]);
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

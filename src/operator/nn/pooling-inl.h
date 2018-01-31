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
 * \file pooling-inl.h
 * \brief
 * \author Bing Xu, Jun Wu, Da Zheng
*/

#ifndef MXNET_OPERATOR_NN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "./pool.h"

namespace mxnet {
namespace op {

struct PoolingParam : public dmlc::Parameter<PoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pool_type;
  int pooling_convention;
  bool global_pool;
  bool cudnn_off;
  DMLC_DECLARE_PARAMETER(PoolingParam) {
    DMLC_DECLARE_FIELD(global_pool).set_default(false)
    .describe("Ignore kernel size, do global pooling based on current input feature map. ");

    DMLC_DECLARE_FIELD(cudnn_off).set_default(false)
    .describe("Turn off cudnn pooling and use MXNet pooling operator. ");

    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("Pooling kernel size: (y, x) or (d, y, x)");

    DMLC_DECLARE_FIELD(pool_type)
    .add_enum("max", pool_enum::kMaxPooling)
    .add_enum("avg", pool_enum::kAvgPooling)
    .add_enum("sum", pool_enum::kSumPooling)
    .describe("Pooling type to be applied.");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(pool_enum::kValid)
    .add_enum("full", pool_enum::kFull)
    .add_enum("valid", pool_enum::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("Stride: for pooling (y, x) or (d, y, x). Defaults to 1 for each dimension.");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Pad for pooling: (y, x) or (d, y, x). Defaults to no padding.");
  }

  bool operator==(const PoolingParam& other) const {
    return this->kernel             == other.kernel &&
           this->stride             == other.stride &&
           this->pad                == other.pad &&
           this->pool_type          == other.pool_type &&
           this->pooling_convention == other.pooling_convention &&
           this->global_pool        == other.global_pool &&
           this->cudnn_off          == other.cudnn_off;
  }
};

}  // namespace op
}  // namespace mxnet

namespace std {
template<>
struct hash<mxnet::op::PoolingParam> {
  size_t operator()(const mxnet::op::PoolingParam& val) {
    size_t ret = 0;
    ret = dmlc::HashCombine(ret, val.kernel);
    ret = dmlc::HashCombine(ret, val.stride);
    ret = dmlc::HashCombine(ret, val.pad);
    ret = dmlc::HashCombine(ret, val.pool_type);
    ret = dmlc::HashCombine(ret, val.pooling_convention);
    ret = dmlc::HashCombine(ret, val.global_pool);
    ret = dmlc::HashCombine(ret, val.cudnn_off);
    return ret;
  }
};
}  // namespace std

namespace mxnet {
namespace op {

/*
 * When MKLDNN is enabled, we might want 2 outputs instead of one inputs, which
 * also changes the number of inputs for backward.
 */
int GetNumOutputs(const PoolingParam &param);
int GetNumBackInputs(const PoolingParam &param);

template<typename xpu, typename DType>
void PoolingForward(const OpContext& ctx, const PoolingParam &param,
                    const TBlob& in_data, const OpReqType& req,
                    const TBlob& out_data) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TShape& ishape = in_data.shape_;

  pool(s, in_data.dptr<DType>(), in_data.shape_, out_data.shape_,
       param.global_pool?
       TShape(ishape.data()+ishape.ndim()-param.kernel.ndim(), ishape.data()+ishape.ndim())
       : param.kernel,
       param.pad,
       param.global_pool? TShape(param.kernel.ndim()) : param.stride,
       param.pool_type, req, out_data.dptr<DType>());
}

template<typename xpu, typename DType>
void PoolingBackward(const OpContext& ctx, const PoolingParam &param,
                     const TBlob& out_grad, const TBlob& in_data,
                     const TBlob& out_data, const OpReqType& req,
                     const TBlob& in_grad) {
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TShape& ishape = in_data.shape_;

  unpool(s, out_grad.dptr<DType>(), in_data.dptr<DType>(), out_data.dptr<DType>(),
         in_grad.shape_, out_grad.shape_,
         param.global_pool?
         TShape(ishape.data()+ishape.ndim()-param.kernel.ndim(), ishape.data()+ishape.ndim())
         : param.kernel,
         param.pad,
         param.global_pool? TShape(param.kernel.ndim()) : param.stride,
         param.pool_type, req, in_grad.dptr<DType>());
}

template<typename xpu>
void PoolingCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), GetNumOutputs(param));
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      PoolingForward<xpu, DType>(ctx, param, inputs[0], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
}

template<typename xpu>
void PoolingGradCompute(const nnvm::NodeAttrs& attrs,
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
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      PoolingBackward<xpu, DType>(ctx, param, inputs[ograd_idx],
                                  inputs[in_data_idx], inputs[out_data_idx],
                                  req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOLING_INL_H_

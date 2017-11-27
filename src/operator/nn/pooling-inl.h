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
};

template<typename xpu, typename DType>
class PoolingOp {
 public:
  void Init(PoolingParam p) {
    this->param_ = p;
  }

  void Forward(const OpContext& ctx, const TBlob& in_data,
               const OpReqType& req, const TBlob& out_data) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data.shape_;

    pool(s, in_data.dptr<DType>(), in_data.shape_, out_data.shape_,
         param_.global_pool?
           TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
           : param_.kernel,
         param_.pad,
         param_.global_pool? TShape(param_.kernel.ndim()) : param_.stride,
         param_.pool_type, req, out_data.dptr<DType>());
  }

  void Backward(const OpContext& ctx, const TBlob& out_grad,
                const TBlob& in_data, const TBlob& out_data,
                const OpReqType& req, const TBlob& in_grad) {
    using namespace mshadow;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    const TShape& ishape = in_data.shape_;

    unpool(s, out_grad.dptr<DType>(), in_data.dptr<DType>(), out_data.dptr<DType>(),
           in_grad.shape_, out_grad.shape_,
           param_.global_pool?
             TShape(ishape.data()+ishape.ndim()-param_.kernel.ndim(), ishape.data()+ishape.ndim())
             : param_.kernel,
           param_.pad,
           param_.global_pool? TShape(param_.kernel.ndim()) : param_.stride,
           param_.pool_type, req, in_grad.dptr<DType>());
  }

 private:
  PoolingParam param_;
};  // class PoolingOp

template<typename xpu, typename DType>
PoolingOp<xpu, DType> &GetPoolingOp(const PoolingParam &param) {
  static thread_local PoolingOp<xpu, DType> op;
  op.Init(param);
  return op;
}

template<typename xpu>
void PoolingCompute(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<TBlob>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      GetPoolingOp<xpu, DType>(param).Forward(ctx, inputs[0], req[0], outputs[0]);
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
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    if (pool_enum::kMaxPooling == param.pool_type
        || pool_enum::kAvgPooling == param.pool_type
        || pool_enum::kSumPooling == param.pool_type) {
      GetPoolingOp<xpu, DType>(param).Backward(ctx,
          inputs[0], inputs[1], inputs[2], req[0], outputs[0]);
    } else {
      LOG(FATAL) << "unknown pooling type";
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_POOLING_INL_H_

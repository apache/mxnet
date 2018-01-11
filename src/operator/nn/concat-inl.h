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
 * \file concat-inl.h
 * \brief
 * \author Bing Xu
*/
#ifndef MXNET_OPERATOR_CONCAT_INL_H_
#define MXNET_OPERATOR_CONCAT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include "../operator_common.h"
#include "../channel_op_common.h"
#include "../tensor/broadcast_reduce_op.h"

namespace mxnet {
namespace op {

namespace concat_enum {
enum ConcatOpInputs {kData0, kData1, kData2, kData3, kData4};
enum ConcatOpOutputs {kOut};
}  // namespace concat_enum

struct ConcatParam : public dmlc::Parameter<ConcatParam> {
  int num_args;
  int dim;
  DMLC_DECLARE_PARAMETER(ConcatParam) {
    DMLC_DECLARE_FIELD(num_args).set_lower_bound(1)
    .describe("Number of inputs to be concated.");
    DMLC_DECLARE_FIELD(dim).set_default(1)
    .describe("the dimension to be concated.");
  }
};  // struct ConcatParam

template<typename xpu, typename DType>
class ConcatOp {
 public:
  void Init(const ConcatParam &param) {
    this->size_ = param.num_args;
    this->dimension_ = param.dim;
  }

  void Forward(const OpContext &ctx,
               const std::vector<TBlob> &in_data,
               const std::vector<OpReqType> &req,
               const std::vector<TBlob> &out_data) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), size_);
    CHECK_EQ(out_data.size(), 1U);
    int axis = CheckAxis(dimension_, in_data[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3, DType> > data(size_);
    Tensor<xpu, 3, DType> out;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= out_data[concat_enum::kOut].shape_[i];
    }
    for (int i = axis + 1; i < out_data[concat_enum::kOut].ndim(); ++i) {
      trailing *= out_data[concat_enum::kOut].shape_[i];
    }
    size_t mid = out_data[concat_enum::kOut].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    out = out_data[concat_enum::kOut].get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_data[i].shape_[axis], trailing);
      data[i] = in_data[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Concatenate(data, &out, 1, req[concat_enum::kOut]);
  }

  void Backward(const OpContext &ctx,
                const std::vector<TBlob> &out_grad,
                const std::vector<OpReqType> &req,
                const std::vector<TBlob> &in_grad) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_grad.size(), static_cast<size_t>(size_));
    int axis = CheckAxis(dimension_, out_grad[concat_enum::kData0].ndim());
    Stream<xpu> *s = ctx.get_stream<xpu>();
    std::vector<Tensor<xpu, 3, DType> > grad_in(size_);
    Tensor<xpu, 3, DType> grad;
    size_t leading = 1, trailing = 1;
    for (int i = 0; i < axis; ++i) {
      leading *= out_grad[concat_enum::kOut].shape_[i];
    }
    for (int i = axis + 1; i < out_grad[concat_enum::kOut].ndim(); ++i) {
      trailing *= out_grad[concat_enum::kOut].shape_[i];
    }
    size_t mid = out_grad[concat_enum::kOut].shape_[axis];
    Shape<3> oshape = Shape3(leading, mid, trailing);
    grad = out_grad[concat_enum::kOut].get_with_shape<xpu, 3, DType>(oshape, s);

    for (int i = 0; i < size_; ++i) {
      Shape<3> dshape = Shape3(leading, in_grad[i].shape_[axis], trailing);
      grad_in[i] = in_grad[i].get_with_shape<xpu, 3, DType>(dshape, s);
    }
    Split(grad, &grad_in, 1, req);
  }

 private:
  int size_;
  int dimension_;
};  // class ConcatOp

template<typename xpu>
void ConcatCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                   const std::vector<TBlob>& inputs,
                   const std::vector<OpReqType>& req,
                   const std::vector<TBlob>& outputs) {
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kData0].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Forward(ctx, inputs, req, outputs);
  });
}

template<typename xpu>
void ConcatGradCompute(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  const ConcatParam& param = nnvm::get<ConcatParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(inputs[concat_enum::kOut].type_flag_, DType, {
    ConcatOp<xpu, DType> op;
    op.Init(param);
    op.Backward(ctx, inputs, req, outputs);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONCAT_INL_H_

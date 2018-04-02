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
 * Copyright (c) 2016 by Contributors
 * \file pad-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_PAD_INL_H_
#define MXNET_OPERATOR_PAD_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace pad_enum {
enum PadOpInputs { kData };
enum PadOpType { kConstant, kEdge, kReflect };
enum PadOpOutputs { kOut };
}

struct PadParam : public dmlc::Parameter<PadParam> {
  int mode;
  double constant_value;
  TShape pad_width;
  DMLC_DECLARE_PARAMETER(PadParam) {
    DMLC_DECLARE_FIELD(mode)
        .add_enum("constant", pad_enum::kConstant)
        .add_enum("edge", pad_enum::kEdge)
        .add_enum("reflect", pad_enum::kReflect)
        .describe(
            "Padding type to use."
            " \"constant\" pads with `constant_value`"
            " \"edge\" pads using the edge values of the input array"
            " \"reflect\" pads by reflecting values with respect to the edges.");

    DMLC_DECLARE_FIELD(pad_width).describe(
        "Widths of the padding regions applied to the edges of each axis. "
        "It is a tuple of integer padding widths for each axis of the format "
        "``(before_1, after_1, ... , before_N, after_N)``. "
        "It should be of length ``2*N`` where ``N`` is the number of dimensions of the array."
        "This is equivalent to pad_width in numpy.pad, but flattened.");
    DMLC_DECLARE_FIELD(constant_value)
        .describe("The value used for padding when `mode` is \"constant\".")
        .set_default(0.0);
  }
};

template <typename xpu, typename DType>
class PadOp : public Operator {
 public:
  explicit PadOp(PadParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Get any size input + output into required form
    int rank = in_data[pad_enum::kData].ndim();
    auto pad = param_.pad_width;
    DType constant_value = param_.constant_value;
    // TODO(nswamy@): update the documentation and log below when support is added for more than
    // 4D/5D arrays and not requiring higher dimensions to be zero.
    switch (rank) {
      case 4:
        {
          Tensor<xpu, 4, DType> data =
            in_data[pad_enum::kData].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> out =
            out_data[pad_enum::kOut].get<xpu, 4, DType>(s);
          pad_image(out, data, param_.pad_width, param_.mode, constant_value);
          break;
        }
      case 5:
        {
          Tensor<xpu, 5, DType> data =
            in_data[pad_enum::kData].get<xpu, 5, DType>(s);
          Tensor<xpu, 5, DType> out =
            out_data[pad_enum::kOut].get<xpu, 5, DType>(s);
          pad_image(out, data, param_.pad_width, param_.mode, constant_value);
          break;
        }
      default:
        LOG(FATAL) << "Attempted to run forward pass "
                        "with input dimensions other than 4 or 5.";
    }
    // Assign(out, req[pad_enum::kOut], F<mshadow_op::identity>(data));
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // Get any size input + output into required form
    auto pad = param_.pad_width;
    int rank = in_grad[pad_enum::kData].ndim();
    switch (rank) {
      case 4:
        {
          Tensor<xpu, 4, DType> in =
            in_grad[pad_enum::kData].get<xpu, 4, DType>(s);
          Tensor<xpu, 4, DType> out =
            out_grad[pad_enum::kOut].get<xpu, 4, DType>(s);
          if (req[pad_enum::kData] == kWriteTo) in = 0.0f;
          pad_image_grad(in, out, param_.pad_width, param_.mode);
          break;
        }
      case 5:
        {
          Tensor<xpu, 5, DType> in =
            in_grad[pad_enum::kData].get<xpu, 5, DType>(s);
          Tensor<xpu, 5, DType> out =
            out_grad[pad_enum::kOut].get<xpu, 5, DType>(s);
          if (req[pad_enum::kData] == kWriteTo) in = 0.0f;
          pad_image_grad(in, out, param_.pad_width, param_.mode);
          break;
        }
      default:
        LOG(FATAL) << "Attempted to run backward pass "
                        "with input dimensions other than 4 or 5.";
    }
  }

 private:
  PadParam param_;
};  // class PadOp

template <typename xpu>
Operator *CreateOp(PadParam param, int dtype);

#if DMLC_USE_CXX11
class PadProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 1; }

  std::vector<std::string> ListArguments() const override { return {"data"}; }

  std::vector<std::string> ListOutputs() const override { return {"output"}; }

  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1U) << "Can only be one input to symbol.";

    const TShape &dshape = (*in_shape)[pad_enum::kData];

    auto rank = dshape.ndim();
    auto pad = param_.pad_width;
    auto pad_spec_len = param_.pad_width.ndim();

    if (rank == 0) return false;
    if ((rank != 4) && (rank != 5)) {
      LOG(FATAL) << "Current implementation only supports 4-D or 5-D input.";
    }
    if ((pad[0] != 0) || (pad[1] != 0) || (pad[2] != 0) || (pad[3] != 0)) {
      LOG(FATAL) << "Current implementation expects padding on the first two axes to be zero.";
    }
    if ((2*rank) != pad_spec_len) {
      LOG(FATAL) << "Input shape vs padding spec mismatch.";
    }
    if (param_.mode == pad_enum::kReflect) {
      auto size = dshape.data();
      if ((pad[4] >= size[2]) || (pad[5] >= size[2]) ||
            (pad[6] >= size[3]) || (pad[7] >= size[3])) {
        LOG(FATAL) << "Current implementation of reflection padding "
                        "only supports padding sizes smaller than the input size.";
      }
    }
    TShape oshape = dshape;
    for (size_t i = 0; i < dshape.ndim(); ++i) {
      oshape[i] =
          param_.pad_width[2 * i] + param_.pad_width[2 * i + 1] + dshape[i];
    }
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new PadProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "Pad"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[pad_enum::kOut]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  PadParam param_;
};      // class PadProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_PAD_INL_H_

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
 * \file wl_sequence_mask-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_SEQUENCE_MASK_INL_H_
#define MXNET_OPERATOR_SEQUENCE_MASK_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace seq_mask {
enum SequenceMaskOpInputs { kData, kSequenceLength };
enum SequenceMaskOpOutputs { kOut };
enum SequenceMaskOpBackResource { kTempSpace };
}

struct SequenceMaskParam : public dmlc::Parameter<SequenceMaskParam> {
  bool use_sequence_length;
  float value;
  int axis;
  DMLC_DECLARE_PARAMETER(SequenceMaskParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter "
            "`sequence_length` "
            "to specify variable length sequence");
    DMLC_DECLARE_FIELD(value).set_default(0.).describe(
        "The value to be used as a mask.");
    DMLC_DECLARE_FIELD(axis).set_default(0).describe(
        "The sequence axis. Only values of 0 and 1 are currently supported.");
  }
};

template<typename DType, typename IType>
void SequenceMaskExec(const mshadow::Tensor<cpu, 3, DType> &data,
                  const mshadow::Tensor<cpu, 1, IType> &indices,
                  const OpReqType req, mshadow::Stream<cpu> *const s,
                  int axis, DType val);
#ifdef __CUDACC__
template<typename DType, typename IType>
void SequenceMaskExec(const mshadow::Tensor<gpu, 3, DType> &data,
                  const mshadow::Tensor<gpu, 1, IType> &indices,
                  const OpReqType req, mshadow::Stream<gpu> *const s,
                  int axis, DType val);
#endif

template <typename xpu, typename DType, typename IType>
class SequenceMaskOp : public Operator {
 public:
  explicit SequenceMaskOp(SequenceMaskParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), param_.use_sequence_length ? 2U : 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Get any size input + output into required form
    auto d0 = in_data[seq_mask::kData].size(0);
    auto d1 = in_data[seq_mask::kData].size(1);
    auto dsize = in_data[seq_mask::kData].Size();
    auto rest_size = dsize / (d0 * d1);

    Shape<3> s3 = Shape3(d0, d1, rest_size);
    Tensor<xpu, 3, DType> data =
        in_data[seq_mask::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out =
        out_data[seq_mask::kOut].get_with_shape<xpu, 3, DType>(s3, s);
    // Actual implementation of masking
    Assign(out, req[seq_mask::kOut], F<mshadow_op::identity>(data));
    if (param_.use_sequence_length) {
      Tensor<xpu, 1, IType> indices =
          in_data[seq_mask::kSequenceLength].get<xpu, 1, IType>(s);
      SequenceMaskExec<DType, IType>(out, indices, req[seq_mask::kOut], s,
                   param_.axis, static_cast<DType>(param_.value));
    }
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
    CHECK_EQ(in_data.size(), param_.use_sequence_length ? 2U : 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Get any size input + output into required form
    auto d0 = in_grad[seq_mask::kData].size(0);
    auto d1 = in_grad[seq_mask::kData].size(1);
    auto dsize = in_grad[seq_mask::kData].Size();
    auto rest_size = dsize / (d0 * d1);

    Shape<3> s3 = Shape3(d0, d1, rest_size);
    Tensor<xpu, 3, DType> data_g =
        in_grad[seq_mask::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out_g =
        out_grad[seq_mask::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    // Actual implementation of masking
    if (req[seq_mask::kData] == kNullOp) return;
    if (!param_.use_sequence_length) {
      Assign(data_g, req[seq_mask::kData], F<mshadow_op::identity>(out_g));
    } else {
      Tensor<xpu, 1, IType> indices =
          in_data[seq_mask::kSequenceLength].get<xpu, 1, IType>(s);
      if (req[seq_mask::kData] == kAddTo) {
        Tensor<xpu, 3, DType> out_g_temp =
            ctx.requested[seq_mask::kTempSpace].get_space_typed<xpu, 3, DType>(
                s3, s);
        out_g_temp = F<mshadow_op::identity>(out_g);
        out_g = out_g_temp;
        SequenceMaskExec<DType, IType>(out_g, indices, kWriteInplace, s, param_.axis, DType(0.));
        Assign(data_g, kAddTo, F<mshadow_op::identity>(out_g));
      } else {
        Assign(data_g, req[seq_mask::kData], F<mshadow_op::identity>(out_g));
        SequenceMaskExec<DType, IType>(
          data_g, indices, req[seq_mask::kData], s, param_.axis, DType(0.));
      }
    }
  }

 private:
  SequenceMaskParam param_;
};  // class SequenceMaskOp

template <typename xpu>
Operator *CreateOp(SequenceMaskParam param, int dtype, int itype);

#if DMLC_USE_CXX11
class SequenceMaskProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 1; }

  std::vector<std::string> ListArguments() const override {
    if (param_.use_sequence_length)
      return {"data", "sequence_length"};
    else
      return {"data"};
  }

  std::vector<std::string> ListOutputs() const override { return {"output"}; }

  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs)
      override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape,
                  mxnet::ShapeVector *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), param_.use_sequence_length ? 2U : 1U)
        << "Input:[data, sequence_length]";

    const mxnet::TShape &dshape = (*in_shape)[seq_mask::kData];
    CHECK_GT(dshape.ndim(), 1U)
        << "The data array must be of rank 2 or greater.";
    CHECK((param_.axis == 0) || (param_.axis == 1))
        << "Current implementation expects axis to be 0 or 1.";

    // seq length vector is same as batch size
    int sbatch = param_.axis ? dshape[0] : dshape[1];
    if (param_.use_sequence_length)
      SHAPE_ASSIGN_CHECK(*in_shape, seq_mask::kSequenceLength, Shape1(sbatch));

    const mxnet::TShape &oshape = dshape;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type, std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), param_.use_sequence_length ? 2U : 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (size_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new SequenceMaskProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "SequenceMask"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    if (param_.use_sequence_length)
      return {out_grad[seq_mask::kOut], in_data[seq_mask::kSequenceLength]};
    else
      return {out_grad[seq_mask::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const mxnet::ShapeVector &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<std::pair<int, void *> > BackwardInplaceOption(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data,
      const std::vector<void *> &in_grad) const override {
    return {{out_grad[seq_mask::kOut], in_grad[seq_mask::kData]}};
  }

  std::vector<std::pair<int, void *> > ForwardInplaceOption(
      const std::vector<int> &in_data,
      const std::vector<void *> &out_data) const override {
    return {{in_data[seq_mask::kData], out_data[seq_mask::kOut]}};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return nullptr;
  }

  Operator *CreateOperatorEx(Context ctx, mxnet::ShapeVector *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SequenceMaskParam param_;
};      // class SequenceMaskProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_MASK_INL_H_

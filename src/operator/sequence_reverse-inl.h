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

/*
 * Copyright (c) 2016 by Contributors
 * \file sequence_reverse-inl.h
 * \brief
 * \author Sebastian Bodenstien
 * \author Marek Kolodziej
*/

#ifndef MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_
#define MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "./mshadow_op.h"
#include "./mxnet_op.h"
#include "./operator_common.h"
#include "./sequence_op_common.h"

namespace mxnet {
namespace op {

namespace seq_reverse {
enum SequenceReverseOpInputs { kData, kSequenceLength };
enum SequenceReverseOpOutputs { kOut };
}

struct SequenceReverseParam : public dmlc::Parameter<SequenceReverseParam> {
  bool use_sequence_length;
  int axis;
  DMLC_DECLARE_PARAMETER(SequenceReverseParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter "
            "`sequence_length` "
            "to specify variable length sequence");
    DMLC_DECLARE_FIELD(axis).set_default(0).describe(
        "The sequence axis. Only 0 is currently supported.");
  }
};

struct ReverseKernel {
  template <typename DType>
  MSHADOW_XINLINE static void Map(const int i, DType *const out_data,
                                  const DType *const in_data,
                                  const OpReqType req,
                                  const index_t max_seq_len,
                                  const index_t batch_size,
                                  const index_t other_dim, const index_t numel,
                                  const DType *const indices) {
    for (index_t batch = 0; batch < batch_size; ++batch) {
      const index_t num_seq =
          indices ? static_cast<index_t>(indices[batch]) : max_seq_len;
      const index_t padded_periods = max_seq_len - num_seq;
      // padded part
      if (padded_periods > 0 && i < static_cast<int>(padded_periods)) {
        const int padded_in_offset =
            (i + num_seq) * batch_size * other_dim + batch * other_dim;

        for (index_t j = 0; j < other_dim; ++j) {
          KERNEL_ASSIGN(out_data[padded_in_offset + j], req,
                        in_data[padded_in_offset + j]);
        }
      }
      // unpadded part
      if (i < static_cast<int>(num_seq)) {
        const int in_offset = i * batch_size * other_dim + batch * other_dim;
        const int out_offset =
            numel - (i + 1 + padded_periods) * batch_size * other_dim +
            batch * other_dim;

        for (index_t j = 0; j < other_dim; ++j) {
          KERNEL_ASSIGN(out_data[out_offset + j], req, in_data[in_offset + j]);
        }
      }
    }
  }
};

template <typename xpu, typename DType>
class SequenceReverseOp : public Operator {
 public:
  explicit SequenceReverseOp(SequenceReverseParam p) { this->param_ = p; }
  void sequence_reverse(const mshadow::Tensor<xpu, 3, DType> &data,
                        const mshadow::Tensor<xpu, 3, DType> &out,
                        const OpReqType req, const DType *const indices,
                        mshadow::Stream<xpu> *const s) {
    using namespace mshadow;
    using namespace mshadow::expr;

    const index_t max_seq_len = data.size(0);
    const index_t batch_size = data.size(1);
    const index_t other_dim = data.size(2);
    const index_t tensor_numel = data.shape_.Size();

    mxnet_op::Kernel<ReverseKernel, xpu>::Launch(
        s, max_seq_len, out.dptr_, data.dptr_, req, max_seq_len, batch_size,
        other_dim, tensor_numel, indices);
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), param_.use_sequence_length ? 2U : 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *const s = ctx.get_stream<xpu>();

    // Get any size input + output into required form
    auto max_seq_len = in_data[seq_reverse::kData].size(0);
    auto n = in_data[seq_reverse::kData].size(1);
    auto total_size = in_data[seq_reverse::kData].Size();
    auto rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);
    Tensor<xpu, 3, DType> data =
        in_data[seq_reverse::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out =
        out_data[seq_reverse::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    const DType *const indices =
        param_.use_sequence_length
            ? in_data[seq_reverse::kSequenceLength].dptr<DType>()
            : nullptr;

    sequence_reverse(data, out, req[seq_reverse::kOut], indices, s);
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
    auto max_seq_len = in_grad[seq_reverse::kData].size(0);
    auto n = in_grad[seq_reverse::kData].size(1);
    auto total_size = in_grad[seq_reverse::kData].Size();
    auto rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);

    Tensor<xpu, 3, DType> data_grad =
        in_grad[seq_reverse::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> output_grad =
        out_grad[seq_reverse::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    const DType *const indices =
        param_.use_sequence_length
            ? in_data[seq_reverse::kSequenceLength].dptr<DType>()
            : nullptr;

    sequence_reverse(output_grad, data_grad, req[seq_reverse::kData], indices,
                     s);
  }

 private:
  SequenceReverseParam param_;
};  // class SequenceReverseOp

template <typename xpu>
Operator *CreateOp(SequenceReverseParam param, int dtype);

#if DMLC_USE_CXX11
class SequenceReverseProp : public OperatorProperty {
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

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), param_.use_sequence_length ? 2U : 1U)
        << "Input:[data, sequence_length]";
    CHECK_EQ(param_.axis, 0) << "Current implementation expects axis to be 0.";

    const TShape &dshape = (*in_shape)[seq_reverse::kData];
    CHECK_GT(dshape.ndim(), 1U)
        << "The data array must be of rank 2 or greater.";
    // seq length vector is same as batch size
    if (param_.use_sequence_length)
      SHAPE_ASSIGN_CHECK(*in_shape, seq_reverse::kSequenceLength,
                         Shape1(dshape[1]));

    const TShape &oshape = dshape;
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type, std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), param_.use_sequence_length ? 2U : 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new SequenceReverseProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "SequenceReverse"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    if (param_.use_sequence_length)
      return {out_grad[seq_reverse::kOut],
              in_data[seq_reverse::kSequenceLength]};
    else
      return {out_grad[seq_reverse::kOut]};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SequenceReverseParam param_;
};      // class SequenceReverseProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_

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
 * \file sequence_last-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/
#ifndef MXNET_OPERATOR_SEQUENCE_LAST_INL_H_
#define MXNET_OPERATOR_SEQUENCE_LAST_INL_H_

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
#include "./operator_common.h"
#include "./sequence_op_common.h"

namespace mxnet {
namespace op {

namespace seq_last {
enum SequenceLastOpInputs { kData, kSequenceLength };
enum SequenceLastOpOutputs { kOut };
}

struct SequenceLastParam : public dmlc::Parameter<SequenceLastParam> {
  bool use_sequence_length;
  uint32_t axis;
  DMLC_DECLARE_PARAMETER(SequenceLastParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter "
            "`sequence_length` "
            "to specify variable length sequence");
    DMLC_DECLARE_FIELD(axis).set_default(0).describe(
        "The sequence axis. Only values of 0 and 1 are current supported.");
  }
};

template <int req>
struct SequenceLastTimewiseKernel {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const DType *idx, int maxseqlen,
                                  mshadow::Shape<2> oshape) {
    auto opos = mxnet_op::unravel(i, oshape);
    int seqpos = static_cast<int>(idx[opos[0]]) - 1;
    int ipos = seqpos * (oshape[0] * oshape[1]) + opos[0] * oshape[1] + opos[1];
    KERNEL_ASSIGN(out[i], req, in[ipos]);
  }
};

template <int req>
struct SequenceLastBatchwiseKernel {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType *in,
                                  const DType *idx, int maxseqlen,
                                  mshadow::Shape<2> oshape) {
    auto opos = mxnet_op::unravel(i, oshape);
    int seqpos = static_cast<int>(idx[opos[0]]) - 1;
    int ipos = opos[0] * (maxseqlen * oshape[1]) + seqpos * oshape[1] + opos[1];
    KERNEL_ASSIGN(out[i], req, in[ipos]);
  }
};

template <typename xpu, typename DType>
class SequenceLastOp : public Operator {
 public:
  explicit SequenceLastOp(SequenceLastParam p) { this->param_ = p; }

  void sequence_last(const mshadow::Tensor<xpu, 3, DType> &data,
                     const mshadow::Tensor<xpu, 2, DType> &out,
                     const mshadow::Tensor<xpu, 1, DType> &indices,
                     const OpReqType req, mshadow::Stream<xpu> *const s) {
    using namespace mshadow;
    using namespace mshadow::expr;

    int out_size = out.size(0) * out.size(1);
    int max_seq_len = data.size(param_.axis);

    MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
      if (param_.axis == 1)
        mxnet_op::Kernel<SequenceLastBatchwiseKernel<req_type>, xpu>::Launch(
            s, out_size, out.dptr_, data.dptr_, indices.dptr_, max_seq_len,
            out.shape_);
      else
        mxnet_op::Kernel<SequenceLastTimewiseKernel<req_type>, xpu>::Launch(
            s, out_size, out.dptr_, data.dptr_, indices.dptr_, max_seq_len,
            out.shape_);
    });
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;

    CHECK_EQ(in_data.size(), param_.use_sequence_length ? 2U : 1U);
    CHECK_EQ(out_data.size(), 1U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    // only support axis of 0 or 1 for now
    bool axis = static_cast<bool>(param_.axis);

    // Get any size input + output into required form
    index_t d0 = in_data[seq_last::kData].size(0);
    index_t d1 = in_data[seq_last::kData].size(1);
    int dsize = in_data[seq_last::kData].Size();

    int batch = axis ? d0 : d1;
    int max_seq_len = axis ? d1 : d0;
    int rest_size = dsize / (d0 * d1);

    Tensor<xpu, 3, DType> data =
        in_data[seq_last::kData].get_with_shape<xpu, 3, DType>(
            Shape3(d0, d1, rest_size), s);
    Tensor<xpu, 2, DType> out =
        out_data[seq_last::kOut].get_with_shape<xpu, 2, DType>(
            Shape2(batch, rest_size), s);

    if (param_.use_sequence_length) {
      sequence_last(data, out,
                    in_data[seq_last::kSequenceLength].get<xpu, 1, DType>(s),
                    req[seq_last::kOut], s);
    } else {
      Assign(out, req[seq_last::kOut],
             F<mshadow_op::identity>(data[max_seq_len - 1]));
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

    // break immediately if null grad
    if (req[seq_last::kData] == kNullOp) return;

    Stream<xpu> *s = ctx.get_stream<xpu>();

    // Get any size input + output into required form
    index_t n = in_grad[seq_last::kData].size(1);
    int max_seq_len = in_grad[seq_last::kData].size(0);
    int total_size = in_grad[seq_last::kData].Size();
    Shape<2> s2 = Shape2(n, static_cast<int>(total_size / n / max_seq_len));
    Shape<3> s3 =
        Shape3(max_seq_len, n, static_cast<int>(total_size / n / max_seq_len));

    Tensor<xpu, 3, DType> data_grad =
        in_grad[seq_last::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> output_grad =
        out_grad[seq_last::kOut].get_with_shape<xpu, 2, DType>(s2, s);

    // copy indices to vector
    std::vector<index_t> indices_vec(n, max_seq_len);
    if (param_.use_sequence_length)
      IndexTensorToVector(
          in_data[seq_last::kSequenceLength].get<xpu, 1, DType>(s),
          &indices_vec);

    index_t seq_ind;
    if (req[seq_last::kData] == kWriteTo) data_grad = 0.0f;
    for (index_t i = 0; i < n; ++i) {
      seq_ind = indices_vec[i] - 1;
      data_grad[seq_ind][i] += output_grad[i];
    }
  }

 private:
  SequenceLastParam param_;
};  // class SequenceLastOp

template <typename xpu>
Operator *CreateOp(SequenceLastParam param, int dtype);

#if DMLC_USE_CXX11
class SequenceLastProp : public OperatorProperty {
 public:
  int NumOutputs() const override { return 1; }

  std::vector<std::string> ListArguments() const override {
    if (param_.use_sequence_length)
      return {"data", "sequence_length"};
    else
      return {"data"};
  }

  std::vector<std::string> ListOutputs() const override { return {"output"}; }

  void Init(
      const std::vector<std::pair<std::string, std::string>> &kwargs) override {
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

    if ((param_.axis != 0) && (param_.axis != 1)) {
      LOG(FATAL) << "Current implementation expects axis to be 0 or 1.";
    }

    const TShape &dshape = (*in_shape)[seq_last::kData];
    CHECK_GT(dshape.ndim(), 1U)
        << "The data array must be of rank 2 or greater.";
    // seq length vector is same as batch size
    int batchdim = param_.axis ? 0 : 1;
    if (param_.use_sequence_length)
      SHAPE_ASSIGN_CHECK(*in_shape, seq_last::kSequenceLength,
                         Shape1(dshape[batchdim]));

    // calculate output size
    TShape shape_o(dshape.ndim() - 1);
    shape_o[0] = dshape[batchdim];
    for (index_t i = 1; i < shape_o.ndim(); ++i) shape_o[i] = dshape[i + 1];

    const TShape &oshape = shape_o;
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
    auto ptr = new SequenceLastProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "SequenceLast"; }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    if (param_.use_sequence_length)
      return {out_grad[seq_last::kOut], in_data[seq_last::kSequenceLength]};
    else
      return {out_grad[seq_last::kOut]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  SequenceLastParam param_;
};      // class SequenceLastProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_LAST_INL_H_

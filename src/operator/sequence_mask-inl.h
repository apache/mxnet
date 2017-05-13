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
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace seq_mask {
enum SequenceMaskOpInputs { kData, kSequenceLength };
enum SequenceMaskOpOutputs { kOut };
}

struct SequenceMaskParam : public dmlc::Parameter<SequenceMaskParam> {
  bool use_sequence_length;
  float value;
  DMLC_DECLARE_PARAMETER(SequenceMaskParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter `sequence_length` "
            "to specify variable length sequence");
    DMLC_DECLARE_FIELD(value).set_default(0.).describe(
        "The value to be used as a mask.");
  }
};

template <typename xpu, typename DType>
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
    int max_seq_len = in_data[seq_mask::kData].size(0);
    int n = in_data[seq_mask::kData].size(1);
    int total_size = in_data[seq_mask::kData].Size();
    int rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);
    Tensor<xpu, 3, DType> data =
        in_data[seq_mask::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out =
        out_data[seq_mask::kOut].get_with_shape<xpu, 3, DType>(s3, s);
    Assign(out, req[seq_mask::kOut], F<mshadow_op::identity>(data));
    if (param_.use_sequence_length) {
      Tensor<xpu, 1, DType> indices =
          in_data[seq_mask::kSequenceLength].get<xpu, 1, DType>(s);
      SequenceMask(out, indices, static_cast<DType>(param_.value));
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
    int max_seq_len = in_grad[seq_mask::kData].size(0);
    int n = in_grad[seq_mask::kData].size(1);
    int total_size = in_grad[seq_mask::kData].Size();
    int rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);

    Tensor<xpu, 3, DType> data_grad =
        in_grad[seq_mask::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> output_grad =
        out_grad[seq_mask::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    Assign(data_grad, req[seq_mask::kData],
           F<mshadow_op::identity>(output_grad));

    if (param_.use_sequence_length) {
      Tensor<xpu, 1, DType> indices =
          in_data[seq_mask::kSequenceLength].get<xpu, 1, DType>(s);
      SequenceMask(data_grad, indices, DType(0));
    }
  }

 private:
  SequenceMaskParam param_;
};  // class SequenceMaskOp

template <typename xpu>
Operator *CreateOp(SequenceMaskParam param, int dtype);

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

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), param_.use_sequence_length ? 2U : 1U)
        << "Input:[data, sequence_length]";

    const TShape &dshape = (*in_shape)[seq_mask::kData];
    CHECK_GT(dshape.ndim(), 2U)
        << "The data array must be of rank 3 or greater.";
    // seq length vector is same as batch size
    if (param_.use_sequence_length)
      SHAPE_ASSIGN_CHECK(*in_shape, seq_mask::kSequenceLength,
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
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at "
                                       << ListArguments()[i];
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
  SequenceMaskParam param_;
};      // class SequenceMaskProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_SEQUENCE_MASK_INL_H_

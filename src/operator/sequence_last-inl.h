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
  DMLC_DECLARE_PARAMETER(SequenceLastParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter `sequence_length` "
            "to specify variable length sequence");
  }
};

template <typename xpu, typename DType>
class SequenceLastOp : public Operator {
 public:
  explicit SequenceLastOp(SequenceLastParam p) { this->param_ = p; }

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
    index_t n = in_data[seq_last::kData].size(1);
    int max_seq_len = in_data[seq_last::kData].size(0);
    int total_size = in_data[seq_last::kData].Size();
    Shape<2> s2 = Shape2(n, static_cast<int>(total_size / n / max_seq_len));
    Shape<3> s3 =
        Shape3(max_seq_len, n, static_cast<int>(total_size / n / max_seq_len));
    Tensor<xpu, 3, DType> data =
        in_data[seq_last::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 2, DType> out =
        out_data[seq_last::kOut].get_with_shape<xpu, 2, DType>(s2, s);

    if (param_.use_sequence_length) {
      std::vector<index_t> indices_vec(n, max_seq_len);
      IndexTensorToVector(
          in_data[seq_last::kSequenceLength].get<xpu, 1, DType>(s),
          &indices_vec);
      if (req[seq_last::kOut] == kWriteTo) out = 0.0f;
      index_t seq_ind;
      for (index_t i = 0; i < n; ++i) {
        seq_ind = indices_vec[i] - 1;  // 1-indexing
        out[i] += data[seq_ind][i];
      }
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

    const TShape &dshape = (*in_shape)[seq_last::kData];
    CHECK_GT(dshape.ndim(), 2U)
        << "The data array must be of rank 3 or greater.";
    // seq length vector is same as batch size
    if (param_.use_sequence_length)
      SHAPE_ASSIGN_CHECK(*in_shape, seq_last::kSequenceLength,
                         Shape1(dshape[1]));

    // calculate output size
    TShape shape_o(dshape.ndim() - 1);
    for (index_t i = 0; i < shape_o.ndim(); ++i) shape_o[i] = dshape[i + 1];

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

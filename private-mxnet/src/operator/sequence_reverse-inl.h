/*!
 * Copyright (c) 2016 by Contributors
 * \file sequence_reverse-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_
#define MXNET_OPERATOR_SEQUENCE_REVERSE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./sequence_op_common.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

namespace seq_reverse {
enum SequenceReverseOpInputs { kData, kSequenceLength };
enum SequenceReverseOpOutputs { kOut };
}

struct SequenceReverseParam : public dmlc::Parameter<SequenceReverseParam> {
  bool use_sequence_length;
  DMLC_DECLARE_PARAMETER(SequenceReverseParam) {
    DMLC_DECLARE_FIELD(use_sequence_length)
        .set_default(false)
        .describe(
            "If set to true, this layer takes in an extra input parameter `sequence_length` "
            "to specify variable length sequence");
  }
};

template <typename xpu, typename DType>
class SequenceReverseOp : public Operator {
 public:
  explicit SequenceReverseOp(SequenceReverseParam p) { this->param_ = p; }
  void sequence_reverse(const mshadow::Tensor<xpu, 3, DType> data,
                        const mshadow::Tensor<xpu, 3, DType> &out,
                        std::vector<index_t> indices, OpReqType req) {
    using namespace mshadow;
    using namespace mshadow::expr;
    index_t seq_length;
    index_t max_seq_len = data.size(0);
    index_t batch_size = data.size(1);
    for (index_t b = 0; b < batch_size; ++b) {
      seq_length = indices[b];
      for (index_t s = 0; s < max_seq_len; ++s) {
        if (s < seq_length)
          Assign(
              out[s][b], req,
              F<mshadow_op::identity>(
                  data[seq_length - s - 1][b]))
        else  // preserve padding type
          Assign(out[s][b], req, F<mshadow_op::identity>(data[s][b]))
      }
    }
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

    // Get any size input + output into required form
    int max_seq_len = in_data[seq_reverse::kData].size(0);
    int n = in_data[seq_reverse::kData].size(1);
    int total_size = in_data[seq_reverse::kData].Size();
    int rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);
    Tensor<xpu, 3, DType> data =
        in_data[seq_reverse::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> out =
        out_data[seq_reverse::kOut].get_with_shape<xpu, 3, DType>(s3, s);

    // copy indices to vector
    std::vector<index_t> indices_vec(n, max_seq_len);
    if (param_.use_sequence_length)
      IndexTensorToVector(
          in_data[seq_reverse::kSequenceLength].get<xpu, 1, DType>(s),
          &indices_vec);

    sequence_reverse(data, out, indices_vec, req[seq_reverse::kOut]);
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
    int max_seq_len = in_grad[seq_reverse::kData].size(0);
    int n = in_grad[seq_reverse::kData].size(1);
    int total_size = in_grad[seq_reverse::kData].Size();
    int rest_dim = static_cast<int>(total_size / n / max_seq_len);

    Shape<3> s3 = Shape3(max_seq_len, n, rest_dim);

    Tensor<xpu, 3, DType> data_grad =
        in_grad[seq_reverse::kData].get_with_shape<xpu, 3, DType>(s3, s);
    Tensor<xpu, 3, DType> output_grad =
        out_grad[seq_reverse::kOut].get_with_shape<xpu, 3, DType>(s3, s);
    // copy indices to vector
    std::vector<index_t> indices_vec(n, max_seq_len);
    if (param_.use_sequence_length)
      IndexTensorToVector(
          in_data[seq_reverse::kSequenceLength].get<xpu, 1, DType>(s),
          &indices_vec);

    sequence_reverse(output_grad, data_grad, indices_vec,
                     req[seq_reverse::kData]);
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

    const TShape &dshape = (*in_shape)[seq_reverse::kData];
    CHECK_GT(dshape.ndim(), 2U)
        << "The data array must be of rank 3 or greater.";
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

/*!
 * Copyright (c) 2016 by Contributors
 * \file ctc_loss-inl.h
 * \brief
 * \author Sebastian Bodenstien
*/

#ifndef MXNET_OPERATOR_CONTRIB_CTC_LOSS_INL_H_
#define MXNET_OPERATOR_CONTRIB_CTC_LOSS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <ctime>
#include <cstring>
#include <iostream>
#include "../operator_common.h"
#include "../sequence_op_common.h"
#include "../mshadow_op.h"

namespace mxnet {
namespace op {

namespace ctc_loss {
enum CTCLossOpInputs { kData, kLabel };
enum CTCLossOpOutputs { kOut, kGrad };
enum CTCLossOpForwardResource { kTempSpace };
}

template <typename T>
inline void get_workspace_size(std::vector<int> *label_lengths,
                               std::vector<int> *input_lengths,
                               int alphabet_size, int minibatch, bool gpu,
                               size_t *size_bytes) {
  // This is the max of all S and T for all examples in the minibatch.
  int maxL = *std::max_element(label_lengths->data(),
                               label_lengths->data() + minibatch);
  int maxT = *std::max_element(input_lengths->data(),
                               input_lengths->data() + minibatch);

  const int S = 2 * maxL + 1;

  *size_bytes = 0;

  if (gpu) {
    // GPU storage
    // nll_forward, nll_backward
    *size_bytes += 2 * sizeof(T) * minibatch;

    // repeats
    *size_bytes += sizeof(int) * minibatch;

    // label offsets
    *size_bytes += sizeof(int) * minibatch;

    // utt_length
    *size_bytes += sizeof(int) * minibatch;

    // label lengths
    *size_bytes += sizeof(int) * minibatch;

    // labels without blanks - overallocate for now
    *size_bytes += sizeof(int) * maxL * minibatch;

    // labels with blanks
    *size_bytes += sizeof(int) * S * minibatch;

    // alphas
    *size_bytes += sizeof(T) * S * maxT * minibatch;

    // denoms
    *size_bytes += sizeof(T) * maxT * minibatch;

    // probs (since we will pass in activations)
    *size_bytes += sizeof(T) * alphabet_size * maxT * minibatch;

  } else {
    // cpu can eventually replace all minibatch with
    // max number of concurrent threads if memory is
    // really tight

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // output
    per_minibatch_bytes += sizeof(T) * alphabet_size;

    // alphas
    per_minibatch_bytes += sizeof(T) * S * maxT;

    // betas
    per_minibatch_bytes += sizeof(T) * S;

    // labels w/blanks, e_inc, s_inc
    per_minibatch_bytes += 3 * sizeof(int) * S;

    *size_bytes = per_minibatch_bytes * minibatch;

    // probs
    *size_bytes += sizeof(T) * alphabet_size * maxT * minibatch;
  }
}

// Takes a tensor of labels, and interprets 0-elements at the end of the vector
// as padding. The tensor is packed into a std::vector without padding
// characters. The sequence lengths are also inferred from the padding chars
template <typename DType, typename xpu>
inline void LabelTensorToPackedVector(mshadow::Tensor<xpu, 2, DType> labels,
                                      std::vector<int> *packed_labels,
                                      std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  std::vector<index_t> cpu_labels(max_num_labels);

  for (int b = 0; b < batch; ++b) {
    IndexTensorToVector(labels[b], &cpu_labels);
    auto res = std::find(cpu_labels.begin(), cpu_labels.end(), 0);
    int len = std::distance(cpu_labels.begin(), res);
    std::copy(cpu_labels.begin(), cpu_labels.begin() + len,
              std::back_inserter(*packed_labels));
    label_lengths->emplace_back(len);
  }
}

struct CTCLossParam : public dmlc::Parameter<CTCLossParam> {
  DMLC_DECLARE_PARAMETER(CTCLossParam) {}
};

template <typename xpu>
class CTCLossOp : public Operator {
 public:
  explicit CTCLossOp(CTCLossParam p) { this->param_ = p; }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 2U);
    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> data =
        in_data[ctc_loss::kData].get<xpu, 3, real_t>(s);
    Tensor<xpu, 2, real_t> labels =
        in_data[ctc_loss::kLabel].get<xpu, 2, real_t>(s);

    Tensor<xpu, 1, real_t> costs =
        out_data[ctc_loss::kOut].get<xpu, 1, real_t>(s);
    Tensor<xpu, 3, real_t> grad =
        out_data[ctc_loss::kGrad].get<xpu, 3, real_t>(s);

    int max_seq_len = data.size(0);
    int batch_size = data.size(1);
    int alphabet_size = data.size(2);

    // label_lengths
    std::vector<int> packed_labels;
    std::vector<int> label_lengths;
    LabelTensorToPackedVector(labels, &packed_labels, &label_lengths);

    // allocate temporary workspace
    std::vector<int> input_lengths(batch_size, max_seq_len);
    size_t size_bytes;
    bool gpu = data.kDevCPU ? false : true;
    get_workspace_size<real_t>(&label_lengths, &input_lengths, alphabet_size,
                               batch_size, gpu, &size_bytes);

    // round-up so there are enough elems in memory
    int num_tmp_elems = (size_bytes + sizeof(real_t) - 1) / sizeof(real_t);
    Tensor<xpu, 1, real_t> workspace =
        ctx.requested[ctc_loss::kTempSpace].get_space_typed<xpu, 1, real_t>(
            Shape1(num_tmp_elems), s);

    compute_ctc_cost(data, costs.dptr_, grad.dptr_, packed_labels.data(),
                     label_lengths.data(), input_lengths.data(),
                     workspace.dptr_, ctx.is_train);
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

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> data_grad =
        in_grad[ctc_loss::kData].get<xpu, 3, real_t>(s);
    Tensor<xpu, 1, real_t> output_grad =
        out_grad[ctc_loss::kOut].get<xpu, 1, real_t>(s);

    Tensor<xpu, 3, real_t> data_grad_computed =
        out_data[ctc_loss::kGrad].get<xpu, 3, real_t>(s);

    Assign(data_grad, req[ctc_loss::kData],
           broadcast<1>(output_grad, data_grad.shape_) * data_grad_computed);
  }

 private:
  CTCLossParam param_;
};  // class CTCLossOp

template <typename xpu>
Operator *CreateOp(CTCLossParam param, int dtype);

#if DMLC_USE_CXX11
class CTCLossProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 2; }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "grad"};
  }

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
    CHECK_EQ(in_shape->size(), 2U) << "Expect two inputs to the symbol.";

    const TShape &dshape = (*in_shape)[ctc_loss::kData];
    const TShape &lshape = (*in_shape)[ctc_loss::kLabel];
    CHECK_EQ(dshape.ndim(), 3U) << "The data array must be of rank 3.";
    CHECK_EQ(lshape.ndim(), 2U) << "The labels array must be of rank 2.";
    CHECK_EQ(dshape[1], lshape[0])
        << "The batch size for the labels and data arrays must be the same.";

    CHECK_GE(dshape[0], lshape[1]) << "The max number of labels cannot exceed "
                                      "the maximum sequence length of the "
                                      "input.";

    TShape oshape(1);
    oshape[0] = dshape[1];  // batch size
    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(dshape);  // grad output
    return true;
  }

  OperatorProperty *Copy() const override {
    auto ptr = new CTCLossProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override { return "_contrib_CTCLoss"; }

  std::vector<ResourceRequest> ForwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad, const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {out_grad[ctc_loss::kOut], out_data[ctc_loss::kGrad]};
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  CTCLossParam param_;
};      // class CTCLossProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_CTC_LOSS_INL_H_

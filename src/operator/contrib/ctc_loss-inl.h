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
#include "../nn/sequence_mask-inl.h"

#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
#define CUDNN_LABEL_LENGTH_LIMIT 256
#include "../nn/softmax-inl.h"
#endif  // CUDNN

namespace mxnet {
namespace op {

namespace ctc_loss {
enum CTCLossOpInputs { kData, kLabel };
enum CTCLossOpOutputs { kOut, kGrad };
enum CTCLossOpForwardResource { kTempSpace };
}

template <typename T>
inline void get_workspace_size(std::vector<int> *label_lengths,
                               std::vector<int> *data_lengths,
                               int alphabet_size, int minibatch, bool gpu,
                               size_t *size_bytes) {
  // This is the max of all S and T for all examples in the minibatch.
  int maxL = *std::max_element(label_lengths->data(),
                               label_lengths->data() + minibatch);
  int maxT = *std::max_element(data_lengths->data(),
                               data_lengths->data() + minibatch);

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
// as padding. The tensor is packed into an std::vector without padding
// characters. The label sequence lengths are also inferred from the padding chars.
// When cudnn is enabled, the return value signifies whether the cudnn length limit is exceeded.
template <typename DType, typename xpu>
inline bool LabelTensorToPackedVector(mshadow::Tensor<xpu, 2, DType> labels,
                                      int padding_mask,
                                      std::vector<int> *packed_labels,
                                      std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  bool exceed_limit = false;

  std::vector<int> cpu_labels(max_num_labels*batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data()+b*max_num_labels;
    auto res = std::find(start, start+max_num_labels, padding_mask);
    int len = std::distance(start, res);
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    exceed_limit = exceed_limit || len > CUDNN_LABEL_LENGTH_LIMIT;
#endif
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
    label_lengths->at(b) = len;
  }
  return exceed_limit;
}

// Takes a tensor of labels, and a vector which specifies the actual length of each label
// The tensor is packed into an std::vector without padding characters.
// The label length vector is copied into an std::vector.
// When cudnn is enabled, the return value signifies whether the cudnn length limit is exceeded.
template <typename DType, typename xpu>
inline bool PackLabelByLength(mshadow::Tensor<xpu, 2, DType> labels,
                              mshadow::Tensor<xpu, 1, DType> in_label_lengths,
                              std::vector<int> *packed_labels,
                              std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);
  bool exceed_limit = false;

  IndexTensorToVector(in_label_lengths, label_lengths);

  std::vector<int> cpu_labels(max_num_labels*batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data()+b*max_num_labels;
    int len = label_lengths->at(b);
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    exceed_limit = exceed_limit || len > CUDNN_LABEL_LENGTH_LIMIT;
#endif
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
  }
  return exceed_limit;
}

struct CTCLossParam : public dmlc::Parameter<CTCLossParam> {
  bool use_data_lengths;
  bool use_label_lengths;
  int blank_label;
  DMLC_DECLARE_PARAMETER(CTCLossParam) {
    DMLC_DECLARE_FIELD(use_data_lengths).set_default(false)
      .describe("Whether the data lenghts are decided by `data_lengths`. "
                "If false, the lengths are equal to the max sequence length.");
    DMLC_DECLARE_FIELD(use_label_lengths).set_default(false)
      .describe("Whether the label lenghts are decided by "
                "`label_lengths`, or derived from `padding_mask`. "
                "If false, the lengths are derived from the "
                "first occurrence of the value of `padding_mask`. "
                "The value of `padding_mask` is ``0`` when first CTC label is reserved for blank, "
                "and ``-1`` when last label is reserved for blank. See `blank_label`.");
    DMLC_DECLARE_FIELD(blank_label)
      .add_enum("first", 0)
      .add_enum("last", 1)
      .set_default(0)
      .describe("Set the label that is reserved for blank label."
                "If \"first\", 0-th label is reserved, and "
                "label values for tokens in the vocabulary are "
                "between ``1`` and ``alphabet_size-1``, and the padding mask is ``-1``. "
                "If \"last\", last label value ``alphabet_size-1`` "
                "is reserved for blank label instead, "
                "and label values for tokens in the vocabulary are "
                "between ``0`` and ``alphabet_size-2``, and the padding mask is ``0``.");
  }
};

template <typename xpu>
class CTCLossOp : public Operator {
 public:
  explicit CTCLossOp(CTCLossParam p) {
    this->param_ = p;
    exceed_cudnn_limit = false;
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    CUDNN_CALL(cudnnCreateCTCLossDescriptor(&ctc_desc_));
    CUDNN_CALL(cudnnSetCTCLossDescriptor(ctc_desc_, CUDNN_DATA_FLOAT));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&prob_desc_));
    CUDNN_CALL(cudnnCreateTensorDescriptor(&grad_desc_));
#endif
  }

  ~CTCLossOp() {
#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
    CUDNN_CALL(cudnnDestroyCTCLossDescriptor(ctc_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(prob_desc_));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(grad_desc_));
#endif
  }

  virtual void Forward(const OpContext &ctx, const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 2U+param_.use_data_lengths+param_.use_label_lengths);
    CHECK_EQ(out_data.size(), 2U);
    exceed_cudnn_limit = false;
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

    // data_lengths
    std::vector<int> data_lengths(batch_size, max_seq_len);
    if (param_.use_data_lengths) {
      int kInputLength = 2;
      IndexTensorToVector(in_data[kInputLength].get<xpu, 1, real_t>(s), &data_lengths);
    }

    // label_lengths
    std::vector<int> packed_labels;
    std::vector<int> label_lengths(batch_size);

    if (param_.use_label_lengths) {
      int kLabelLength = 2+param_.use_data_lengths;
      exceed_cudnn_limit = PackLabelByLength(labels, in_data[kLabelLength].get<xpu, 1, real_t>(s),
                                             &packed_labels, &label_lengths);
    } else {
      exceed_cudnn_limit = LabelTensorToPackedVector(labels, param_.blank_label == 0?0:-1,
                                                     &packed_labels, &label_lengths);
    }

// CUDNN is disabled due to lack of support for input lengths
/* #if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7 */
/*     if (!exceed_cudnn_limit) { */
/*       cudnn_forward(ctx, s, data, costs, grad, */
/*                     &data_lengths, &label_lengths, &packed_labels, */
/*                     max_seq_len, batch_size, alphabet_size, */
/*                     req[ctc_loss::kGrad] != mxnet::kNullOp); */
/*     } else { */
/*       baidu_forward(ctx, s, data, costs, grad, */
/*                     &data_lengths, &label_lengths, &packed_labels, */
/*                     batch_size, alphabet_size, req[ctc_loss::kGrad] != mxnet::kNullOp); */
/*     } */
/* #else */

    baidu_forward(ctx, s, data, costs, grad,
                  &data_lengths, &label_lengths, &packed_labels,
                  batch_size, alphabet_size, req[ctc_loss::kGrad] != mxnet::kNullOp);

    if (param_.use_data_lengths) {
      // baidu warp CTC implementation sometimes includes undefined gradients
      // for data outside of length mask. Setting to 0 to make it consistent
      // with CPU implementation.
      int kInputLength = 2;
      mxnet_op::SequenceMask(grad, in_data[kInputLength].get<xpu, 1, real_t>(s),
                             static_cast<real_t>(0));
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

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 3, real_t> data_grad =
        in_grad[ctc_loss::kData].get<xpu, 3, real_t>(s);
    Tensor<xpu, 1, real_t> output_grad =
        out_grad[ctc_loss::kOut].get<xpu, 1, real_t>(s);

    Tensor<xpu, 3, real_t> data_grad_computed =
        out_data[ctc_loss::kGrad].get<xpu, 3, real_t>(s);

    Assign(data_grad, req[ctc_loss::kData],
           mshadow::expr::broadcast<1>(output_grad, data_grad.shape_) * data_grad_computed);
  }

 private:
  CTCLossParam param_;
  bool exceed_cudnn_limit;

#if defined(__CUDACC__) && MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 7
  cudnnDataType_t dtype_;
  cudnnCTCLossDescriptor_t ctc_desc_;
  cudnnTensorDescriptor_t prob_desc_, grad_desc_;

  inline virtual void cudnn_forward(const OpContext &ctx,
                                    mshadow::Stream<xpu>* s,
                                    mshadow::Tensor<xpu, 3, real_t> data,
                                    mshadow::Tensor<xpu, 1, real_t> costs,
                                    mshadow::Tensor<xpu, 3, real_t> grad,
                                    std::vector<int>* data_lengths,
                                    std::vector<int>* label_lengths,
                                    std::vector<int>* packed_labels,
                                    int max_seq_len,
                                    int batch_size,
                                    int alphabet_size,
                                    bool req_grad) {
    using namespace mshadow;

    // call cudnn to calculate ctc loss
    dtype_ = CUDNN_DATA_FLOAT;
    int dims[3], strides[3];
    size_t workspace_bytes;
    int workspace_size;
    dims[0] = max_seq_len;
    dims[1] = batch_size;
    dims[2] = alphabet_size;
    strides[0] = batch_size*alphabet_size;
    strides[1] = alphabet_size;
    strides[2] = 1;
    cudnnCTCLossAlgo_t ctc_algo = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
    CUDNN_CALL(cudnnSetTensorNdDescriptor(prob_desc_,
                                          dtype_,
                                          3,
                                          dims,
                                          strides));
    CUDNN_CALL(cudnnSetTensorNdDescriptor(grad_desc_,
                                          dtype_,
                                          3,
                                          dims,
                                          strides));
    CUDNN_CALL(cudnnGetCTCLossWorkspaceSize(s->dnn_handle_,
                                            prob_desc_,
                                            req_grad?grad_desc_:NULL,
                                            packed_labels->data(),
                                            label_lengths->data(),
                                            data_lengths->data(),
                                            ctc_algo,
                                            ctc_desc_,
                                            &workspace_bytes));
    workspace_size = (workspace_bytes + sizeof(real_t) - 1)/sizeof(real_t);

    Tensor<xpu, 1, real_t> temp_space =
      ctx.requested[ctc_loss::kTempSpace].get_space_typed<xpu, 1, real_t>(
          mshadow::Shape1(workspace_size+data.shape_.FlatTo1D()[0]), s);

    Tensor<gpu, 1, real_t> work_space(temp_space.dptr_,
                                      mshadow::Shape1(workspace_size), s);
    Tensor<xpu, 3, real_t> prob(temp_space.dptr_+workspace_size,
                                data.shape_, s);

    // since the input is activation before softmax and cudnn ctc takes softmax
    // apply softmax to inputs first.
    mxnet_op::Softmax<mxnet_op::softmax_fwd>(s, data.dptr_, prob.dptr_, data.shape_, 2);

    CUDNN_CALL(cudnnCTCLoss(s->dnn_handle_,
                            prob_desc_,
                            prob.dptr_,
                            packed_labels->data(),
                            label_lengths->data(),
                            data_lengths->data(),
                            costs.dptr_,
                            req_grad?grad_desc_:NULL,
                            req_grad?grad.dptr_:NULL,
                            ctc_algo,
                            ctc_desc_,
                            work_space.dptr_,
                            workspace_bytes));

    if (req_grad) {
      mxnet_op::SoftmaxGrad<mshadow_op::mul, mxnet_op::softmax_bwd>(s,
          prob.dptr_, grad.dptr_, grad.dptr_, data.shape_, 2);
      Assign(grad, mxnet::kWriteInplace, grad * alphabet_size);
    }
  }
#endif  // __CUDACC__ && CUDNN

  inline virtual void baidu_forward(const OpContext &ctx,
                                    mshadow::Stream<xpu>* s,
                                    mshadow::Tensor<xpu, 3, real_t> data,
                                    mshadow::Tensor<xpu, 1, real_t> costs,
                                    mshadow::Tensor<xpu, 3, real_t> grad,
                                    std::vector<int>* data_lengths,
                                    std::vector<int>* label_lengths,
                                    std::vector<int>* packed_labels,
                                    int batch_size,
                                    int alphabet_size,
                                    bool req_grad) {
    using namespace mshadow;
    // allocate temporary workspace
    size_t size_bytes;
    bool gpu = data.kDevCPU ? false : true;
    get_workspace_size<real_t>(label_lengths, data_lengths, alphabet_size,
                               batch_size, gpu, &size_bytes);

    // round-up so there are enough elems in memory
    int num_tmp_elems = (size_bytes + sizeof(real_t) - 1) / sizeof(real_t);
    Tensor<xpu, 1, real_t> workspace =
        ctx.requested[ctc_loss::kTempSpace].get_space_typed<xpu, 1, real_t>(
            Shape1(num_tmp_elems), s);

    compute_ctc_cost(data, costs.dptr_, grad.dptr_, packed_labels->data(),
                     label_lengths->data(), data_lengths->data(),
                     workspace.dptr_, req_grad,
                     param_.blank_label == 0?0:(alphabet_size-1));
  }
};  // class CTCLossOp

template <typename xpu>
Operator *CreateOp(CTCLossParam param, int dtype);

#if DMLC_USE_CXX11
class CTCLossProp : public OperatorProperty {
 public:
  int NumVisibleOutputs() const override { return 1; }

  int NumOutputs() const override { return 2; }

  std::vector<std::string> ListArguments() const override {
    if (param_.use_data_lengths && param_.use_label_lengths) {
      return {"data", "label", "data_lengths", "label_lengths"};
    } else if (param_.use_data_lengths) {
      return {"data", "label", "data_lengths"};
    } else if (param_.use_label_lengths) {
      return {"data", "label", "label_lengths"};
    } else {
      return {"data", "label"};
    }
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "grad"};
  }

  void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape, std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    index_t expected_inputs = 2+param_.use_data_lengths+param_.use_label_lengths;
    CHECK_EQ(in_shape->size(), expected_inputs)
        << "Expect " << expected_inputs << " inputs to the symbol.";

    const TShape &dshape = (*in_shape)[ctc_loss::kData];
    const TShape &lshape = (*in_shape)[ctc_loss::kLabel];
    CHECK_EQ(dshape.ndim(), 3U) << "The data array must be of rank 3.";
    CHECK_EQ(lshape.ndim(), 2U) << "The labels array must be of rank 2.";
    CHECK_EQ(dshape[1], lshape[0])
        << "The batch size for the labels and data arrays must be the same.";
    if (param_.use_data_lengths) {
      int kInputLength = 2;
      const TShape &dlshape = (*in_shape)[kInputLength];
      CHECK_EQ(dlshape.ndim(), 1U) << "Data length array must be a vector.";
      CHECK_EQ(dlshape[0], dshape[1])
          << "The batch size for the data and data lengths must be the same.";
    }
    if (param_.use_label_lengths) {
      int kLabelLength = 2+param_.use_data_lengths;
      const TShape &llshape = (*in_shape)[kLabelLength];
      CHECK_EQ(llshape.ndim(), 1U) << "Label length array must be a vector.";
      CHECK_EQ(llshape[0], lshape[0])
          << "The batch size for the labels and label lengths must be the same.";
    }

    CHECK_GE(dshape[0], lshape[1]) << "The max number of labels cannot exceed "
                                      "the maximum sequence length of the "
                                      "data.";

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

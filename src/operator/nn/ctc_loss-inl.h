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
 * Copyright (c) 2017 by Contributors
 * \file ctc_loss-inl.h
 * \brief CTC Loss operator
*/

#ifndef MXNET_OPERATOR_NN_CTC_LOSS_INL_H_
#define MXNET_OPERATOR_NN_CTC_LOSS_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <string>
#include "../mshadow_op.h"
#include "./sequence_mask-inl.h"
#include "../sequence_op_common.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

namespace ctc_loss {
enum CTCLossOpInputs { kData, kLabel };
enum CTCLossOpOutputs { kOut, kGrad };
}

template <typename T>
inline void get_workspace_size(const std::vector<int> *label_lengths,
                               const std::vector<int> *data_lengths,
                               int alphabet_size, int minibatch, bool isGPU,
                               size_t *size_bytes) {
  // This is the max of all S and T for all examples in the minibatch.
  int maxL = *std::max_element(label_lengths->data(),
                               label_lengths->data() + minibatch);
  int maxT = *std::max_element(data_lengths->data(),
                               data_lengths->data() + minibatch);

  const int S = 2 * maxL + 1;

  *size_bytes = 0;

  if (isGPU) {
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
template <typename DType, typename xpu>
inline void LabelTensorToPackedVector(mshadow::Tensor<xpu, 2, DType> labels,
                                      int padding_mask,
                                      std::vector<int> *packed_labels,
                                      std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);

  std::vector<int> cpu_labels(max_num_labels * batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data() + b * max_num_labels;
    auto res = std::find(start, start + max_num_labels, padding_mask);
    int len = std::distance(start, res);
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
    label_lengths->at(b) = len;
  }
}

// Takes a tensor of labels, and a vector which specifies the actual length of each label
// The tensor is packed into an std::vector without padding characters.
// The label length vector is copied into an std::vector.
template <typename DType, typename xpu>
inline void PackLabelByLength(mshadow::Tensor<xpu, 2, DType> labels,
                              mshadow::Tensor<xpu, 1, DType> in_label_lengths,
                              std::vector<int> *packed_labels,
                              std::vector<int> *label_lengths) {
  int batch = labels.size(0);
  int max_num_labels = labels.size(1);

  IndexTensorToVector(in_label_lengths, label_lengths);

  std::vector<int> cpu_labels(max_num_labels * batch);
  mshadow::Tensor<xpu, 1, DType> flat_labels = labels.FlatTo1D();
  IndexTensorToVector(flat_labels, &cpu_labels);

  for (int b = 0; b < batch; ++b) {
    auto start = cpu_labels.data() + b * max_num_labels;
    int len = label_lengths->at(b);
    std::copy(start, start + len,
              std::back_inserter(*packed_labels));
  }
}

struct CTCLossOpParam : public dmlc::Parameter<CTCLossOpParam> {
  bool use_data_lengths;
  bool use_label_lengths;
  int blank_label;
  DMLC_DECLARE_PARAMETER(CTCLossOpParam) {
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

// By default, the inputs must include data array and label array
// if use_data_lengths parameter is set, user should also supply
// data_lengths array; if use_label_lengths parameter is set, user
// should also specify label_lengths array.
inline uint32_t CTCLossOpNumInputs(const NodeAttrs& attrs) {
  const CTCLossOpParam& param = nnvm::get<CTCLossOpParam>(attrs.parsed);
  return 2U + param.use_data_lengths + param.use_label_lengths;
}

inline bool CTCLossOpShape(const nnvm::NodeAttrs &attrs,
                           mxnet::ShapeVector* in_attrs,
                           mxnet::ShapeVector* out_attrs) {
    const CTCLossOpParam& param = nnvm::get<CTCLossOpParam>(attrs.parsed);
    CHECK_EQ(in_attrs->size(), CTCLossOpNumInputs(attrs));
    CHECK_EQ(out_attrs->size(), 2U);

    const mxnet::TShape &dshape = (*in_attrs)[ctc_loss::kData];
    const mxnet::TShape &lshape = (*in_attrs)[ctc_loss::kLabel];
    CHECK_EQ(dshape.ndim(), 3U) << "The number of dimensions of data array must be 3.";
    CHECK_EQ(lshape.ndim(), 2U) << "The number of dimensions of labels array must be 2.";
    CHECK_EQ(dshape[1], lshape[0])
        << "The batch size for the labels and data arrays must be the same.";

    if (param.use_data_lengths) {
      int kInputLength = 2;
      const mxnet::TShape &dlshape = (*in_attrs)[kInputLength];
      CHECK_EQ(dlshape.ndim(), 1U) << "Data length array must be a vector.";
      CHECK_EQ(dlshape[0], dshape[1])
          << "The batch size for the data and data lengths must be the same.";
    }
    if (param.use_label_lengths) {
      int kLabelLength = 2 + param.use_data_lengths;
      const mxnet::TShape &llshape = (*in_attrs)[kLabelLength];
      CHECK_EQ(llshape.ndim(), 1U) << "Label length array must be a vector.";
      CHECK_EQ(llshape[0], lshape[0])
          << "The batch size for the labels and label lengths must be the same.";
    }
    CHECK_GE(dshape[0], lshape[1]) << "The max number of labels cannot exceed "
                                      "the maximum sequence length of the "
                                      "data.";

    mxnet::TShape oshape(1, -1);
    oshape[0] = dshape[1];  // batch size
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);  // forward output
    SHAPE_ASSIGN_CHECK(*out_attrs, 1, dshape);  // grad output
    return true;
}

inline bool CTCLossOpType(const nnvm::NodeAttrs& attrs,
                          std::vector<int>* in_attrs,
                          std::vector<int>* out_attrs) {
    CHECK_GE(in_attrs->size(), 2U);
    CHECK_EQ(out_attrs->size(), 2U);
    int dtype = (*in_attrs)[ctc_loss::kData];
    CHECK_NE(dtype, -1) << "Input data must have specified type";

    TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));  // forward output
    TYPE_ASSIGN_CHECK(*out_attrs, 1, in_attrs->at(0));  // grad output
    return true;
}

inline bool CTCLossOpStorageType(const nnvm::NodeAttrs& attrs,
                                 const int dev_mask,
                                 DispatchMode* dispatch_mode,
                                 std::vector<int>* in_attrs,
                                 std::vector<int>* out_attrs) {
  CHECK_GE(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 2U);
  const int in_stype = in_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && in_stype == kDefaultStorage) {
    // dns -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}


inline std::vector<std::string> CTCLossOpListInputNames(const NodeAttrs& attrs) {
  const CTCLossOpParam& param = nnvm::get<CTCLossOpParam>(attrs.parsed);
  if (param.use_data_lengths && param.use_label_lengths) {
    return {"data", "label", "data_lengths", "label_lengths"};
  } else if (param.use_data_lengths) {
    return {"data", "label", "data_lengths"};
  } else if (param.use_label_lengths) {
    return {"data", "label", "label_lengths"};
  } else {
    return {"data", "label"};
  }
}

template<typename xpu>
void CTCLossOpForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<TBlob>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;

  const CTCLossOpParam& param = nnvm::get<CTCLossOpParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), CTCLossOpNumInputs(attrs));
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  const TBlob& in_data = inputs[ctc_loss::kData];
  const TBlob& in_label = inputs[ctc_loss::kLabel];
  const TBlob& out_data = outputs[ctc_loss::kOut];
  const TBlob& out_grad = outputs[ctc_loss::kGrad];

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(inputs[ctc_loss::kLabel].type_flag_, DType, {
    Tensor<xpu, 3, real_t> data = in_data.get<xpu, 3, real_t>(s);
    Tensor<xpu, 2, DType> labels = in_label.get<xpu, 2, DType>(s);
    Tensor<xpu, 1, real_t> costs = out_data.get<xpu, 1, real_t>(s);
    Tensor<xpu, 3, real_t> grad = out_grad.get<xpu, 3, real_t>(s);

    int max_seq_len = data.size(0);
    int batch_size = data.size(1);
    int alphabet_size = data.size(2);

    // data_lengths
    std::vector<int> data_lengths(batch_size, max_seq_len);
    if (param.use_data_lengths) {
      int kInputLength = 2;
      IndexTensorToVector(inputs[kInputLength].get<xpu, 1, real_t>(s), &data_lengths);
    }

    // label_lengths
    std::vector<int> packed_labels;
    std::vector<int> label_lengths(batch_size);

    if (param.use_label_lengths) {
      int kLabelLength = 2 + param.use_data_lengths;
      PackLabelByLength(labels, inputs[kLabelLength].get<xpu, 1, DType>(s),
                        &packed_labels, &label_lengths);
    } else {
      LabelTensorToPackedVector(labels, param.blank_label == 0 ? 0 : -1,
                                &packed_labels, &label_lengths);
    }

    size_t size_bytes;
    get_workspace_size<real_t>(&label_lengths, &data_lengths, alphabet_size,
                               batch_size, data.kDevCPU ? false : true, &size_bytes);

    // round-up so there are enough elems in memory
    int num_tmp_elems = (size_bytes + sizeof(real_t) - 1) / sizeof(real_t);
    Tensor<xpu, 1, real_t> workspace =
      ctx.requested[0].get_space_typed<xpu, 1, real_t>(Shape1(num_tmp_elems), s);

    compute_ctc_cost(data, costs.dptr_, grad.dptr_, packed_labels.data(),
                     label_lengths.data(), data_lengths.data(),
                     workspace.dptr_, req[ctc_loss::kGrad] != mxnet::kNullOp,
                     param.blank_label == 0 ? 0 : (alphabet_size - 1));

    if (param.use_data_lengths) {
      // baidu warp CTC implementation sometimes includes undefined gradients
      // for data outside of length mask. Setting to 0 to make it consistent
      // with CPU implementation.
      int kInputLength = 2;
      mxnet_op::SequenceMask(grad, inputs[kInputLength].get<xpu, 1, real_t>(s),
                             static_cast<real_t>(0));
    }
  });
}

template<typename xpu>
void CTCLossOpBackward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_grad = outputs[0];
  const TBlob& out_grad = inputs[0];
  const TBlob& grad_computed = inputs[3];  // grad computed in the forward step

  Tensor<xpu, 3, real_t> igrad_data = in_grad.get<xpu, 3, real_t>(s);
  Tensor<xpu, 1, real_t> ograd_data = out_grad.get<xpu, 1, real_t>(s);
  Tensor<xpu, 3, real_t> computed_grad_data = grad_computed.get<xpu, 3, real_t>(s);

  Assign(igrad_data, req[0],
         mshadow::expr::broadcast<1>(ograd_data, computed_grad_data.shape_) * computed_grad_data);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CTC_LOSS_INL_H_


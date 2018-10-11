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
 *  Copyright (c) 2018 by Contributors
 * \file optimizer_op-inl.h
 * \brief Optimizer operators
 * \author Leonard Lausen
 */
#ifndef MXNET_OPERATOR_CONTRIB_OPTIMIZER_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_OPTIMIZER_OP_INL_H_
#include <dmlc/parameter.h>
#include <mshadow/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

namespace mxnet {
namespace op {

struct GroupAdagradParam : public dmlc::Parameter<GroupAdagradParam> {
  float lr;
  float epsilon;
  float rescale_grad;
  float clip_gradient;
  DMLC_DECLARE_PARAMETER(GroupAdagradParam) {
    DMLC_DECLARE_FIELD(lr).describe("Learning rate");
    DMLC_DECLARE_FIELD(rescale_grad)
        .set_default(1.0f)
        .describe("Rescale gradient to grad = rescale_grad*grad.");
    DMLC_DECLARE_FIELD(clip_gradient)
        .set_default(-1.0f)
        .describe(
            "Clip gradient to the range of [-clip_gradient, clip_gradient] "
            "If clip_gradient <= 0, gradient clipping is turned off. "
            "grad = max(min(grad, clip_gradient), -clip_gradient).");
    DMLC_DECLARE_FIELD(epsilon).set_default(1.0e-5).describe(
        "Epsilon for numerical stability");
  }
};

inline bool GroupAdagradStorageType(const nnvm::NodeAttrs &attrs,
                                    const int dev_mask,
                                    DispatchMode *dispatch_mode,
                                    std::vector<int> *in_attrs,
                                    std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int weight_stype = in_attrs->at(0);
  const int grad_stype = in_attrs->at(1);
  const int state_stype = in_attrs->at(2);
  bool dispatched = false;
  if (!dispatched && common::ContainsOnlyStorage(*in_attrs, kDefaultStorage)) {
    // dns, ... -> dns
    dispatched = storage_type_assign(out_attrs, kDefaultStorage, dispatch_mode,
                                     DispatchMode::kFCompute);
  }
  if (!dispatched && grad_stype == kRowSparseStorage &&
      (weight_stype == kRowSparseStorage || weight_stype == kDefaultStorage) &&
      state_stype == weight_stype) {
    // weight and state share stype, grad's stype = rsp
    dispatched = storage_type_assign(
        out_attrs, static_cast<NDArrayStorageType>(weight_stype), dispatch_mode,
        DispatchMode::kFComputeEx);
  }
  return dispatched;
}

/*! \brief kernel for sparse adagrad update with group sparsity regularization
 */
template <typename xpu> struct GroupAdagradDnsRspKernel {
  template <typename DType, typename IType>
  MSHADOW_XINLINE static void
  Map(int i, const index_t row_length, DType *out_data, DType *state_data,
      DType *weight_data, const IType *grad_idx, const DType *grad_data,
      const DType clip_gradient, const DType rescale_grad, const DType lr,
      const DType eps) {
    using namespace mshadow_op;

    // Helper to obtain index into weight / state arrays
    auto get_data_j = [&i, &grad_idx, &row_length](index_t j) -> index_t {
      return grad_idx[i] * row_length + j;
    };
    // Helper to obtain explicit rescaled and clipped grad
    auto get_grad_rescaled = [&i, &row_length, &grad_data, &rescale_grad,
                              &clip_gradient](index_t j) -> DType {
      index_t grad_j = i * row_length + j;
      DType grad_rescaled = grad_data[grad_j] * rescale_grad;
      if (clip_gradient >= 0.0f) {
        grad_rescaled = clip::Map(grad_rescaled, clip_gradient);
      }
      return grad_rescaled;
    };

    // Update history states
    DType grad_ssq = 0;
    for (index_t j = 0; j < row_length; j++) {
      const DType grad_rescaled = get_grad_rescaled(j);
      grad_ssq += grad_rescaled * grad_rescaled;
    }
    state_data[grad_idx[i]] += grad_ssq / row_length;

    // Standard Adagrad Update
    for (index_t j = 0; j < row_length; j++) {
      // clang-format off
      const DType grad_rescaled = get_grad_rescaled(j);
      index_t data_j = get_data_j(j);
      const DType div = lr * grad_rescaled / square_root::Map(state_data[grad_idx[i]] + eps);
      out_data[data_j] = weight_data[data_j] - div;
      // clang-format on
    }
  }
};

/*
 * \brief Group Adagrad update implementation for dense weight and row_sparse
 * grad.
 */
template <typename xpu>
inline void GroupAdagradUpdateDnsRspDnsImpl(
    const GroupAdagradParam &param, const OpContext &ctx, const TBlob &weight,
    const NDArray &grad, const TBlob &state, const OpReqType &req, TBlob *out) {
  using namespace mshadow;
  using namespace mshadow::expr;
  using namespace mshadow_op;
  using namespace mxnet_op;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(grad.storage_type(), kRowSparseStorage);
  // if gradients are zeros, no weights are updated
  if (req == kNullOp) {
    return;
  }
  CHECK_EQ(req, kWriteInplace)
      << "kWriteInplace is expected for sparse group_adagrad_update";
  CHECK_GT(weight.shape_.Size(), 0);
  CHECK_GT(state.shape_.Size(), 0);

  MSHADOW_REAL_TYPE_SWITCH(weight.type_flag_, DType, {
    MSHADOW_IDX_TYPE_SWITCH(grad.aux_type(rowsparse::kIdx), IType, {
      DType *weight_data = weight.dptr<DType>();
      DType *out_data = out->dptr<DType>();
      const IType *grad_idx = grad.aux_data(rowsparse::kIdx).dptr<IType>();
      const DType *grad_val = grad.data().dptr<DType>();
      DType *state_data = state.dptr<DType>();
      const nnvm::dim_t num_grad = grad.aux_shape(rowsparse::kIdx)[0];
      const auto row_length = weight.shape_.ProdShape(1, weight.ndim());

      if (!grad.storage_initialized()) {
        // Lazy update with 0 gradient
        return;
      }

      Kernel<GroupAdagradDnsRspKernel<xpu>, xpu>::Launch(
          s, num_grad, row_length, out_data, state_data, weight_data, grad_idx,
          grad_val, static_cast<DType>(param.clip_gradient),
          static_cast<DType>(param.rescale_grad), static_cast<DType>(param.lr),
          static_cast<DType>(param.epsilon));
    });
  });
}

/*
 * \brief AdaGrad update implementation for row_sparse grad. Both standard
 *        update and lazy update are supported.
 */
template <typename xpu>
inline void
GroupAdagradUpdateRspRspRspImpl(const GroupAdagradParam &param,
                                const OpContext &ctx, const NDArray &weight,
                                const NDArray &grad, const NDArray &state,
                                const OpReqType &req, NDArray *out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace rowsparse;
  CheckAllRowsPresent(weight, "GroupAdagradUpdate", "weights");
  Stream<xpu> *s = ctx.get_stream<xpu>();
  // fill history with zero values
  if (!state.storage_initialized()) {
    NDArray state_zeros = state;
    FillDnsZerosRspImpl(s, &state_zeros);
  } else {
    CheckAllRowsPresent(state, "GroupAdagradUpdate", "states");
  }
  // reuse dns rsp implementation when storage_shape == shape
  TBlob out_blob = out->data();
  GroupAdagradUpdateDnsRspDnsImpl<xpu>(param, ctx, weight.data(), grad,
                                       state.data(), req, &out_blob);
}

template <typename xpu>
inline void GroupAdagradUpdateEx(const nnvm::NodeAttrs &attrs,
                                 const OpContext &ctx,
                                 const std::vector<NDArray> &inputs,
                                 const std::vector<OpReqType> &req,
                                 const std::vector<NDArray> &outputs) {
  const GroupAdagradParam &param = nnvm::get<GroupAdagradParam>(attrs.parsed);
  const auto weight_stype = inputs[0].storage_type();
  const auto grad_stype = inputs[1].storage_type();
  const auto state_stype = inputs[2].storage_type();
  const auto output_stype = outputs[0].storage_type();

  if (state_stype == weight_stype && output_stype == weight_stype &&
      weight_stype == kRowSparseStorage && grad_stype == kRowSparseStorage) {
    NDArray out = outputs[0];
    GroupAdagradUpdateRspRspRspImpl<xpu>(param, ctx, inputs[0], inputs[1],
                                         inputs[2], req[0], &out);
  } else if (state_stype == weight_stype && output_stype == weight_stype &&
             weight_stype == kDefaultStorage &&
             grad_stype == kRowSparseStorage) {
    TBlob out_blob = outputs[0].data();
    GroupAdagradUpdateDnsRspDnsImpl<xpu>(param, ctx, inputs[0].data(),
                                         inputs[1], inputs[2].data(), req[0],
                                         &out_blob);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_OPTIMIZER_OP_INL_H_

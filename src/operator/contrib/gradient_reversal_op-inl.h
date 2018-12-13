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
 * Copyright (c) 2018 by Contributors
 * \file gradient_reversal_op-inl.h
 * \brief
 * \author Istvan Fehervari
*/
#ifndef MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_

#include <mxnet/operator_util.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct GradientReversalParam : public dmlc::Parameter<GradientReversalParam> {
  float l;
  DMLC_DECLARE_PARAMETER(GradientReversalParam) {
    DMLC_DECLARE_FIELD(l)
      .set_default(0.0)
      .describe("Lambda coefficient of the gradient reversal function.");
  }
};

inline bool GradientReversalOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool GradientReversalOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

inline bool GradientReversalOpStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  const int in_stype = in_attrs->at(0);
  int& out_stype = out_attrs->at(0);
  bool dispatched = false;
  if (!dispatched && in_stype == kDefaultStorage) {
    // dns -> dns
    dispatched = storage_type_assign(&out_stype, kDefaultStorage,
                                     dispatch_mode, DispatchMode::kFCompute);
  }
  if (!dispatched && in_stype == kCSRStorage) {
    // csr -> csr
    dispatched = storage_type_assign(&out_stype, kCSRStorage,
                                     dispatch_mode, DispatchMode::kFComputeEx);
  }
  if (!dispatched) {
    dispatched = dispatch_fallback(out_attrs, dispatch_mode);
  }
  return dispatched;
}

template<int req>
struct gradient_reversal_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<int req>
struct gradient_reversal_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data, const float l) {
    KERNEL_ASSIGN(in_grad[i], req, out_grad[i] * -l);
  }
};

template<typename xpu>
void GradientReversalOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<gradient_reversal_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>());
    });
  });
}

template<typename xpu>
void GradientReversalOpForwardCsrImpl(const OpContext& ctx,
                               const NDArray& input,
                               const OpReqType req,
                               const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "GradientReversalOp with CSR only supports kWriteTo";
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!input.storage_initialized()) {
    FillZerosCsrImpl(s, output);
    return;
  }
  const nnvm::dim_t nnz = input.storage_shape()[0];
  const nnvm::dim_t num_rows = output.shape()[0];
  output.CheckAndAlloc({Shape1(num_rows + 1), Shape1(nnz)});
  CHECK_EQ(output.aux_type(kIdx), output.aux_type(kIndPtr))
    << "The dtypes of indices and indptr don't match";
  MSHADOW_TYPE_SWITCH(output.dtype(), DType, {
    MSHADOW_IDX_TYPE_SWITCH(output.aux_type(kIdx), IType, {
      MXNET_ASSIGN_REQ_SWITCH(req, req_type, {
        Kernel<gradient_reversal_forward<req_type>, xpu>::Launch(
            s, nnz, output.data().dptr<DType>(), input.data().dptr<DType>());
        Copy(output.aux_data(kIdx).FlatTo1D<xpu, IType>(s),
             input.aux_data(kIdx).FlatTo1D<xpu, IType>(s), s);
        Copy(output.aux_data(kIndPtr).FlatTo1D<xpu, IType>(s),
             input.aux_data(kIndPtr).FlatTo1D<xpu, IType>(s), s);
      });
    });
  });
}

template<typename xpu>
void GradientReversalOpForwardEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (in_stype == kCSRStorage && out_stype == kCSRStorage) {
    GradientReversalOpForwardCsrImpl<xpu>(ctx, inputs[0], req[0], outputs[0]);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void GradientReversalOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  const GradientReversalParam& param = nnvm::get<GradientReversalParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<gradient_reversal_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>(), param.l);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_GRADIENT_REVERSAL_OP_INL_H_
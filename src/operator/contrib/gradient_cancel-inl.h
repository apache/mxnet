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
 * \file quad_function-inl.h
 * \brief Operator implementing quadratic function.
 * For using as an exmaple in the tutorial of adding operators
 * in MXNet backend.
 */
#ifndef MXNET_OPERATOR_CONTRIB_GRADIENT_CANCEL_INL_H_
#define MXNET_OPERATOR_CONTRIB_GRADIENT_CANCEL_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

struct GradCancelParam : public dmlc::Parameter<GradCancelParam> {
  float threshold;
  DMLC_DECLARE_PARAMETER(GradCancelParam) {
    DMLC_DECLARE_FIELD(threshold)
      .set_default(1.0)
      .describe("Threshold for gradient cancelling.");
  }
};

inline bool GradCancelOpShape(const nnvm::NodeAttrs& attrs,
                             std::vector<TShape>* in_attrs,
                             std::vector<TShape>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool GradCancelOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
}

inline bool GradCancelOpStorageType(const nnvm::NodeAttrs& attrs,
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
struct gradcancel_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data) {
    KERNEL_ASSIGN(out_data[i], req, in_data[i]);
  }
};

template<int req>
struct gradcancel_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data, const float threshold) {
    KERNEL_ASSIGN(in_grad[i], req, math::fabs(out_grad[i]) <= threshold ? out_grad[i] : DType(0));
  }
};

template<typename xpu>
void GradCancelOpForward(const nnvm::NodeAttrs& attrs,
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
      Kernel<gradcancel_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>());
    });
  });
}

template<typename xpu>
void GradCancelOpForwardCsrImpl(const GradCancelParam& param,
                               const OpContext& ctx,
                               const NDArray& input,
                               const OpReqType req,
                               const NDArray& output) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;
  if (req == kNullOp) return;
  CHECK_EQ(req, kWriteTo) << "GradCancelOp with CSR only supports kWriteTo";
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
        Kernel<gradcancel_forward<req_type>, xpu>::Launch(
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
void GradCancelOpForwardEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  const GradCancelParam& param = nnvm::get<GradCancelParam>(attrs.parsed);
  const auto in_stype = inputs[0].storage_type();
  const auto out_stype = outputs[0].storage_type();
  if (in_stype == kCSRStorage && out_stype == kCSRStorage) {
    GradCancelOpForwardCsrImpl<xpu>(param, ctx, inputs[0], req[0], outputs[0]);
  } else {
    LogUnimplementedOp(attrs, ctx, inputs, req, outputs);
  }
}

template<typename xpu>
void GradCancelOpBackward(const nnvm::NodeAttrs& attrs,
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
  const GradCancelParam& param = nnvm::get<GradCancelParam>(attrs.parsed);
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<gradcancel_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>(), param.threshold);
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_GRADIENT_CANCEL_INL_H_

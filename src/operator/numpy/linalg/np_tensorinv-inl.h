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
 * Copyright (c) 2019 by Contributors
 * \file np_tensorinv-inl.h
 * \brief Placeholder for tensor inverse operator
 */
#ifndef MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORINV_INL_H_
#define MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORINV_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../operator_common.h"
#include "../../mshadow_op.h"
#include "../../tensor/la_op.h"
#include "../../tensor/la_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct TensorinvParam : public dmlc::Parameter<TensorinvParam> {
  int ind;
  DMLC_DECLARE_PARAMETER(TensorinvParam) {
    DMLC_DECLARE_FIELD(ind)
      .set_default(2)
      .describe("Number of first indices that are involved in the inverse sum.");
  }
};

template<typename xpu>
void TensorinvOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const mxnet::TBlob& a_tblob = inputs[0];
  const mxnet::TBlob& inv_a_tblob = outputs[0];
  const mxnet::TShape& a_shape = a_tblob.shape_;
  CHECK_EQ(inv_a_tblob.type_flag_, a_tblob.type_flag_)
      << "Binary function only support input/output with the same type.";
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    const int ind = nnvm::get<TensorinvParam>(attrs.parsed).ind;
    dim_t prod_front = 1, prod_back = 1;
    if (ind < a_shape.ndim()) {
      for (int i = 0; i < ind; ++i) {
        prod_front *= a_shape[i];
      }
      for (int i = ind; i < a_shape.ndim(); ++i) {
        prod_back *= a_shape[i];
      }
    } else {
      for (int i = 0; i < a_shape.ndim(); ++i) {
        prod_front *= a_shape[i];
      }
    }
    Tensor<xpu, 3, OType> A =
      a_tblob.get_with_shape<xpu, 3, OType>(Shape3(1, prod_back, prod_front), s);
    Tensor<xpu, 3, OType> inv_A =
      inv_a_tblob.get_with_shape<xpu, 3, OType>(Shape3(1, prod_back, prod_front), s);
    inverse::op(A, inv_A, ctx, attrs);
  });
}

template<typename xpu>
void TensorinvOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& inv_a = inputs[1];
  const TBlob& grad_a = outputs[0];
  const TShape& inv_a_shape = inv_a.shape_;
  MSHADOW_SGL_DBL_TYPE_SWITCH(outputs[0].type_flag_, OType, {
    const int axes = nnvm::get<TensorinvParam>(attrs.parsed).ind;
    CHECK_LE(inv_a_shape.ndim(), 6U)
      << "tensorinv backward only support tensor's dimension <= 6";
    if (axes < inv_a_shape.ndim()) {
      const int axes1 = inv_a_shape.ndim() - axes, axes2 = axes;
      TShape inv_a_transpose_shape(inv_a_shape.ndim(), -1);
      for (int i = 0; i < axes; ++i) {
        inv_a_transpose_shape[i] = inv_a_shape[i + inv_a_shape.ndim() - axes];
      }
      for (int i = axes; i < inv_a_shape.ndim(); ++i) {
        inv_a_transpose_shape[i] = inv_a_shape[i - axes];
      }
      TShape temp_shape(2 * axes, -1);
      for (int i = 0; i < axes; ++i) {
        temp_shape[i] = inv_a_transpose_shape[i];
        temp_shape[i + axes] = inv_a_transpose_shape[i];
      }
      Tensor<xpu, 1, char> workspace =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_shape.Size() * sizeof(OType)),
                                                       ctx.get_stream<xpu>());
      TBlob temp_tblob =
        TBlob(reinterpret_cast<OType*>(workspace.dptr_), temp_shape, xpu::kDevMask);
      dim_t a1 = 1, a2 = 1;
      for (int i = 0; i < axes2; ++i) {
        a1 *= inv_a_transpose_shape[i];
      }
      for (int i = 0; i < axes1; ++i) {
        a2 *= inv_a_shape[i];
      }
      Tensor<xpu, 3, OType> inv_a_tensor =
        inv_a.get_with_shape<xpu, 3, OType>(Shape3(1, a2, a1), s);
      Tensor<xpu, 3, OType> out_grad_tensor =
        out_grad.get_with_shape<xpu, 3, OType>(Shape3(1, a2, a1), s);
      Tensor<xpu, 3, OType> temp_tensor =
        temp_tblob.get_with_shape<xpu, 3, OType>(Shape3(1, a1, a1), s);
      Tensor<xpu, 3, OType> grad_a_tensor =
        grad_a.get_with_shape<xpu, 3, OType>(Shape3(1, a1, a2), s);
      gemm2::op(inv_a_tensor, out_grad_tensor, temp_tensor, OType(1), true, false, s);
      gemm2::op(temp_tensor, inv_a_tensor, grad_a_tensor, OType(-1), false, true, s);
    } else {  // axes >= inv_a_shape.ndim()
      dim_t a = 1;
      for (int i = 0; i < inv_a_shape.ndim(); ++i) {
        a *= inv_a_shape[i];
      }
      // check again
      CHECK_EQ(a, 1U)
        << "a shape must be square, i. e., prod(a.shape[:ind]) == prod(a.shape[ind:]).";
      Tensor<xpu, 1, OType> inv_a_tensor =
        inv_a.get_with_shape<xpu, 1, OType>(Shape1(1), s);
      Tensor<xpu, 1, OType> out_grad_tensor =
        out_grad.get_with_shape<xpu, 1, OType>(Shape1(1), s);
      Tensor<xpu, 1, OType> grad_a_tensor =
        grad_a.get_with_shape<xpu, 1, OType>(Shape1(1), s);
      ASSIGN_DISPATCH(grad_a_tensor, kWriteTo,
        OType(-1) * inv_a_tensor * out_grad_tensor * inv_a_tensor);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_LINALG_NP_TENSORINV_INL_H_

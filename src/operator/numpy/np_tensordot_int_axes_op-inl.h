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
 * \file np_tensordot_int_axes_op-inl.h
 * \brief Implementation of numpy-compatible tensordot_int_axes
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_TENSORDOT_INT_AXES_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TENSORDOT_INT_AXES_OP_INL_H_

#include <vector>
#include "np_tensordot_op-inl.h"

namespace mxnet {
namespace op {

using namespace mxnet;
using namespace mshadow;

struct TensordotIntAxesParam : public dmlc::Parameter<TensordotIntAxesParam> {
  int axes;
  DMLC_DECLARE_PARAMETER(TensordotIntAxesParam) {
    DMLC_DECLARE_FIELD(axes);
  }
};

/**
 * gets summed axes of a and b from parameter axes.
 */
inline void GetSummedAxes(
    mxnet::Tuple<int>* a_axes_summed_ptr,
    mxnet::Tuple<int>* b_axes_summed_ptr,
    const int axes,
    const mxnet::TShape& a_shape) {
  std::vector<int> a_axes_summed_vector;
  for (int i = 0; i < axes; i++) {
    a_axes_summed_vector.push_back(a_shape.ndim() - axes + i);
  }
  *a_axes_summed_ptr = mxnet::Tuple<int>(a_axes_summed_vector);

  std::vector<int> b_axes_summed_vector;
  for (int i = 0; i < axes; i++) {
    b_axes_summed_vector.push_back(i);
  }
  *b_axes_summed_ptr = mxnet::Tuple<int>(b_axes_summed_vector);
}

/**
 * Calculates tensordot.
 */
template<typename xpu>
void TensordotIntAxesImpl(
    const int axes,
    const OpContext& ctx,
    const TBlob& a,
    const TBlob& b,
    const TBlob& out,
    const std::vector<OpReqType>& req) {

  if (req[0] == kNullOp) {
    return;
  }

  if (out.shape_.Size() == 0U) {
    return;  // zero-size output, no need to launch kernel
  }

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
      (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "Tensordot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;
  GetSummedAxes(&a_axes_summed, &b_axes_summed, axes, a_shape);

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
    &b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
    b_axes_remained, b_axes_summed, a_shape, b_shape);

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {  // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
      return;
    }  

    MatrixDot<xpu>(ctx, a, b, out, req[0], ad1, ad2, bd1, bd2);
  });
}

/**
 * forward function
 */
template<typename xpu>
void TensordotIntAxesOpForward(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];

  const TensordotIntAxesParam& param = nnvm::get<TensordotIntAxesParam>(attrs.parsed);
  const int axes = param.axes;

  TensordotIntAxesImpl<xpu>(axes, ctx, a, b, out, req);
}

template<typename xpu>
void TensordotIntAxesBackwardImpl(
    const int axes,
    const OpContext& ctx,
    const TBlob& out_grad,
    const TBlob& a,
    const TBlob& b,
    const TBlob& grad_a,
    const TBlob& grad_b,
    const std::vector<OpReqType>& req) {
  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  Tuple<int> a_axes_summed;
  Tuple<int> b_axes_summed;
  GetSummedAxes(&a_axes_summed, &b_axes_summed, axes, a_shape);

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
    &b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
    b_axes_remained, b_axes_summed, a_shape, b_shape);

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MatrixDot<xpu>(ctx, a, out_grad, grad_b, req[1], ad1, ad2, ad1, bd2, true, false);
    MatrixDot<xpu>(ctx, out_grad, b, grad_a, req[0], ad1, bd2, bd1, bd2, false, true);
  });
}

/**
 * backward function.
 */
template<typename xpu>
void TensordotIntAxesOpBackward(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);
  CHECK_EQ(req.size(), 2U);

  const TBlob& out_grad = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];

  const TensordotIntAxesParam& param = nnvm::get<TensordotIntAxesParam>(attrs.parsed);
  const int axes = param.axes;

  TensordotIntAxesBackwardImpl<xpu>(axes, ctx, out_grad, a, b, grad_a, grad_b, req);
}
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_TENSORDOT_INT_AXES_OP_INL_H_

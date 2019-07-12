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
 * \file np_tensordot_op-inl.h
 * \brief CPU Implementation of numpy-compatible tensordot
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_

#include <vector>
#include "np_matrix_op-inl.h"

namespace mxnet {
namespace op {

using namespace mshadow;

struct TensordotParam : public dmlc::Parameter<TensordotParam> {
  mxnet::Tuple<int> a_axes_summed, b_axes_summed;
  DMLC_DECLARE_PARAMETER(TensordotParam) {
    DMLC_DECLARE_FIELD(a_axes_summed);
    DMLC_DECLARE_FIELD(b_axes_summed);
  }
};

/**
 * Gets matrix dimensions of a and b after transpose and reshape.
 */
inline void GetMatrixDimensions(int* ad1,
                                int* ad2,
                                int* bd1,
                                int* bd2,
                                const mxnet::Tuple<int>& a_axes_remained,
                                const mxnet::Tuple<int>& a_axes_summed,
                                const mxnet::Tuple<int>& b_axes_remained,
                                const mxnet::Tuple<int>& b_axes_summed,
                                const mxnet::TShape& a_shape,
                                const mxnet::TShape& b_shape) {
  *ad1 = 1;
  *ad2 = 1;
  *bd1 = 1;
  *bd2 = 1;

  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    *ad1 *= a_shape[a_axes_remained[i]];
  }
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    *ad2 *= a_shape[a_axes_summed[i]];
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    *bd1 *= b_shape[b_axes_summed[i]];
  }
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    *bd2 *= b_shape[b_axes_remained[i]];
  }
}

/**
 * gets new axes of a and b after transpose and reshape.
 */
inline void GetReorderedAxes(const mxnet::Tuple<int>& a_axes_summed,
                             mxnet::Tuple<int>* a_axes_remained,
                             mxnet::Tuple<int>* a_axes,
                             const mxnet::Tuple<int>& b_axes_summed,
                             mxnet::Tuple<int>* b_axes_remained,
                             mxnet::Tuple<int>* b_axes,
                             const mxnet::TShape& a_shape,
                             const mxnet::TShape& b_shape) {
  std::vector<int> a_axes_remained_vector;
  for (int i = 0; i < a_shape.ndim(); i++) {
    a_axes_remained_vector.push_back(i);
  }
  for (auto& i : a_axes_summed) {
    a_axes_remained_vector.erase(std::find(a_axes_remained_vector.begin(),
      a_axes_remained_vector.end(), i));
  }
  *a_axes_remained = mxnet::Tuple<int>(a_axes_remained_vector);

  std::vector<int> a_axes_vector(a_axes_remained_vector);
  for (auto& i : a_axes_summed) {
    a_axes_vector.push_back(i);
  }
  *a_axes = mxnet::Tuple<int>(a_axes_vector);

  std::vector<int> b_axes_remained_vector;
  for (int i = 0; i < b_shape.ndim(); i++) {
    b_axes_remained_vector.push_back(i);
  }
  for (auto& i : b_axes_summed) {
    b_axes_remained_vector.erase(std::find(b_axes_remained_vector.begin(),
                                           b_axes_remained_vector.end(), i));
  }
  *b_axes_remained = mxnet::Tuple<int>(b_axes_remained_vector);

  std::vector<int> b_axes_vector;
  for (auto& i : b_axes_summed) {
    b_axes_vector.push_back(i);
  }
  for (auto& i : b_axes_remained_vector) {
    b_axes_vector.push_back(i);
  }
  *b_axes = mxnet::Tuple<int>(b_axes_vector);
}

/**
 * gets shapes of a and b after transpose and reshape.
 */
inline mxnet::TShape GetReorderedShape(const mxnet::TShape& shape, const mxnet::Tuple<int>& axes) {
  mxnet::TShape new_shape(shape);
  for (int i = 0; i < axes.ndim(); i++) {
    new_shape[i] = shape[axes[i]];
  }
  return new_shape;
}

/**
 * gets matrix dot. Reshapes tensor a as ad1-by-ad2 matrix, tensor b as bd1-by-bd2 matrix, then 
 * calculates matrix dot a * b and stores in tensor out.
 */
template<typename xpu>
void MatrixDot(const OpContext& ctx,
               const TBlob& a,
               const TBlob& b,
               const TBlob& out,
               const OpReqType req,
               const int ad1,
               const int ad2,
               const int bd1,
               const int bd2,
               const bool aT = false,
               const bool bT = false) {
  using namespace mshadow;
  using namespace mshadow_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    Tensor<xpu, 2, DType> a_tensor = a.get_with_shape<xpu, 2, DType>(Shape2(ad1, ad2), s);
    Tensor<xpu, 2, DType> b_tensor = b.get_with_shape<xpu, 2, DType>(Shape2(bd1, bd2), s);

    if (aT && bT) {
      CHECK_EQ(ad1, bd2);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad2, bd1), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor.T(), b_tensor.T()));
    } else if (aT && !bT) {
      CHECK_EQ(ad1, bd1);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad2, bd2), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor.T(), b_tensor));
    } else if (!aT && bT) {
      CHECK_EQ(ad2, bd2);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad1, bd1), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor, b_tensor.T()));
    } else {
      CHECK_EQ(ad2, bd1);
      Tensor<xpu, 2, DType> out_tensor = out.get_with_shape<xpu, 2, DType>(Shape2(ad1, bd2), s);
      ASSIGN_DISPATCH(out_tensor, req, dot(a_tensor, b_tensor));
    }
  });
}

/**
 * Calculates tensordot.
 */
template<typename xpu>
void TensordotImpl(const Tuple<int>& a_axes_summed,
                   const Tuple<int>& b_axes_summed,
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

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                   &b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                      b_axes_remained, b_axes_summed, a_shape, b_shape);

  mxnet::TShape a_temp_shape = GetReorderedShape(a_shape, a_axes);
  mxnet::TShape b_temp_shape = GetReorderedShape(b_shape, b_axes);

  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {  // 0-size input
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
      return;
    }

    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>
      (Shape1(a.Size() + b.Size()), s);
    DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + a.Size());
    TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask);
    TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask);

    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res,
                                  mxnet::TShape(a_axes.begin(), a_axes.end()));
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res,
                                  mxnet::TShape(b_axes.begin(), b_axes.end()));

    MatrixDot<xpu>(ctx, a_res, b_res, out, req[0], ad1, ad2, bd1, bd2);
  });
}

/**
 * forward function
 */
template<typename xpu>
void TensordotOpForward(const nnvm::NodeAttrs& attrs,
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

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);
  const Tuple<int>& a_axes_summed = param.a_axes_summed;
  const Tuple<int>& b_axes_summed = param.b_axes_summed;

  TensordotImpl<xpu>(a_axes_summed, b_axes_summed, ctx, a, b, out, req);
}

/**
 * gets shapes for inverse transpose.
 */
inline mxnet::TShape GetReverseShape(const mxnet::Tuple<int>& shape) {
  mxnet::TShape shape2(shape.begin(), shape.end());
  for (int i = 0; i < shape.ndim(); i++) {
    shape2[shape[i]] = i;
  }
  return shape2;
}

/**
 * calculates tensordot derivative.
 */
template<typename xpu>
void TensordotBackwardImpl(const Tuple<int>& a_axes_summed,
                           const Tuple<int>& b_axes_summed,
                           const OpContext& ctx,
                           const TBlob& out_grad,
                           const TBlob& a,
                           const TBlob& b,
                           const TBlob& grad_a,
                           const TBlob& grad_b,
                           const std::vector<OpReqType>& req) {
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  const mxnet::TShape& a_shape = a.shape_;
  const mxnet::TShape& b_shape = b.shape_;

  Tuple<int> a_axes_remained;
  Tuple<int> b_axes_remained;
  Tuple<int> a_axes;
  Tuple<int> b_axes;
  GetReorderedAxes(a_axes_summed, &a_axes_remained, &a_axes, b_axes_summed, &b_axes_remained,
                   &b_axes, a_shape, b_shape);

  int ad1 = 1, ad2 = 1, bd1 = 1, bd2 = 1;
  GetMatrixDimensions(&ad1, &ad2, &bd1, &bd2, a_axes_remained, a_axes_summed,
                      b_axes_remained, b_axes_summed, a_shape, b_shape);

  std::vector<int> a_T_axes;
  for (int i = 0; i < a_axes_summed.ndim(); i++) {
    a_T_axes.push_back(a_axes_summed[i]);
  }
  for (int i = 0; i < a_axes_remained.ndim(); i++) {
    a_T_axes.push_back(a_axes_remained[i]);
  }
  mxnet::TShape a_temp_shape(GetReorderedShape(a_shape, a_axes));
  mxnet::TShape a_T_temp_shape(GetReorderedShape(a_shape, a_T_axes));

  std::vector<int> b_T_axes;
  for (int i = 0; i < b_axes_remained.ndim(); i++) {
    b_T_axes.push_back(b_axes_remained[i]);
  }
  for (int i = 0; i < b_axes_summed.ndim(); i++) {
    b_T_axes.push_back(b_axes_summed[i]);
  }
  mxnet::TShape b_temp_shape(GetReorderedShape(b_shape, b_axes));
  mxnet::TShape b_T_temp_shape(GetReorderedShape(b_shape, b_T_axes));

  MSHADOW_REAL_TYPE_SWITCH(out_grad.type_flag_, DType, {
    Tensor<xpu, 1, DType> workspace = ctx.requested[0].get_space_typed<xpu, 1, DType>
      (Shape1((a.Size() + b.Size()) * 2), s);
    DType* a_ptr = reinterpret_cast<DType*>(workspace.dptr_);
    DType* a_ptr2 = reinterpret_cast<DType*>(workspace.dptr_ + a.Size());
    DType* b_ptr = reinterpret_cast<DType*>(workspace.dptr_ + 2 * a.Size());
    DType* b_ptr2 = reinterpret_cast<DType*>(workspace.dptr_ + 2 * a.Size() + b.Size());

    TBlob a_res = TBlob(a_ptr, a_temp_shape, xpu::kDevMask);
    TBlob b_res = TBlob(b_ptr, b_temp_shape, xpu::kDevMask);
    TBlob a_res2 = TBlob(a_ptr2, a_T_temp_shape, xpu::kDevMask);
    TBlob b_res2 = TBlob(b_ptr2, b_T_temp_shape, xpu::kDevMask);

    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a, a_res2,
                                  mxnet::TShape(a_T_axes.begin(), a_T_axes.end()));
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b, b_res2,
                                  mxnet::TShape(b_T_axes.begin(), b_T_axes.end()));

    MatrixDot<xpu>(ctx, a_res2, out_grad, b_res, req[1], ad2, ad1, ad1, bd2);
    MatrixDot<xpu>(ctx, out_grad, b_res2, a_res, req[0], ad1, bd2, bd2, bd1);

    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, a_res, grad_a, GetReverseShape(a_axes));
    mxnet::op::TransposeImpl<xpu>(ctx.run_ctx, b_res, grad_b, GetReverseShape(b_axes));
  });
}

/**
 * backward function.
 */
template<typename xpu>
void TensordotOpBackward(const nnvm::NodeAttrs& attrs,
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

  const TensordotParam& param = nnvm::get<TensordotParam>(attrs.parsed);
  const Tuple<int>& a_axes_summed = param.a_axes_summed;
  const Tuple<int>& b_axes_summed = param.b_axes_summed;

  TensordotBackwardImpl<xpu>(a_axes_summed, b_axes_summed, ctx, out_grad, a, b, grad_a,
                             grad_b, req);
}

struct TensordotIntAxesParam : public dmlc::Parameter<TensordotIntAxesParam> {
  int axes;
  DMLC_DECLARE_PARAMETER(TensordotIntAxesParam) {
    DMLC_DECLARE_FIELD(axes);
  }
};

/**
 * gets summed axes of a and b from parameter axes.
 */
inline void GetSummedAxes(mxnet::Tuple<int>* a_axes_summed_ptr,
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
void TensordotIntAxesImpl(const int axes,
                          const OpContext& ctx,
                          const TBlob& a,
                          const TBlob& b,
                          const TBlob& out,
                          const OpReqType req) {
  if (req == kNullOp) {
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
      if (req != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
      return;
    }

    MatrixDot<xpu>(ctx, a, b, out, req, ad1, ad2, bd1, bd2);
  });
}

/**
 * forward function
 */
template<typename xpu>
void TensordotIntAxesOpForward(const nnvm::NodeAttrs& attrs,
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

  TensordotIntAxesImpl<xpu>(axes, ctx, a, b, out, req[0]);
}

template<typename xpu>
void TensordotIntAxesBackwardImpl(const int axes,
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
void TensordotIntAxesOpBackward(const nnvm::NodeAttrs& attrs,
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

#endif  // MXNET_OPERATOR_NUMPY_NP_TENSORDOT_OP_INL_H_

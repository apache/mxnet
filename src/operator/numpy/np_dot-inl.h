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
 * \file np_dot-inl.h
 * \brief Function definition of matrix numpy-compatible dot operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_

#include <mxnet/operator_util.h>
#include "../tensor/dot-inl.h"
#include "../tensor/elemwise_binary_op.h"
#ifdef __CUDACC__
#include "./np_dot-inl.cuh"
#endif

namespace mxnet {
namespace op {

inline bool NumpyDotShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector *in_attrs,
                          mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& a_shape = in_attrs->at(0);
  const mxnet::TShape& b_shape = in_attrs->at(1);

  if (a_shape.ndim() == 1 && b_shape.ndim() == 1) {
    // Case 1: both 1-D arrays, inner product of vectors
    CHECK_EQ(a_shape[0], b_shape[0]);
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
  } else if (a_shape.ndim() == 2 && b_shape.ndim() == 2) {
    // Case 2: both 2-D arrays, matrix multiplication
    CHECK_EQ(a_shape[1], b_shape[0]);
    mxnet::TShape mm_shape(2, 0);
    mm_shape[0] = a_shape[0];
    mm_shape[1] = b_shape[1];
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mm_shape);
  } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
    // Case 3: both 0-D scalars, equivalent to multiply
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, mxnet::TShape(0, 0));
  } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
    // Case 3.5: either of them is a scalar, just scale by one of them
    mxnet::TShape oshape = (a_shape.ndim() == 0) ? b_shape : a_shape;
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);
  } else if (b_shape.ndim() == 1) {
    // Case 4: a is N-D array and b is 1-D array, sum product over the last axis
    CHECK_EQ(a_shape[a_shape.ndim() - 1], b_shape[0]);
    mxnet::TShape out_shape(a_shape.ndim() - 1, 0);
    for (int i = 0; i < a_shape.ndim() - 1; ++i) {
      out_shape[i] = a_shape[i];
    }
    SHAPE_ASSIGN_CHECK(*out_attrs, 0, out_shape);
  } else {
    // Case 5: a is N-D array and b is M-D array, sum product over the last axis
    //         of a and the 2nd-to-last axis of b
    LOG(FATAL) << "Case 5 not implemented yet...";
  }
  return true;
}

template<typename xpu>
inline void MMImpl(const OpContext& ctx,
                   const TBlob& a,
                   const TBlob& b,
                   const TBlob& out,
                   const OpReqType req,
                   const bool trans_a = false,
                   const bool trans_b = false) {
  using namespace mshadow;
  using namespace mshadow_op;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  int ma, na, mb, nb, m, n;
  na = a.size(a.ndim() - 1);
  ma = a.Size() / na;
  m = ma;
  mb = b.size(0);
  nb = b.Size() / mb;
  n = nb;
  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    Tensor<xpu, 2, DType> input0 = a.get_with_shape<xpu, 2, DType>(Shape2(ma, na), s);
    Tensor<xpu, 2, DType> input1 = b.get_with_shape<xpu, 2, DType>(Shape2(mb, nb), s);
    Tensor<xpu, 2, DType> output0 = out.get_with_shape<xpu, 2, DType>(Shape2(m, n), s);
    if (trans_a && trans_b) {
      ASSIGN_DISPATCH(output0, req, dot(input0.T(), input1.T()));
    } else if (!trans_a && trans_b) {
      ASSIGN_DISPATCH(output0, req, dot(input0, input1.T()));
    } else if (trans_a && trans_b) {
      ASSIGN_DISPATCH(output0, req, dot(input0.T(), input1));
    } else {
      ASSIGN_DISPATCH(output0, req, dot(input0, input1));
    }
  });
}

template<int req>
struct scalar_mul_kernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType *out, const DType* tensor, const DType *scalar) {
    KERNEL_ASSIGN(out[i], req, tensor[i] * scalar[0]);
  }
};

template<typename xpu>
inline void NumpyDotForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  if (req[0] == kNullOp) return;
  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Binary function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
      (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
      << "dot only supports float32/float64 for CPU, and float16/float32/float64 for GPU";
  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (a_shape.ndim() == 1 && b_shape.ndim() == 1) {
      // Case 1: both 1-D arrays, inner product of vectors
      if (out.type_flag_ == kFloat16) {
        MMImpl<xpu>(ctx, a, b, out, req[0]);
      } else {
        CHECK_NE(req[0], kAddTo) << "AddTo not yet supported";
        Tensor<xpu, 1, DType> mock_1d = out.get_with_shape<xpu, 1, DType>(Shape1(1), s);
        VectorDot(mock_1d, a.get<xpu, 1, DType>(s), b.get<xpu, 1, DType>(s));
      }
    } else if (a_shape.ndim() == 2 && b_shape.ndim() == 2) {
      // Case 2: both 2-D arrays, matrix multiplication
      MMImpl<xpu>(ctx, a, b, out, req[0]);
    } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Case 3: both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(out_data, req[0], a_data * b_data);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      const DType* tensor = (a_shape.ndim() == 0) ? b.dptr<DType>() : a.dptr<DType>();
      const DType* scalar = (a_shape.ndim() == 0) ? a.dptr<DType>() : b.dptr<DType>();
      // Case 3.5: either of them is a scalar, just scale by one of them
      MXNET_ASSIGN_REQ_SWITCH(req[0], Req, {
        Kernel<scalar_mul_kernel<Req>, xpu>::Launch(
          s, out.Size(), out.dptr<DType>(), tensor, scalar);
      });
    } else if (b_shape.ndim() == 1) {
      // Case 4: a is N-D array and b is 1-D array, sum product over the last axis
      MMImpl<xpu>(ctx, a, b, out, req[0]);
    } else {
      // Case 5: a is N-D array and b is M-D array, sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      LOG(FATAL) << "Case 5 not implemented yet...";
    }
  });
}

template<typename xpu>
inline void NumpyDotBackward(const nnvm::NodeAttrs& attrs,
                             const OpContext& ctx,
                             const std::vector<TBlob>& inputs,
                             const std::vector<OpReqType>& req,
                             const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow_op;

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    if (a_shape.ndim() == 1 && b_shape.ndim() == 1) {
      // Case 1: both 1-D arrays, inner product of vectors
      Tensor<xpu, 1, DType> out_grad = ograd.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> a_data = a.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> b_data = b.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> a_grad = grad_a.get<xpu, 1, DType>(s);
      Tensor<xpu, 1, DType> b_grad = grad_b.get<xpu, 1, DType>(s);
      ASSIGN_DISPATCH(a_grad, req[1],
                      broadcast_scalar(out_grad, a_data.shape_) * a_data);
      ASSIGN_DISPATCH(b_grad, req[0],
                      broadcast_scalar(out_grad, a_data.shape_) * b_data);
    } else if (a_shape.ndim() == 2 && b_shape.ndim() == 2) {
      // Case 2: both 2-D arrays, matrix multiplication
      MMImpl<xpu>(ctx, a, ograd, grad_b, req[1], true, false);
      MMImpl<xpu>(ctx, b, ograd, grad_a, req[0], false, true);
    } else if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
      // Case 3: both 0-D scalars, equivalent to multiply
      Tensor<xpu, 1, DType> out_grad = ograd.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> a_data = a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_data = b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> a_grad = grad_a.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> b_grad = grad_b.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      ASSIGN_DISPATCH(a_grad, req[0], b_data * out_grad);
      ASSIGN_DISPATCH(b_grad, req[1], a_data * out_grad);
    } else if (a_shape.ndim() == 0 || b_shape.ndim() == 0) {
      // Case 3.5: either of them is a scalar, just scale by one of them
      const TBlob& tensor = (a_shape.ndim() == 0) ? b : a;
      const TBlob& tensor_grad = (a_shape.ndim() == 0) ? grad_b : grad_a;
      const TBlob& scalar = (a_shape.ndim() == 0) ? a : b;
      const TBlob& scalar_grad = (a_shape.ndim() == 0) ? grad_a : grad_b;
      Tensor<xpu, 1, DType> scalar_ = scalar.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> scalar_grad_ = scalar_grad.get_with_shape<xpu, 1, DType>(Shape1(1), s);
      Tensor<xpu, 1, DType> tensor_ = tensor.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> tensor_grad_ = tensor_grad.FlatTo1D<xpu, DType>(s);
      Tensor<xpu, 1, DType> ograd_ = ograd.FlatTo1D<xpu, DType>(s);
      const OpReqType& tensor_req = (a_shape.ndim() == 0) ? req[1] : req[0];
      const OpReqType& scalar_req = (a_shape.ndim() == 0) ? req[0] : req[1];
      ASSIGN_DISPATCH(tensor_grad_, tensor_req, broadcast_scalar(scalar_, tensor_grad_.shape_) * ograd_);
    } else if (b_shape.ndim() == 1) {
      // Case 4: a is N-D array and b is 1-D array, sum product over the last axis
      MMImpl<xpu>(ctx, a, ograd, grad_b, req[1], true, false);
      MMImpl<xpu>(ctx, b, ograd, grad_a, req[0], false, true);
    } else {
      // Case 5: a is N-D array and b is M-D array, sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      LOG(FATAL) << "Case 5 not implemented yet...";
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_

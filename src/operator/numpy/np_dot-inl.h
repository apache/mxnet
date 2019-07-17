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
#include <vector>
#include "../tensor/dot-inl.h"
#include "../tensor/elemwise_binary_op.h"
#include "../tensor/broadcast_reduce_op.h"
#include "np_tensordot_op-inl.h"

namespace mxnet {
namespace op {

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
  if (out.shape_.Size() == 0U) return;  // zero-size tensor, no need to launch kernel
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
    if (a_shape.Size() == 0U || b_shape.Size() == 0U) {
      if (req[0] != kAddTo) {
        Tensor<xpu, 1, DType> out_data = out.get_with_shape<xpu, 1, DType>(
            Shape1(out.shape_.Size()), s);
        out_data = static_cast<DType>(0);
      }
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
    } else if (b_shape.ndim() < 3) {
      // Case 1, 2, 4, 5: a is N-D array (N >= 1) and b is vector or matrix, sum product
      //        over the last axis of a and the first axis of b
      TensordotIntAxesImpl<xpu>(1, ctx, a, b, out, req[0]);
    } else {
      // Case 5.5: a is N-D array and b is M-D array (M > 2), sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      const Tuple<int> a_axes_summed({a_shape.ndim() - 1});
      const Tuple<int> b_axes_summed({b_shape.ndim() - 2});
      TensordotImpl<xpu>(a_axes_summed, b_axes_summed, ctx, a, b, out, req);
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
  if (ograd.shape_.Size() == 0U) return;
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  const mxnet::TShape a_shape = a.shape_;
  const mxnet::TShape b_shape = b.shape_;
  if (a_shape.Size() == 0U || b_shape.Size() == 0U) return;

  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    if (a_shape.ndim() == 0 && b_shape.ndim() == 0) {
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
      ASSIGN_DISPATCH(tensor_grad_, tensor_req,
                      broadcast_scalar(scalar_, tensor_grad_.shape_) * ograd_);
      // TODO(haojin2): Get rid of temporary space.
      Tensor<xpu, 1, DType> temp_space =
        ctx.requested[0].get_space_typed<xpu, 1, DType>(Shape1(ograd.shape_.Size()), s);
      ASSIGN_DISPATCH(temp_space, kWriteTo, tensor_ * ograd_);

      ReduceAxesComputeImpl<xpu, mshadow_op::sum, true>(
        ctx, {TBlob(temp_space)}, {scalar_req}, {TBlob(scalar_grad_)}, scalar_grad_.shape_);
    } else if (b_shape.ndim() < 3) {
      // Case 1, 2, 4, 5: a is N-D array (N >= 1) and b is vector or matrix, sum product
      //        over the last axis of a and the first axis of b
      TensordotIntAxesBackwardImpl<xpu>(1, ctx, ograd, a, b, grad_a, grad_b, req);
    } else {
      // Case 5.5: a is N-D array and b is M-D array (M > 2), sum product over the last axis
      //         of a and the 2nd-to-last axis of b
      const Tuple<int> a_axes_summed({a_shape.ndim() - 1});
      const Tuple<int> b_axes_summed({b_shape.ndim() - 2});
      TensordotBackwardImpl<xpu>(a_axes_summed, b_axes_summed, ctx, ograd, a, b, grad_a,
          grad_b, req);
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_DOT_INL_H_

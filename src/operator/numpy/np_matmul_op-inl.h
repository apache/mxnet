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
 * \file np_matmul_op-inl.h
 * \brief Function definition of matrix numpy-compatible matmul operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_MATMUL_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_MATMUL_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <algorithm>
#include <memory>
#include "../mxnet_op.h"
#include "np_tensordot_op-inl.h"
#include "np_dot-inl.h"

namespace mxnet {
namespace op {

inline bool MatmulNeedBroadcast(const mxnet::TShape& ashape,
                                const mxnet::TShape& bshape) {
  bool need_bcast = false;
  for (int i = ashape.ndim() - 3, j = bshape.ndim() - 3;
       i >= 0 || j >=0; --i, --j) {
    if (i >= 0 && j >= 0) {
      need_bcast |= (ashape[i] != bshape[j]);
    } else if (i >= 0) {
      need_bcast |= (ashape[i] != 1);
    } else {
      need_bcast |= (bshape[j] != 1);
    }
  }
  return need_bcast;
}

/*!
 * \brief Get mshadow::Shape from mxnet::TShape.
 * \note fill ndim = 1 into extra ndims if outshape.ndim > input.ndim
 * \example i        0 1 2 3 4   (shape.ndim() = 5)
            shape    2 3 - - -   (N = 2)
            k_shape  1 1 2 3     (ndim = 4)
 * \tparam ndim - ndim of outshape.
 * \param shape - inshape.
 * \param N - the count of valid inshape ndims.
 */
template<int ndim>
mshadow::Shape<ndim> GetKernelShape(const mxnet::TShape& shape, size_t N) {
  mshadow::Shape<ndim>k_shape;
  for (int i = shape.ndim() - 1, j = N - 1; i >= 0 || j >= 0 ; --i, --j) {
    if (i >= 0) {
      k_shape[j] = shape[i];
    } else {
      k_shape[j] = 1;
    }
  }
  return k_shape;
}

/*!
 * \brief Broadcast in_shape to broadcast_shape in [dimstart, dimend].
          Make sure that before use this function:
          If in_shape[i] != broadcast_shape[i], in_shape[i] == 1.
 * \param N - ndim of both in_shape and broadcast_shape.
 * \param dimstart start dimension
 * \param dimend end dimension
 */
template<int ndim>
mshadow::Shape<ndim> GetBroadcastKernelShape(mshadow::Shape<ndim> in_shape,
                                             mshadow::Shape<ndim> broadcast_shape,
                                             int dimstart, int dimend) {
  CHECK_GE(dimstart, 0) << "dimstart must be >= 0, while received " << dimstart;
  CHECK_LT(dimend, ndim) << "dimend must be < " << ndim
                         << ", while received " << dimend;
  mshadow::Shape<ndim>out_shape(in_shape);
  for (int i = dimstart; i < dimend; ++i) {
    out_shape[i] = std::max(in_shape[i], broadcast_shape[i]);
  }
  return out_shape;
}

struct SumByShape {
  /*!
   * \brief squash input into output by addition
   * \example input.flatten.shape = (10, ),
              output.flatten.shape = (2, ),
              in_size = 10, out_size = 2, then,
              output[0] = sum(input[0, 2, 4, 6, 8])
              output[1] = sum(input[1, 3, 5, 7, 9])
   * \note in_size >= out_size
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* output, DType* input,
                                  size_t in_size, size_t out_size,
                                  const int req){
    // i is the global position in flattened output
    size_t pos = static_cast<size_t>(i);
    DType temp = 0;
    while (pos < in_size) {
      temp += input[pos];
      pos += out_size;
    }
    KERNEL_ASSIGN(output[i], req, temp);
  }
};

template<typename xpu, typename DType>
inline void MatmulImpl(const OpContext& ctx,
                       const TBlob& input_a, const TBlob& input_b,
                       const OpReqType& req, const TBlob& output,
                       Tensor<xpu, 1, char> temp_mem,
                       const size_t ndim, const size_t batch_size,
                       const size_t bc_size_a, const size_t bc_size_b,
                       const mxnet::TShape& a_shape,
                       const mxnet::TShape& b_shape,
                       const mxnet::TShape& out_shape,
                       const bool TA, const bool TB) {
  using namespace mshadow;
  using namespace mxnet_op;
  mshadow::Tensor<xpu, 1, DType*> workspace;
  mshadow::Tensor<xpu, 3, DType> ans, mlhs, mrhs;
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  if (MatmulNeedBroadcast(a_shape, b_shape)) {
    // e.g. a.shape = (2, 3, 1, 4, 2)
    //      b.shape =       (5, 2, 4)
    //      c = matmul(a, b), need to broadcast a and b
    //      c.shape = (2, 3, 5, 4, 4)
    mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> k_a_shape =
      GetKernelShape<MXNET_SPECIAL_MAX_NDIM>(a_shape, ndim);
    mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> k_b_shape =
      GetKernelShape<MXNET_SPECIAL_MAX_NDIM>(b_shape, ndim);
    mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> k_out_shape =
      GetKernelShape<MXNET_SPECIAL_MAX_NDIM>(out_shape, ndim);
    const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> k_a_shape_bc =
      GetBroadcastKernelShape<MXNET_SPECIAL_MAX_NDIM>(k_a_shape, k_out_shape, 0, ndim - 2);
    const mshadow::Shape<MXNET_SPECIAL_MAX_NDIM> k_b_shape_bc =
      GetBroadcastKernelShape<MXNET_SPECIAL_MAX_NDIM>(k_b_shape, k_out_shape, 0, ndim - 2);
    DType* bc_a_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
    DType* bc_b_ptr = bc_a_ptr + bc_size_a;
    MSHADOW_TYPE_SWITCH_WITH_BOOL(input_a.type_flag_, IType, {
      MSHADOW_TYPE_SWITCH_WITH_BOOL(input_b.type_flag_, OType, {
        Kernel<broadcast_kernel<mshadow_op::identity>, xpu>::Launch(
          s, bc_size_a, input_a.dptr<IType>(), bc_a_ptr,
          k_a_shape, k_a_shape_bc, OpReqType::kWriteTo, ndim);
        Kernel<broadcast_kernel<mshadow_op::identity>, xpu>::Launch(
          s, bc_size_b, input_b.dptr<IType>(), bc_b_ptr,
          k_b_shape, k_b_shape_bc, OpReqType::kWriteTo, ndim);
      });
    });
    ans = mshadow::Tensor<xpu, 3, DType>(output.dptr<DType>(),
      Shape3(batch_size, k_out_shape[ndim - 2], k_out_shape[ndim - 1]), s);
    mlhs = mshadow::Tensor<xpu, 3, DType>(bc_a_ptr,
      Shape3(batch_size, k_a_shape_bc[ndim - 2], k_a_shape_bc[ndim - 1]), s);
    mrhs = mshadow::Tensor<xpu, 3, DType>(bc_b_ptr,
      Shape3(batch_size, k_b_shape_bc[ndim - 2], k_b_shape_bc[ndim - 1]), s);
    DType** workspace_ptr = reinterpret_cast<DType**>(bc_b_ptr + bc_size_b);
    workspace = mshadow::Tensor<xpu, 1, DType*>(workspace_ptr, Shape1(3 * ans.size(0)), s);
  } else {
    ans = output.get_with_shape<xpu, 3, DType>(
      Shape3(batch_size, out_shape[ndim - 2], out_shape[ndim - 1]), s);
    mlhs = input_a.get_with_shape<xpu, 3, DType>(
      Shape3(batch_size, (a_shape.ndim() == 1) ? 1 : a_shape[a_shape.ndim() - 2],
             a_shape[a_shape.ndim() - 1]), s);
    mrhs = input_b.get_with_shape<xpu, 3, DType>(
      Shape3(batch_size, b_shape[b_shape.ndim() - 2], b_shape[b_shape.ndim() - 1]), s);
    workspace = ctx.requested[0].get_space_typed<xpu, 1, DType*>
      (mshadow::Shape1(3 * ans.size(0)), s);
  }
  if (TA && TB) {
    mshadow::BatchGEMM<true, true>(ans, mlhs, mrhs, (DType)1.0f,
                                   (kAddTo == req) ? (DType)1.0f : (DType)0.0f,
                                   workspace);
  } else if (TA && !TB) {
    mshadow::BatchGEMM<true, false>(ans, mlhs, mrhs, (DType)1.0f,
                                    (kAddTo == req) ? (DType)1.0f : (DType)0.0f,
                                    workspace);
  } else if (!TA && TB) {
    mshadow::BatchGEMM<false, true>(ans, mlhs, mrhs, (DType)1.0f,
                                    (kAddTo == req) ? (DType)1.0f : (DType)0.0f,
                                    workspace);
  } else {
    mshadow::BatchGEMM<false, false>(ans, mlhs, mrhs, (DType)1.0f,
                                     (kAddTo == req) ? (DType)1.0f : (DType)0.0f,
                                     workspace);
  }
}

template<typename xpu>
void NumpyMatmulForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) return;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_NE(a.shape_.ndim(), 0)
    << "Multiplication by scalars is not allowed.\n";
  CHECK_NE(b.shape_.ndim(), 0)
    << "Multiplication by scalars is not allowed.\n";
  CHECK_EQ(out.type_flag_, a.type_flag_)
      << "Mstmul function only support input/output with the same type";
  CHECK_EQ(out.type_flag_, b.type_flag_)
      << "Matmul function only support input/output with the same type";
  CHECK(out.type_flag_ == kFloat32 || out.type_flag_ == kFloat64 ||
    (out.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
    << "Matmul only supports float32/float64 for CPU, and float16/float32/float64 for GPU";

  if ((a.shape_.Size() == 0) || (b.shape_.Size() == 0)) {
    MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
      Kernel<mxnet_op::set_zero, xpu>::Launch(
        s, outputs[0].shape_.Size(), outputs[0].dptr<DType>());
    });
    return;
  }

  mxnet::TShape a_shape = (a.shape_.ndim() == 1) ? Shape2(1, a.shape_.Size()) : a.shape_;
  mxnet::TShape b_shape = (b.shape_.ndim() == 1) ? Shape2(b.shape_.Size(), 1) : b.shape_;
  mxnet::TShape out_shape = out.shape_;
  size_t ndim = out_shape.ndim();
  if ((a.shape_.ndim() == 1) && (b.shape_.ndim() == 1)) {
    ndim = 2;
    std::vector<size_t> newshape({1, 1});
    out_shape.assign(newshape.begin(), newshape.end());
  } else if ((a.shape_.ndim() == 1) && (b.shape_.ndim() != 1)) {
    ndim = out_shape.ndim() + 1;
    std::vector<size_t> newshape(ndim);
    for (size_t i = 0; i < ndim - 2; ++i) {
      newshape[i] = out_shape[i];
    }
    newshape[ndim - 2] = 1;
    newshape[ndim - 1] = out_shape[ndim - 2];
    out_shape.assign(newshape.begin(), newshape.end());
  } else if ((a.shape_.ndim() != 1) && (b.shape_.ndim() == 1)) {
    ndim = out_shape.ndim() + 1;
    std::vector<size_t> newshape(ndim);
    for (size_t i = 0; i < ndim - 1; ++i) {
      newshape[i] = out_shape[i];
    }
    newshape[ndim - 1] = 1;
    out_shape.assign(newshape.begin(), newshape.end());
  }
  MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    size_t batch_size = out_shape.ProdShape(0, ndim - 2);
    size_t bc_size_a = batch_size * a_shape[a_shape.ndim() - 2] * a_shape[a_shape.ndim() - 1];
    size_t bc_size_b = batch_size * b_shape[b_shape.ndim() - 2] * b_shape[b_shape.ndim() - 1];
    size_t temp_mem_size = (bc_size_a + bc_size_b) * sizeof(DType) +
                           3 * batch_size * sizeof(DType*);
    Tensor<xpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    MatmulImpl<xpu, DType>(ctx, inputs[0], inputs[1], req[0], outputs[0], temp_mem,
                           ndim, batch_size, bc_size_a, bc_size_b,
                           a_shape, b_shape, out_shape, false, false);
  });
}


template<typename xpu>
void NumpyMatmulBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  if (req[0] == kNullOp && req[1] == kNullOp) return;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK(grad_a.type_flag_ == kFloat32 || grad_a.type_flag_ == kFloat64 ||
        (grad_a.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
    << "Matmul only supports float32/float64 for CPU, and float16/float32/float64 for GPU";
  CHECK(grad_b.type_flag_ == kFloat32 || grad_b.type_flag_ == kFloat64 ||
        (grad_b.type_flag_ == kFloat16 && ctx.run_ctx.ctx.dev_mask() == mshadow::gpu::kDevMask))
    << "Matmul only supports float32/float64 for CPU, and float16/float32/float64 for GPU";
  // ograd.shape_.Size() == 0 if and only if a.shape_.Size() == 0 && b.shape_.Size() == 0
  if (a.shape_.Size() == 0 && b.shape_.Size() == 0) return;
  if (a.shape_.Size() == 0) {  // a.shape_.Size() == 0 && b.shape_.Size() != 0
    if (req[1] == kWriteTo) {
      MSHADOW_REAL_TYPE_SWITCH(outputs[1].type_flag_, DType, {
        Kernel<mxnet_op::set_zero, xpu>::Launch(s, grad_b.shape_.Size(), grad_b.dptr<DType>());
      });
    }
    return;
  }
  if (b.shape_.Size() == 0) {  // b.shape_.Size() == 0 && a.shape_.Size() != 0
    if (req[0] == kWriteTo) {
      MSHADOW_REAL_TYPE_SWITCH(outputs[0].type_flag_, DType, {
        Kernel<mxnet_op::set_zero, xpu>::Launch(s, grad_a.shape_.Size(), grad_a.dptr<DType>());
      });
    }
    return;
  }

  mxnet::TShape a_shape = (a.shape_.ndim() == 1) ? Shape2(1, a.shape_.Size()) : a.shape_;
  mxnet::TShape b_shape = (b.shape_.ndim() == 1) ? Shape2(b.shape_.Size(), 1) : b.shape_;
  mxnet::TShape out_shape = ograd.shape_;
  size_t ndim = out_shape.ndim();
  if ((a.shape_.ndim() == 1) && (b.shape_.ndim() == 1)) {
    // e.g. a.shape = (x), b.shape = (x)
    //      c = matmul(a, b)
    //      a.shape -> (1, x), b.shape -> (x, 1)
    //      newshape = (1, 1) -> c.shape = ()
    ndim = 2;
    std::vector<size_t> newshape({1, 1});
    out_shape.assign(newshape.begin(), newshape.end());
  } else if ((a.shape_.ndim() == 1) && (b.shape_.ndim() != 1)) {
    // e.g. a.shape = (x), b.shape = (..., x, y)
    //      c = matmul(a, b)
    //      a.shape -> (1, x)
    //      newshape = (..., 1, y) -> c.shape = (..., y)
    ndim = out_shape.ndim() + 1;
    std::vector<size_t> newshape(ndim);
    for (size_t i = 0; i < ndim - 2; ++i) {
      newshape[i] = out_shape[i];
    }
    newshape[ndim - 2] = 1;
    newshape[ndim - 1] = out_shape[out_shape.ndim() - 1];
    out_shape.assign(newshape.begin(), newshape.end());
  } else if ((a.shape_.ndim() != 1) && (b.shape_.ndim() == 1)) {
    // e.g. a.shape = (..., x, y), b.shape = (y)
    //      c = matmul(a, b)
    //      b.shape -> (y, 1)
    //      newshape = (..., y, 1) -> c.shape = (..., y)
    ndim = out_shape.ndim() + 1;
    std::vector<size_t> newshape(ndim);
    for (size_t i = 0; i < ndim - 1; ++i) {
      newshape[i] = out_shape[i];
    }
    newshape[ndim - 1] = 1;
    out_shape.assign(newshape.begin(), newshape.end());
  }
  std::vector<size_t> vec_grad_a_shape(ndim, -1);
  std::vector<size_t> vec_grad_b_shape(ndim, -1);
  for (unsigned int i = 0; i < ndim - 2; ++i) {
    vec_grad_a_shape[i] = out_shape[i];
    vec_grad_b_shape[i] = out_shape[i];
  }
  vec_grad_a_shape[ndim - 2] = a_shape[a_shape.ndim() - 2];
  vec_grad_a_shape[ndim - 1] = a_shape[a_shape.ndim() - 1];
  mxnet::TShape grad_a_shape = mxnet::TShape(vec_grad_a_shape.begin(), vec_grad_a_shape.end());
  vec_grad_b_shape[ndim - 2] = b_shape[b_shape.ndim() - 2];
  vec_grad_b_shape[ndim - 1] = b_shape[b_shape.ndim() - 1];
  mxnet::TShape grad_b_shape = mxnet::TShape(vec_grad_b_shape.begin(), vec_grad_b_shape.end());
  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    size_t batch_size =
      out_shape.ProdShape(0, ndim - 2);
    size_t bc_size_a =
      batch_size * a_shape[a_shape.ndim() - 2] * a_shape[a_shape.ndim() - 1];
    size_t bc_size_b =
      batch_size * b_shape[b_shape.ndim() - 2] * b_shape[b_shape.ndim() - 1];
    size_t bc_size_out =
      batch_size * out_shape[out_shape.ndim() - 2] * out_shape[out_shape.ndim() - 1];

    size_t temp_mem_size_grada = (bc_size_out + bc_size_b) * sizeof(DType) +
                                 3 * batch_size * sizeof(DType*);
    size_t temp_mem_size_gradb = (bc_size_a + bc_size_out) * sizeof(DType) +
                                 3 * batch_size * sizeof(DType*);
    size_t temp_size_grada = bc_size_a * sizeof(DType);
    size_t temp_size_gradb = bc_size_b * sizeof(DType);
    size_t temp_mem_size = temp_mem_size_grada + temp_mem_size_gradb +
                           temp_size_grada + temp_size_gradb;
    Tensor<xpu, 1, char> temp_mem =
      ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
    Tensor<xpu, 1, char> workspace_grada(temp_mem.dptr_, Shape1(temp_mem_size_grada), s);
    Tensor<xpu, 1, char> workspace_gradb(workspace_grada.dptr_ + temp_mem_size_grada,
                                         Shape1(temp_mem_size_gradb), s);
    Tensor<xpu, 1, DType> temp_grada(
      reinterpret_cast<DType*>(workspace_gradb.dptr_ + temp_mem_size_gradb),
      Shape1(bc_size_a), s);
    Tensor<xpu, 1, DType> temp_gradb(
      reinterpret_cast<DType*>(temp_grada.dptr_ + bc_size_a),
      Shape1(bc_size_b), s);
    MatmulImpl<xpu, DType>(ctx, ograd, b, kWriteTo, temp_grada, workspace_grada,
                           ndim, batch_size, bc_size_out, bc_size_b,
                           out_shape, b_shape, grad_a_shape, false, true);
    MatmulImpl<xpu, DType>(ctx, a, ograd, kWriteTo, temp_gradb, workspace_gradb,
                           ndim, batch_size, bc_size_a, bc_size_out,
                           a_shape, out_shape, grad_b_shape, true, false);
    Kernel<SumByShape, xpu>::Launch(
      s, a_shape.Size(), grad_a.dptr<DType>(), temp_grada.dptr_,
      bc_size_a, a_shape.Size(), req[0]);
    Kernel<SumByShape, xpu>::Launch(
      s, b_shape.Size(), grad_b.dptr<DType>(), temp_gradb.dptr_,
      bc_size_b, b_shape.Size(), req[1]);
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATMUL_OP_INL_H_

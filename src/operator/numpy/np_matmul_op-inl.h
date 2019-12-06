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
#include "np_tensordot_op-inl.h"
#include "np_dot-inl.h"

namespace mxnet {
namespace op {

template<int ndim>
mshadow::Shape<ndim> GetStride(mshadow::Shape<ndim> shape, size_t N) {
  /*!
   * \brief Calculate stride of each dim from shape 
   */
  mshadow::Shape<ndim>stride;
  size_t tmp = 1;
  for (int i = N - 1; i >= 0; --i) {
    stride[i] = tmp;
    tmp *= shape[i];
  }
  return stride;
}

template<int ndim>
mshadow::Shape<ndim> GetKernelShape(const mxnet::TShape& shape,
                                    size_t N, bool T = false) {
  /*!
   * \brief Get mshadow::Shape from mxnet::TShape. Extra dims is filled with 1.
   * \param N - ndim of mshape::Shape shape.
   * \param T - If T is True, transpose the last two axis, otherwise not.
   */
  mshadow::Shape<ndim>k_shape;
  for (int i = shape.ndim() - 1, j = N - 1; i >= 0 || j >= 0 ; --i, --j) {
    if (i >= 0) {
      k_shape[j] = shape[i];
    } else {
      k_shape[j] = 1;
    }
  }
  if (T) {  // transpose the latest two axes
    size_t t = k_shape[N - 1];
    k_shape[N - 1] = k_shape[N - 2];
    k_shape[N - 2] = t;
  }
  return k_shape;
}

template<int ndim>
mshadow::Shape<ndim> BroadcastKernelShape(mshadow::Shape<ndim> in_shape,
                                          mshadow::Shape<ndim> broadcast_shape,
                                          size_t N, size_t* size) {
  /*!
   * \brief Broadcast in_shape(ndim = N) to broadcast_shape(ndim = N) expect the last two axes.
            Make sure that: If i < N - 2 and in_shape[i] != broadcast_shape[i], in_shape[i] == 1.
   * \param N - ndim of both in_shape and broadcast_shape.
   * \param size - The size of the broadcast_shape.
   */
  mshadow::Shape<ndim>out_shape(in_shape);
  *size = 1;
  for (size_t i = 0; i < N - 2; ++i) {
    out_shape[i] = std::max(in_shape[i], broadcast_shape[i]);
    *size *= out_shape[i];
  }
  *size *= (out_shape[N - 2] * out_shape[N - 1]);
  return out_shape;
}

template<int req>
struct NDMatmul {
  /*!
   * \brief matmul(a, b) in both N-D(N >= 2) case.
            It is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
   * \param out - output: insert 'value' to 'arr' according to 'index'.
   * \param a - input: the first argument.
   * \param b - input: the second argument.
   * \param ndim - ndim of a, b and output. Because of broadcast, regard their ndim as equal.  
   */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out,
                                  const DType* a, const DType* b,
                                  const mshadow::Shape<10> a_stride,
                                  const mshadow::Shape<10> b_stride,
                                  const mshadow::Shape<10> out_stride,
                                  const mshadow::Shape<10> a_shape,
                                  const mshadow::Shape<10> b_shape,
                                  const mshadow::Shape<10> out_shape,
                                  const size_t ndim){
    // i is the global flatten index in the output
    mshadow::Shape<10> out_idx;  // position in output's shape
    for (size_t j = 0; j < ndim; ++j) {
      const int64_t head = i / out_stride[j];
      const int64_t mid = head % out_shape[j];
      out_idx[j] = mid;
    }
    mshadow::Shape<10> a_idx(out_idx);  // data block position in a's shape
    size_t a_pos = 0;
    for (size_t j = 0; j < ndim - 2; ++j) {
      a_idx[j] = (a_shape[j] == 1) ? 0 : a_idx[j];  // broadcast
      a_pos += a_idx[j] * a_stride[j];
    }
    a_pos += out_idx[ndim - 2] * a_stride[ndim - 2];
    mshadow::Shape<10> b_idx(out_idx);  // data block position in b's shape
    size_t b_pos = 0;
    for (size_t j = 0; j < ndim - 2; ++j) {
      b_idx[j] = (b_shape[j] == 1) ? 0 : b_idx[j];  // broadcast
      b_pos += b_idx[j] * b_stride[j];
    }
    b_pos += out_idx[ndim - 1];
    DType res = 0;
    for (int j = 0; j < a_shape[ndim - 1]; ++j) {
      res += a[a_pos + j] * b[b_pos + j * b_stride[ndim - 2]];
    }
    KERNEL_ASSIGN(out[i], req, res);
  }
};

struct TransposeLastTwoAxes {
  /*!
  * \brief transpose the last two axes
  * \example (a, b, c, d) -> (a, b, d, c)
  * \param in_row - the second-to-last dim size of in's shape
  * \param in_col - the last dim size of in's shape
  */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* in,
                                  const size_t in_row, const size_t in_col) {
    // i is the global position in flattened output
    const size_t out_col = in_row;
    const size_t last = i % (in_row * in_col);
    const size_t base = i - last;
    const size_t row = last / out_col;
    const size_t col = last % out_col;
    const size_t dest = base + col * in_col + row;
    out[i] = in[dest];
  }
};

template<int req>
struct SumByShape {
  /*!
  * \brief 
  * \param out - output: insert 'value' to 'arr' according to 'index'.
  * \param a - input: the first argument.
  * \param b - input: the second argument.
  * \param ndim - ndim of a, b and output. Because of broadcast, regard their ndim as equal.  
  */
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* output, DType* input,
                                  size_t in_size, size_t out_size){
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

template<typename xpu>
void NumpyMatmulForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;

  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);

  const TBlob& a = inputs[0];
  const TBlob& b = inputs[1];
  const TBlob& out = outputs[0];
  mxnet::TShape a_shape = a.shape_;
  mxnet::TShape b_shape = b.shape_;
  mxnet::TShape out_shape = out.shape_;
  size_t ndim = out_shape.ndim();
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  CHECK_NE(a_shape.ndim(), 0)
    << "Multiplication by scalars is not allowed.\n";
  CHECK_NE(b_shape.ndim(), 0)
    << "Multiplication by scalars is not allowed.\n";
  MSHADOW_REAL_TYPE_SWITCH(out.type_flag_, DType, {
    if (b_shape.ndim() <= 2) {
      // case 1: both 1-D arrays, inner product of vectors
      // case 2: both 2-D arrays, matrix multiplication
      // case 3: If the second argument is 1-D.
      // case 5(1): If the first argument is N-D(N > 2), the second argument is 2-D.
      TensordotIntAxesImpl<xpu>(1, ctx, a, b, out, req[0]);
    } else {
      // Case 4: If the first argument is 1-D.
      if (a_shape.ndim() == 1) {
        ndim += 1;
        std::vector<size_t> newshape(ndim);
        for (size_t i = 0; i < ndim - 2; ++i) {
          newshape[i] = out_shape[i];
        }
        newshape[ndim - 2] = 1;
        newshape[ndim - 1] = out_shape[ndim - 2];
        out_shape.assign(newshape.begin(), newshape.end());
      }
      // case 5(2): If the first argument is N-D(N > 2), the second argument is N-D(N > 2).
      const mshadow::Shape<10> k_a_shape = GetKernelShape<10>(a_shape, ndim);
      const mshadow::Shape<10> k_b_shape = GetKernelShape<10>(b_shape, ndim);
      const mshadow::Shape<10> k_out_shape = GetKernelShape<10>(out_shape, ndim);
      const mshadow::Shape<10> a_stride = GetStride<10>(k_a_shape, ndim);
      const mshadow::Shape<10> b_stride = GetStride<10>(k_b_shape, ndim);
      const mshadow::Shape<10> out_stride = GetStride<10>(k_out_shape, ndim);
      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<NDMatmul<req_type>, xpu>::Launch(
          s, out_shape.Size(), out.dptr<DType>(), a.dptr<DType>(),
          b.dptr<DType>(), a_stride, b_stride, out_stride,
          k_a_shape, k_b_shape, k_out_shape, ndim);
      });
    }
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

  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 2U);

  const TBlob& ograd = inputs[0];
  const TBlob& a = inputs[1];
  const TBlob& b = inputs[2];
  const TBlob& grad_a = outputs[0];
  const TBlob& grad_b = outputs[1];
  mxnet::TShape a_shape = a.shape_;
  mxnet::TShape b_shape = b.shape_;
  mxnet::TShape out_shape = ograd.shape_;
  size_t ndim = out_shape.ndim();
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    if (b_shape.ndim() <= 2) {
      // case 1: both 1-D arrays, inner product of vectors
      // case 2: both 2-D arrays, matrix multiplication
      // case 3: If the second argument is 1-D.
      // case 5(1): If the first argument is N-D(N > 2), the second argument is 2-D.
      TensordotIntAxesBackwardImpl<xpu>(1, ctx, ograd, a, b, grad_a, grad_b, req);
    } else {
      // Case 4: If the first argument is 1-D.
      if (a_shape.ndim() == 1) {
        std::vector<size_t>temp_shape({1, a_shape.Size()});
        a_shape.assign(temp_shape.begin(), temp_shape.end());
        ndim += 1;
        std::vector<size_t> newshape(ndim);
        for (size_t i = 0; i < ndim - 2; ++i) {
          newshape[i] = out_shape[i];
        }
        newshape[ndim - 2] = 1;
        newshape[ndim - 1] = out_shape[ndim - 2];
        out_shape.assign(newshape.begin(), newshape.end());
      }
      // case 5(2): If the first argument is N-D(N > 2), the second argument is N-D(N > 2).
      const mshadow::Shape<10> k_a_shape = GetKernelShape<10>(a_shape, ndim);
      const mshadow::Shape<10> k_a_shape_T = GetKernelShape<10>(a_shape, ndim, true);
      const mshadow::Shape<10> k_b_shape = GetKernelShape<10>(b_shape, ndim);
      const mshadow::Shape<10> k_b_shape_T = GetKernelShape<10>(b_shape, ndim, true);
      const mshadow::Shape<10> k_out_shape = GetKernelShape<10>(out_shape, ndim);
      size_t bc_size_a = 1, bc_size_b = 1;
      const mshadow::Shape<10> k_a_shape_bc =
        BroadcastKernelShape<10>(k_a_shape, k_out_shape, ndim, &bc_size_a);
      const mshadow::Shape<10> k_b_shape_bc =
        BroadcastKernelShape<10>(k_b_shape, k_out_shape, ndim, &bc_size_b);
      const mshadow::Shape<10> a_stride_T = GetStride<10>(k_a_shape_T, ndim);
      const mshadow::Shape<10> a_stride_bc = GetStride<10>(k_a_shape_bc, ndim);
      const mshadow::Shape<10> b_stride_T = GetStride<10>(k_b_shape_T, ndim);
      const mshadow::Shape<10> b_stride_bc = GetStride<10>(k_b_shape_bc, ndim);
      const mshadow::Shape<10> out_stride = GetStride<10>(k_out_shape, ndim);

      size_t temp_mem_size = (a_shape.Size() + b_shape.Size() +
                              bc_size_a + bc_size_b) * sizeof(DType);
      Tensor<xpu, 1, char> temp_mem =
        ctx.requested[0].get_space_typed<xpu, 1, char>(Shape1(temp_mem_size), s);
      DType* a_T_ptr = reinterpret_cast<DType*>(temp_mem.dptr_);
      DType* b_T_ptr = reinterpret_cast<DType*>(
        temp_mem.dptr_ + a_shape.Size() * sizeof(DType));
      DType* grad_a_temp = reinterpret_cast<DType*>(
        temp_mem.dptr_ + (a_shape.Size() + b_shape.Size()) * sizeof(DType));
      DType* grad_b_temp = reinterpret_cast<DType*>(
        temp_mem.dptr_ + (a_shape.Size() + b_shape.Size() + bc_size_a) * sizeof(DType));

      Kernel<TransposeLastTwoAxes, xpu>::Launch(
        s, a_shape.Size(), a_T_ptr, a.dptr<DType>(),
        static_cast<size_t>(k_a_shape[ndim - 2]),
        static_cast<size_t>(k_a_shape[ndim - 1]));
      Kernel<TransposeLastTwoAxes, xpu>::Launch(
        s, b_shape.Size(), b_T_ptr, b.dptr<DType>(),
        static_cast<size_t>(k_b_shape[ndim - 2]),
        static_cast<size_t>(k_b_shape[ndim - 1]));

      Kernel<NDMatmul<OpReqType::kWriteTo>, xpu>::Launch(
        s, bc_size_a, grad_a_temp, ograd.dptr<DType>(),
        b_T_ptr, out_stride, b_stride_T, a_stride_bc,
        k_out_shape, k_b_shape_T, k_a_shape_bc, ndim);
      Kernel<NDMatmul<OpReqType::kWriteTo>, xpu>::Launch(
        s, bc_size_b, grad_b_temp, a_T_ptr,
        ograd.dptr<DType>(), a_stride_T, out_stride, b_stride_bc,
        k_a_shape_T, k_out_shape, k_b_shape_bc, ndim);

      MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<SumByShape<req_type>, xpu>::Launch(
          s, a_shape.Size(), grad_a.dptr<DType>(), grad_a_temp,
          bc_size_a, a_shape.Size());
        Kernel<SumByShape<req_type>, xpu>::Launch(
          s, b_shape.Size(), grad_b.dptr<DType>(), grad_b_temp,
          bc_size_b, b_shape.Size());
      });
    }
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_MATMUL_OP_INL_H_

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
 *  Copyright (c) 2015 by Contributors
 * \file matrix_op.cu
 * \brief GPU Implementation of matrix operations
 */
#include <cub/cub.cuh>
#include "./matrix_op-inl.h"
#include "./elemwise_unary_op.h"


namespace mxnet {
namespace op {

/*!
 * \brief Compute the number of elements of every row.
 */
struct SliceMarkCsrIndPtr {
  /*! 
   * \brief
   * \param i           the i-th row of the output csr ndarray
   * \param prefix_sum  indptr array of the output csr ndarray
   * \param in_idx      indices array of the input csr ndarray
   * \param in_indptr   indptr array of the input csr ndarray
   * \param begin_col   starting indice
   * \param end_col     ending indice
   */
  template<typename IType, typename RType>
  MSHADOW_XINLINE static void Map(int i,
                                  RType* prefix_sum,
                                  const IType* in_idx,
                                  const RType* in_indptr,
                                  const int begin_col, const int end_col) {
    if (i == 0) {
      prefix_sum[0] = 0;
    }
    RType size = 0;
    for (RType j = in_indptr[i]; j < in_indptr[i+1]; j++) {
      // indices of CSRNDArray are in ascending order per row
      if (in_idx[j] >= end_col) {
        break;
      } else if (in_idx[j] >= begin_col) {
        size++;
      }
    }
    prefix_sum[i+1] = size;
  }
};


template<>
void SliceDimTwoCsrImpl<gpu>(const TShape &begin, const TShape &end, const OpContext& ctx,
                             const NDArray &in, const NDArray &out) {
  using namespace mshadow;
  using namespace mxnet_op;
  using namespace csr;

  Stream<gpu> *s = ctx.get_stream<gpu>();

  nnvm::dim_t begin_row = begin[0], end_row = end[0];
  nnvm::dim_t begin_col = begin[1], end_col = end[1];
  nnvm::dim_t indptr_len = end_row - begin_row + 1;
  out.CheckAndAllocAuxData(kIndPtr, Shape1(indptr_len));
  // assume idx indptr share the same type
  MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIndPtr), RType, {
    MSHADOW_IDX_TYPE_SWITCH(in.aux_type(kIdx), IType, {
      MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
        RType *in_indptr = in.aux_data(kIndPtr).dptr<RType>();
        IType *in_idx = in.aux_data(kIdx).dptr<IType>();
        DType *in_data = in.data().dptr<DType>();

        RType *out_indptr = out.aux_data(kIndPtr).dptr<RType>();

        Kernel<SliceMarkCsrIndPtr, gpu>::Launch(s, indptr_len - 1,
                                                out_indptr,
                                                in_idx,
                                                in_indptr + begin_row,
                                                begin_col, end_col);
        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        Tensor<gpu, 1, char> workspace = ctx.requested[0]
            .get_space_typed<gpu, 1, char>(Shape1(temp_storage_bytes), s);
        d_temp_storage = workspace.dptr_;

        cub::DeviceScan::InclusiveSum(d_temp_storage,
                                      temp_storage_bytes,
                                      out_indptr,
                                      out_indptr,
                                      indptr_len,
                                      Stream<gpu>::GetStream(s));
        // retrieve nnr
        RType nnr = 0;
        CUDA_CALL(cudaMemcpy(&nnr, &out_indptr[indptr_len-1], sizeof(RType),
            cudaMemcpyDeviceToHost));

        // returns zeros in csr format if nnr = 0
        if (nnr == 0) {
          out.set_aux_shape(kIdx, Shape1(0));
          return;
        }
        out.CheckAndAllocAuxData(kIdx, Shape1(nnr));
        out.CheckAndAllocData(Shape1(nnr));
        IType *out_idx = out.aux_data(kIdx).dptr<IType>();
        DType *out_data = out.data().dptr<DType>();

        Kernel<SliceDimTwoCsrAssign, gpu>::Launch(s, indptr_len - 1, out_idx, out_data,
                                                  out_indptr, in_idx, in_data,
                                                  in_indptr + begin_row,
                                                  begin_col, end_col);
      });
    });
  });
}


NNVM_REGISTER_OP(Reshape)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(Flatten)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(transpose)
.set_attr<FCompute>("FCompute<gpu>", Transpose<gpu>);

NNVM_REGISTER_OP(expand_dims)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpForward<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", SliceEx<gpu>);

NNVM_REGISTER_OP(_backward_slice)
.set_attr<FCompute>("FCompute<gpu>", SliceOpBackward<gpu>);

NNVM_REGISTER_OP(_slice_assign)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignOpForward<gpu>);

NNVM_REGISTER_OP(_slice_assign_scalar)
.set_attr<FCompute>("FCompute<gpu>", SliceAssignScalarOpForward<gpu>);

NNVM_REGISTER_OP(slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxis<gpu>);

NNVM_REGISTER_OP(_backward_slice_axis)
.set_attr<FCompute>("FCompute<gpu>", SliceAxisGrad_<gpu>);

NNVM_REGISTER_OP(clip)
.set_attr<FCompute>("FCompute<gpu>", Clip<gpu>)
.set_attr<FComputeEx>("FComputeEx<gpu>", ClipEx<gpu>);

NNVM_REGISTER_OP(_backward_clip)
.set_attr<FCompute>("FCompute<gpu>", ClipGrad_<gpu>);

NNVM_REGISTER_OP(repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpForward<gpu>);

NNVM_REGISTER_OP(_backward_repeat)
.set_attr<FCompute>("FCompute<gpu>", RepeatOpBackward<gpu>);

NNVM_REGISTER_OP(tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpForward<gpu>);

NNVM_REGISTER_OP(_backward_tile)
.set_attr<FCompute>("FCompute<gpu>", TileOpBackward<gpu>);

NNVM_REGISTER_OP(reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(_backward_reverse)
.set_attr<FCompute>("FCompute<gpu>", ReverseOpForward<gpu>);

NNVM_REGISTER_OP(stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpForward<gpu>);

NNVM_REGISTER_OP(_backward_stack)
.set_attr<FCompute>("FCompute<gpu>", StackOpBackward<gpu>);

NNVM_REGISTER_OP(squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

NNVM_REGISTER_OP(_backward_squeeze)
.set_attr<FCompute>("FCompute<gpu>", UnaryOp::IdentityCompute<gpu>);

}  // namespace op
}  // namespace mxnet

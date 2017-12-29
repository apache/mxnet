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
 * \file ndarray_function.cu
 * \brief GPU Implementation of ndarray function.
 */

// this will be invoked by nvcc and compile GPU version
#include <cub/cub.cuh>
#include <dmlc/logging.h>
#include "../operator/mxnet_op.h"
#include "../operator/tensor/init_op.h"
#include "../operator/tensor/util/tensor_util-inl.h"
#include "../operator/tensor/util/tensor_util-inl.cuh"
#include "../common/cuda_utils.h"
#include "./ndarray_function.h"
#include "./ndarray_function-inl.h"
#include "./ndarray_function-inl.cuh"

namespace mxnet {
namespace ndarray {
template<>
void Copy<cpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    mshadow::Copy(to->FlatTo1D<gpu, DType>(),
                  from.FlatTo1D<cpu, DType>(),
                  ctx.get_stream<gpu>());
  });
}

template<>
void Copy<gpu, cpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  CHECK_EQ(to->type_flag_, from.type_flag_)
    << "Source and target must have the same data type when copying across devices.";
  MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
    mshadow::Copy(to->FlatTo1D<cpu, DType>(),
                  from.FlatTo1D<gpu, DType>(),
                  ctx.get_stream<gpu>());
  });
}

template<>
void Copy<gpu, gpu>(const TBlob &from, TBlob *to,
                    Context from_ctx, Context to_ctx,
                    RunContext ctx) {
  if (from_ctx.dev_id == to_ctx.dev_id) {
    mshadow::Stream<gpu>* s = ctx.get_stream<gpu>();
    MSHADOW_TYPE_SWITCH(to->type_flag_, DType, {
      if (to->type_flag_ == from.type_flag_) {
        mshadow::Copy(to->FlatTo1D<gpu, DType>(s),
                      from.FlatTo1D<gpu, DType>(s),
                      s);
      } else {
        MSHADOW_TYPE_SWITCH(from.type_flag_, SrcDType, {
          to->FlatTo1D<gpu, DType>(s) =
            mshadow::expr::tcast<DType>(from.FlatTo1D<gpu, SrcDType>(s));
        })
      }
    })
  } else {
    CHECK(from.CheckContiguous() && to->CheckContiguous())
      << "copy across only support continugous memory";
    CHECK_EQ(to->type_flag_, from.type_flag_)
      << "Source and target must have the same data type when copying across devices.";
    mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
    CHECK(s != NULL) << "need stream in GPU context";
    cudaMemcpyPeerAsync(to->dptr_,
                        to_ctx.dev_id,
                        from.dptr_,
                        from_ctx.dev_id,
                        from.shape_.Size() * mshadow::mshadow_sizeof(to->type_flag_),
                        s->stream_);
  }
}

/*!
 * \brief GPU impl of elemwise sum for rowsparse tensors.
 */
void ElementwiseSumRspImpl(mshadow::Stream<gpu>* s,
                           const Resource& rsc,
                           const std::vector<NDArray>& nds,
                           NDArray* out) {
  using namespace mxnet::op;
  using namespace rowsparse;
  using nnvm::dim_t;
  CHECK_EQ(out->storage_type(), kRowSparseStorage)
    << "Expected rowsparse storage_type (" << out->storage_type() << " given)";
  int init = 0;
  for (const auto& nd : nds) {
    if (nd.storage_initialized()) {
      init++;
      break;
    }
  }
  if (init == 0) {
    FillZerosRspImpl(s, *out);
    return;
  }
  const dim_t num_rows = out->shape()[0];
  const dim_t row_length = out->shape().ProdShape(1, out->shape().ndim());
  MSHADOW_TYPE_SWITCH(out->dtype(), DType, {  // data type
    MSHADOW_IDX_TYPE_SWITCH(out->aux_type(kIdx), IType, {  // row_idx type
      // Allocate temporary storage for row_flg array and cub's prefix sum operation
      IType* row_flg = NULL;
      void* d_temp_storage = NULL;
      size_t temp_storage_bytes = 0;
      cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    row_flg,
                                    row_flg,
                                    num_rows,
                                    mshadow::Stream<gpu>::GetStream(s));
      mshadow::Tensor<gpu, 1, char> workspace = rsc
          .get_space_typed<gpu, 1, char>(mshadow::Shape1(num_rows * sizeof(IType) +
                                                         temp_storage_bytes), s);
      row_flg = reinterpret_cast<IType*>(workspace.dptr_);
      d_temp_storage = workspace.dptr_ + num_rows*sizeof(IType);
      // Mark row_flg array with 0 for zero rows and 1 for non-zero rows
      dim_t num_threads = num_rows;
      mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, num_threads, row_flg);
      for (const auto& nd : nds) {
        if (nd.storage_initialized()) {
          const IType* nd_row_idx = nd.aux_data(kIdx).dptr<IType>();
          const dim_t nd_nnr = nd.storage_shape()[0];
          num_threads = nd_nnr;
          mxnet_op::Kernel<MarkRspRowFlgKernel, gpu>::Launch(s, num_threads,
              row_flg, nd_row_idx, nd_nnr);
        }
      }
      // Compute inclusive prefix sum over row_flg
      cub::DeviceScan::InclusiveSum(d_temp_storage,
                                    temp_storage_bytes,
                                    row_flg,
                                    row_flg,
                                    num_rows,
                                    mshadow::Stream<gpu>::GetStream(s));
      // Get total number of output non-zero rows from GPU and allocate out data and row_idx
      dim_t nnr_out = 0;
      CUDA_CALL(cudaMemcpy(&nnr_out, &row_flg[num_rows-1], sizeof(dim_t),
                           cudaMemcpyDeviceToHost));
      out->CheckAndAlloc({mshadow::Shape1(nnr_out)});
      IType* out_row_idx = out->aux_data(kIdx).dptr<IType>();
      DType* out_data = out->data().dptr<DType>();
      // Fill row_idx array of output using row_flg
      num_threads = num_rows;
      mxnet_op::Kernel<FillRspRowIdxKernel, gpu>::Launch(s, num_threads,
          out_row_idx, row_flg, num_rows);
      // Perform elementwise addition, writing to output data
      num_threads = nnr_out * row_length;
      mxnet_op::Kernel<mxnet_op::set_zero, gpu>::Launch(s, num_threads, out_data);
      for (const auto& nd : nds) {
        if (nd.storage_initialized()) {
          const IType* nd_row_idx = nd.aux_data(kIdx).dptr<IType>();
          const DType* nd_data = nd.data().dptr<DType>();
          const dim_t nd_nnr = nd.storage_shape()[0];
          num_threads = nd_nnr * row_length;
          mxnet_op::Kernel<ElementWiseRspAdditionKernel, gpu>::Launch(s, num_threads,
              out_data, row_flg, nd_row_idx, nd_data, nd_nnr, row_length);
        }
      }
    });
  });
}

/*!
 * \brief Parallel gpu impl of elemwise sum for sparse tensors.
 * Currently only support row sparse sum.
 */
template<>
void ElementwiseSum<gpu>(mshadow::Stream<gpu>* s,
                         const Resource& rsc,
                         const std::vector<NDArray>& nds,
                         NDArray* out) {
  if (nds.empty()) return;
  if (nds[0].storage_type() == kRowSparseStorage) {
    ElementwiseSumRspImpl(s, rsc, nds, out);
  } else {
    LOG(FATAL) << "ElementwiseSum<gpu> has not been implemented for storage_type = << "
        << nds[0].storage_type();
  }
}

template<>
void Eval<gpu>(mshadow::Stream<gpu> *s,
               const real_t val, const NDArray& dst) {
  NDArray temp = dst;
  const NDArrayStorageType stype = temp.storage_type();
  if (stype == kRowSparseStorage) {
    SetValueRspImpl(s, val, &temp);
  } else {
    LOG(FATAL) << "Not implemented for storage type" << stype;
  }
}

}  // namespace ndarray
}  // namespace mxnet

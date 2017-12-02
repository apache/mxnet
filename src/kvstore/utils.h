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
 * \file utils.h
 * \brief Basic utilility functions.
 */
#ifndef MXNET_KVSTORE_UTILS_H_
#define MXNET_KVSTORE_UTILS_H_

#include <dmlc/logging.h>
#include "mxnet/ndarray.h"

namespace mxnet {
namespace kvstore {

/*!
 * \brief When src is a rsp with full rows,
 * simply copy retained rows directly from cpu to gpu
 * without invoking sparse_retain op.
 */
template<typename from_xpu, typename to_xpu>
void CopyRetainedRowsImpl(mshadow::Stream<to_xpu>* to_stream,
                          mshadow::Stream<gpu>* gpu_stream,
                          const NDArray& src,
                          const NDArray& indices,
                          NDArray* dst) {
  CHECK_EQ(src.storage_type(), kRowSparseStorage)
    << "CopyRetainedRowsToGPU expects row-sparse src NDArray";
  CHECK_EQ(src.storage_shape()[0], src.shape()[0])
    << "CopyRetainedRowsToGPU only supports src rsp with full rows";
  CHECK_EQ(indices.storage_type(), kDefaultStorage);
  CHECK_EQ(dst->storage_type(), kRowSparseStorage);
  CHECK_EQ(indices.dtype(), dst->aux_type(rowsparse::kIdx))
    << "CopyRetainedRowsToGPU only supports same data type for idx array and dst aux_data(0)";
  if (!src.storage_initialized() || indices.data().Size() == 0U) {
    op::FillZerosRspImpl(to_stream, *dst);
    return;
  }
  using namespace mshadow;

  const TBlob& src_data = src.data();
  const TBlob& idx_data = indices.data();
  const size_t row_length = src.shape().ProdShape(1, src.shape().ndim());
  const size_t num_rows_retained = idx_data.Size();
  dst->CheckAndAlloc({Shape1(num_rows_retained)});
  TBlob dst_data = dst->data();
  TBlob dst_idx_data = dst->aux_data(rowsparse::kIdx);
  MSHADOW_TYPE_SWITCH(src.dtype(), DType, {
    MSHADOW_IDX_TYPE_SWITCH(indices.dtype(), IType, {
      // copy idx array
      Tensor<to_xpu, 1, IType> dst_idx_tensor = dst_idx_data.FlatTo1D<to_xpu, IType>();
      const Tensor<cpu, 1, IType> idx_tensor = idx_data.FlatTo1D<cpu, IType>();
      Copy(dst_idx_tensor, idx_tensor, to_stream);
      // copy src data
      const Tensor<from_xpu, 2, DType> src_data_tensor =
          src_data.get_with_shape<from_xpu, 2, DType>(
              Shape2(src_data.shape_[0], row_length));
      Tensor<to_xpu, 2, DType> dst_data_tensor = dst_data.get_with_shape<to_xpu, 2, DType>(
          Shape2(dst_data.shape_[0], row_length));

      for (size_t i = 0; i < num_rows_retained; ++i) {
        Copy(dst_data_tensor[i], src_data_tensor[idx_tensor[i]], gpu_stream);
      }
    })
  })
}

void CopyRetainedRows(RunContext rctx,
                      const NDArray& src,
                      const NDArray& indices,
                      NDArray* dst) {
  const bool is_src_gpu = src.ctx().dev_mask() == Context::kGPU;
  const bool is_dst_gpu = dst->ctx().dev_mask() == Context::kGPU;
  CHECK(is_src_gpu || is_dst_gpu) << "Not implemented for case of cpu to cpu";
#if MXNET_USE_CUDA == 1
  if (is_src_gpu && is_dst_gpu) {
    CopyRetainedRowsImpl<gpu, gpu>(rctx.get_stream<gpu>(),
      rctx.get_stream<gpu>(), src, indices, dst);
  } else if (!is_src_gpu && is_dst_gpu) {
    CopyRetainedRowsImpl<cpu, gpu>(rctx.get_stream<gpu>(),
      rctx.get_stream<gpu>(), src, indices, dst);
  } else {
    CopyRetainedRowsImpl<gpu, cpu>(rctx.get_stream<cpu>(),
      rctx.get_stream<gpu>(), src, indices, dst);
  }
#else
  LOG(FATAL) << "GPU not enabled";
#endif
}

}  // namespace kvstore
}  // namespace mxnet
#endif  // MXNET_KVSTORE_UTILS_H_

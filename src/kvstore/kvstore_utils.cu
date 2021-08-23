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
 *  Copyright (c) 2017 by Contributors
 * \file kvstore_utils.cu
 * \brief gpu implementation of util functions
 */
#if defined(_MSC_VER) && __CUDACC_VER_MAJOR__ == 8 && __CUDACC_VER_BUILD__ != 44
// Many CUDA 8 compilers other than V8.0.44 crash on Windows
#pragma warning("Potential crash on CUDA compiler detected. Switching sorting from CUB to Thrust")
#define SORT_WITH_THRUST
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#else
#undef SORT_WITH_THRUST
#endif
#include "./kvstore_utils.h"
#include <cub/cub.cuh>
#include <mxnet/resource.h>
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {

template<typename IType>
size_t UniqueImplGPU(NDArray *workspace, mshadow::Stream<gpu> *s,
                   IType *dptr, const size_t size, Context ctx) {
  // estimate unique temp space. The first byte is reserved to store the number
  // of unique values selected
  const size_t num_selected_bytes = sizeof(size_t);
  size_t unique_temp_bytes = 0;
  size_t *null_ptr = nullptr;
  size_t *null_dptr = nullptr;
  cudaStream_t stream = mshadow::Stream<gpu>::GetStream(s);
  cub::DeviceSelect::Unique(nullptr, unique_temp_bytes, null_dptr, null_dptr,
                            null_ptr, size, stream);
  // estimate sort temp space
  const size_t sort_output_bytes = size * sizeof(IType);
  size_t sort_temp_bytes = 0;
#ifndef SORT_WITH_THRUST
  // The least-significant bit index (inclusive) needed for key comparison
  const int begin_bit = 0;
  // The most-significant bit index (exclusive) needed for key comparison
  const int end_bit = sizeof(IType) * 8;
  cub::DeviceRadixSort::SortKeys(nullptr, sort_temp_bytes, null_dptr, null_dptr,
                                 size, begin_bit, end_bit, stream);
#else
  // sort_temp_bytes remains 0 because thrust request memory by itself
#endif
  // request temp storage
  const size_t total_workspace = num_selected_bytes + sort_output_bytes +
                                 std::max(sort_temp_bytes, unique_temp_bytes);
  *workspace = NDArray(mshadow::Shape1((total_workspace + 3) / 4), ctx, false);
  char* workspace_dptr = reinterpret_cast<char*>(workspace->data().dptr_);
  // temp space layout: num_selected_ptr, sort_output_bytes, unique/sort_temp_storage
  size_t* num_selected_ptr = reinterpret_cast<size_t*>(workspace_dptr);
  IType* sort_output_ptr = reinterpret_cast<IType*>(workspace_dptr + num_selected_bytes);
  void *temp_storage = static_cast<void*>(workspace_dptr +
                                          num_selected_bytes + sort_output_bytes);
  // execute the sort kernel
#ifndef SORT_WITH_THRUST
  cub::DeviceRadixSort::SortKeys(temp_storage, sort_temp_bytes, dptr, sort_output_ptr,
                                 size, begin_bit, end_bit, stream);
#else
  thrust::sort(thrust::cuda::par.on(stream),
               dptr, dptr + size, thrust::greater<IType>());
  CUDA_CALL(cudaMemcpyAsync(sort_output_ptr, dptr, sort_output_bytes,
                            cudaMemcpyDeviceToDevice, stream));
#endif
  // execute unique kernel
  cub::DeviceSelect::Unique(temp_storage, unique_temp_bytes, sort_output_ptr, dptr,
                            num_selected_ptr, size, stream);
  // retrieve num selected unique values
  size_t num_selected_out = 0;
  CUDA_CALL(cudaMemcpyAsync(&num_selected_out, num_selected_ptr, num_selected_bytes,
                            cudaMemcpyDeviceToHost, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));
  return num_selected_out;
}

template<>
void UniqueImpl<gpu>(NDArray *workspace, mshadow::Stream<gpu> *s, const NDArray &out) {
  const size_t num_elements = out.shape().Size();
  CHECK_EQ(out.storage_type(), kRowSparseStorage) << "row_sparse NDArray is expected";
  MSHADOW_IDX_TYPE_SWITCH(out.dtype(), IType, {
    IType *dptr = out.data().dptr<IType>();
    size_t num_selected_out = UniqueImplGPU(workspace, s, dptr, num_elements, out.ctx());
    // set the shape of data/aux_data according to the number of unique values
    out.set_aux_shape(rowsparse::kIdx, mshadow::Shape1(num_selected_out));
  });
}

}  // namespace kvstore
}  // namespace mxnet

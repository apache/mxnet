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
size_t UniqueImplGPU(const Resource& rsc, mshadow::Stream<gpu> *s,
                   IType *dptr, const size_t size) {
#ifndef SORT_WITH_THRUST
  size_t sort_temp_bytes = 0;
  cub::DeviceRadixSort::SortKeys(NULL, sort_temp_bytes,
    dptr, dptr, size, 0, sizeof(IType)*8, mshadow::Stream<gpu>::GetStream(s));
  mshadow::Tensor<gpu, 1, char> sort_space = rsc
    .get_space_typed<gpu, 1, char>(
      mshadow::Shape1(sort_temp_bytes), s);
  void *sort_temp_storage = static_cast<void*>(sort_space.dptr_);
  cub::DeviceRadixSort::SortKeys(sort_temp_storage, sort_temp_bytes,
    dptr, dptr, size, 0, sizeof(IType)*8, mshadow::Stream<gpu>::GetStream(s));
#else
  thrust::sort(thrust::cuda::par.on(mshadow::Stream<gpu>::GetStream(s)),
    dptr, dptr + size, thrust::greater<IType>());
#endif
  // estimate unique temp space. The first byte is reserved to store the number of
  // unique values selected
  size_t unique_temp_bytes = 0;
  size_t num_selected_bytes = sizeof(size_t);
  size_t *null_ptr = nullptr;
  cub::DeviceSelect::Unique(NULL, unique_temp_bytes, dptr, dptr,
    null_ptr, size, mshadow::Stream<gpu>::GetStream(s));
  size_t total_temp_bytes = unique_temp_bytes + num_selected_bytes;
  // request temp storage
  mshadow::Tensor<gpu, 1, char> workspace = rsc
    .get_space_typed<gpu, 1, char>(mshadow::Shape1(total_temp_bytes), s);
  void *unique_temp_storage = static_cast<void*>(workspace.dptr_ + num_selected_bytes);
  size_t* num_selected_ptr = reinterpret_cast<size_t*>(workspace.dptr_);
  // execute unique kernel
  cub::DeviceSelect::Unique(unique_temp_storage, unique_temp_bytes, dptr, dptr,
    num_selected_ptr, size, mshadow::Stream<gpu>::GetStream(s));
  // retrieve num selected unique values
  size_t num_selected_out = 0;
  CUDA_CALL(cudaMemcpy(&num_selected_out, num_selected_ptr, num_selected_bytes,
     cudaMemcpyDeviceToHost));
  return num_selected_out;
}

template<>
void UniqueImpl<gpu>(const Resource& rsc, mshadow::Stream<gpu> *s,
                     const NDArray &out) {
  const size_t num_elements = out.shape().Size();
  CHECK_EQ(out.storage_type(), kRowSparseStorage) << "row_sparse NDArray is expected";
  MSHADOW_IDX_TYPE_SWITCH(out.dtype(), IType, {
    IType *dptr = out.data().dptr<IType>();
    size_t num_selected_out = UniqueImplGPU(rsc, s, dptr, num_elements);
    // set the shape of data/aux_data according to the number of unique values
    out.set_aux_shape(rowsparse::kIdx, mshadow::Shape1(num_selected_out));
  });
}

}  // namespace kvstore
}  // namespace mxnet

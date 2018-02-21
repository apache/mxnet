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
void UniqueImplGPU(const Resource& rsc, mshadow::Stream<gpu> *s,
                   IType *dptr, const size_t size, IType *num_results) {
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
  // estimate unique temp space
  size_t unique_temp_bytes = 0;
  size_t *null_ptr = nullptr;
  cub::DeviceSelect::Unique(NULL, unique_temp_bytes, dptr, dptr,
    null_ptr, size, mshadow::Stream<gpu>::GetStream(s));
  // execute unique kernel
  mshadow::Tensor<gpu, 1, char> unique_space = rsc
    .get_space_typed<gpu, 1, char>(mshadow::Shape1(unique_temp_bytes), s);
  void *unique_temp_storage = static_cast<void*>(unique_space.dptr_);
  cub::DeviceSelect::Unique(unique_temp_storage, unique_temp_bytes, dptr, dptr,
    num_results, size, mshadow::Stream<gpu>::GetStream(s));
}

template<>
void UniqueImpl<gpu>(const Resource& rsc, mshadow::Stream<gpu> *s,
                     const NDArray& sized_array) {
  const size_t num_elements = sized_array.shape().Size() - 1;
  MSHADOW_IDX_TYPE_SWITCH(sized_array.data().type_flag_, IType, {
    IType *size_ptr = sized_array.data().dptr<IType>();
    IType *data_ptr = size_ptr + 1;
    UniqueImplGPU(rsc, s, data_ptr, num_elements, size_ptr);
  });
}


}  // namespace kvstore
}  // namespace mxnet

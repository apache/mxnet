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
 * \file utils.cu
 * \brief gpu implementation of util functions
 */
#include "./utils.h"
#include "../common/utils.h"
#include <cub/cub.cuh>
#include <mxnet/resource.h>

namespace mxnet {
namespace kvstore {

/*!
 * \brief sort and get unique values.
 */
template<>
void UniqueImpl<gpu>(const Resource& rsc, mshadow::Stream<gpu> *s,
                     NDArray *out, nnvm::dim_t size) {
  MSHADOW_IDX_TYPE_SWITCH(out->data().type_flag_, IType, {
    IType *dptr = out->data().dptr<IType>();
    size_t sort_temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(NULL, sort_temp_bytes,
      dptr, dptr, size, 0, sizeof(IType)*8, mshadow::Stream<gpu>::GetStream(s));
    size_t unique_temp_bytes = 0;
    mshadow::Tensor<gpu, 1, char> dummy_space = rsc
      .get_space_typed<gpu, 1, char>(
        mshadow::Shape1(sizeof(size_t)), s);
    size_t *dummy_ptr = reinterpret_cast<size_t*>(dummy_space.dptr_);
    cub::DeviceSelect::Unique(NULL, unique_temp_bytes, dptr, dptr,
      dummy_ptr, size, mshadow::Stream<gpu>::GetStream(s));

    size_t cub_temp_bytes = std::max(sort_temp_bytes, unique_temp_bytes);
    mshadow::Tensor<gpu, 1, char> workspace = rsc
      .get_space_typed<gpu, 1, char>(
        mshadow::Shape1((cub_temp_bytes + sizeof(size_t) + 7) / 8 * 8), s);

    void *sort_temp_storage = static_cast<void*>(workspace.dptr_);
    void *unique_temp_storage = static_cast<void*>(
      workspace.dptr_);
    size_t *d_num_selected_out = reinterpret_cast<size_t*>(
      workspace.dptr_ + (cub_temp_bytes + 7) / 8 * 8);

    cub::DeviceRadixSort::SortKeys(sort_temp_storage, sort_temp_bytes,
      dptr, dptr, size, 0, sizeof(IType)*8, mshadow::Stream<gpu>::GetStream(s));

    cub::DeviceSelect::Unique(unique_temp_storage, unique_temp_bytes, dptr, dptr,
      d_num_selected_out, size, mshadow::Stream<gpu>::GetStream(s));
    s->Wait();

    size_t num_selected_out = 0;
    CUDA_CALL(cudaMemcpy(&num_selected_out, d_num_selected_out, sizeof(size_t),
       cudaMemcpyDeviceToHost));
    *out = out->Reshape(mshadow::Shape1(num_selected_out));
  });
}


}  // namespace kvstore
}  // namespace mxnet

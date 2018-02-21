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
 * \file kvstore_utils.cc
 * \brief cpu implementation of util functions
 */

#include "./kvstore_utils.h"
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {


template<>
void UniqueImpl<cpu>(const Resource& rsc, mshadow::Stream<cpu> *s,
                     const NDArray& sized_array) {
  const size_t num_elements = sized_array.shape().Size() - 1;
  MSHADOW_IDX_TYPE_SWITCH(sized_array.data().type_flag_, IType, {
    // the first number is size, followed by elements to sort
    IType *size_ptr = sized_array.data().dptr<IType>();
    IType *data_ptr = size_ptr + 1;
    common::ParallelSort(data_ptr, data_ptr + num_elements,
                         engine::OpenMP::Get()->GetRecommendedOMPThreadCount());
    const IType num_unique_idx = std::unique(data_ptr, data_ptr + num_elements) - data_ptr;
    *size_ptr = num_unique_idx;
  });
}


}  // namespace kvstore
}  // namespace mxnet

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
void UniqueImpl<cpu>(NDArray* workspace, mshadow::Stream<cpu> *s,
                     const NDArray& out) {
  const size_t num_elements = out.shape().Size();
  CHECK_EQ(out.storage_type(), kRowSparseStorage) << "row_sparse NDArray is expected";
  MSHADOW_IDX_TYPE_SWITCH(out.dtype(), IType, {
    IType *dptr = out.data().dptr<IType>();
    common::ParallelSort(dptr, dptr + num_elements,
                         engine::OpenMP::Get()->GetRecommendedOMPThreadCount());
    const size_t num_selected_out = std::unique(dptr, dptr + num_elements) - dptr;
    // set the shape of data/aux_data according to the number of unique values
    out.set_aux_shape(rowsparse::kIdx, mshadow::Shape1(num_selected_out));
  });
}


}  // namespace kvstore
}  // namespace mxnet

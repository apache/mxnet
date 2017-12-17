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
 * \file utils.cc
 * \brief cpu implementation of util functions
 */

#include "./utils.h"
#include "../common/utils.h"

namespace mxnet {
namespace kvstore {


template<>
void UniqueImpl<cpu>(const Resource& rsc, mshadow::Stream<cpu> *s,
                     NDArray *out, nnvm::dim_t size) {
  MSHADOW_IDX_TYPE_SWITCH(out->data().type_flag_, IType, {
    IType *dptr = out->data().dptr<IType>();
    common::ParallelSort(dptr, dptr + size, omp_get_max_threads());
    size_t num_unique_idx = std::unique(dptr, dptr + size) - dptr;
    *out = out->Reshape(mshadow::Shape1(num_unique_idx));
  });
}

bool CheckSameRowid(
    const std::vector<std::pair<NDArray*, NDArray>>& val_rowids) {
  bool is_same_rowid = true;
  MSHADOW_TYPE_SWITCH(val_rowids[0].second.dtype(), IType, {
    const TBlob& rowid_first = val_rowids[0].second.data();
    const IType *first_dptr = rowid_first.dptr<IType>();
    const index_t first_size = rowid_first.Size();
    for (size_t i = 1; i < val_rowids.size(); ++i) {
      const TBlob& rowid_i = val_rowids[i].second.data();
      if (rowid_i.dptr<IType>() != first_dptr
          || rowid_i.Size() != first_size) {
        is_same_rowid = false;
      }
    }
  });
  return is_same_rowid;
}

}  // namespace kvstore
}  // namespace mxnet

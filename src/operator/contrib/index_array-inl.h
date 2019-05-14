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

#ifndef MXNET_OPERATOR_CONTRIB_INDEX_ARRAY_INL_H_
#define MXNET_OPERATOR_CONTRIB_INDEX_ARRAY_INL_H_

#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../tensor/init_op.h"

namespace mxnet {
namespace op {

namespace index_array_enum {
enum IndexArrayOpInputs {kIn};
enum IndexArrayOpOutputs {kOut};
enum IndexArrayOpResource {kTempSpace};
}  // namespace index_array_enum

template<int req>
struct IndexArrayKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int64_t* out_data,
                                  const int n,
                                  const int64_t* workspace) {
    for (ptrdiff_t j = 0; j < n; j++) {
      int64_t upper = workspace[ptrdiff_t(2) * j];
      int64_t lower = workspace[ptrdiff_t(2) * j + ptrdiff_t(1)];
      KERNEL_ASSIGN(out_data[ptrdiff_t(i) * ptrdiff_t(n) + j], req, (i % upper) / lower);
    }
  }
};

template<int req>
struct IndexArrayDefaultKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int64_t* out_data,
                                  const int ndim,
                                  const dim_t* shape) {
    int64_t index = i;
    for (ptrdiff_t j = ndim - 1; j >= 0; j--) {
      KERNEL_ASSIGN(out_data[ptrdiff_t(i) * ptrdiff_t(ndim) + j], req, index % shape[j]);
      index /= shape[j];
    }
  }
};

inline std::vector<int64_t> IndexArrayComputeIndexProducts(const TShape &inshape) {
  const int ndim = inshape.ndim();

  std::vector<int64_t> index_products(static_cast<size_t>(ndim + 1));

  index_products[ndim] = 1;

  for (int i = ndim - 1; i >= 0; i--) {
    index_products[i] = index_products[i + 1] * inshape[i];
  }

  return index_products;
}

inline void IndexArrayBuildSelectedAxesWorkspace(const mxnet::Tuple<int> &axes,
                                                 const std::vector<int64_t> &index_products,
                                                 int64_t* workspace,
                                                 const int ndim) {
  for (int i = 0; i < axes.ndim(); i++) {
    // Make sure that the axis is between 0 and ndim.
    const int axis = ((axes[i] % ndim) + ndim) % ndim;

    workspace[ptrdiff_t(2) * ptrdiff_t(i)] = index_products[axis];
    workspace[ptrdiff_t(2) * ptrdiff_t(i) + ptrdiff_t(1)] = index_products[axis + 1];
  }
}

struct IndexArrayParam : public dmlc::Parameter<IndexArrayParam> {
  dmlc::optional<mxnet::Tuple<int>> axes;
  DMLC_DECLARE_PARAMETER(IndexArrayParam) {
    DMLC_DECLARE_FIELD(axes).set_default(dmlc::optional<mxnet::Tuple<int>>())
      .describe("The axes to include in the index array. Supports negative values.");
  }
};  // struct IndexArrayParam

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_ARRAY_INL_H_

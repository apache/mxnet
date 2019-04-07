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
  MSHADOW_XINLINE static void Map(size_t i,
                                  int64_t* out_data,
                                  const uint32_t n,
                                  const int64_t* workspace) {
    for (uint32_t j = 0; j < n; j++) {
      int64_t upper = workspace[2 * j];
      int64_t lower = workspace[2 * j + 1];
      KERNEL_ASSIGN(out_data[i * n + j], req, (i % upper) / lower);
    }
  }
};

template<int req>
struct IndexArrayDefaultKernel {
  MSHADOW_XINLINE static void Map(size_t i,
                                  int64_t* out_data,
                                  const uint32_t ndim,
                                  const dim_t* shape) {
    int64_t index = i;
    for (uint32_t j = ndim; j-- > 0;) {
      KERNEL_ASSIGN(out_data[i * ndim + j], req, index % shape[j]);
      index /= shape[j];
    }
  }
};

inline std::vector<int64_t> IndexArrayComputeIndexProducts(const TShape &inshape) {
  const uint32_t ndim = inshape.ndim();

  std::vector<int64_t> index_products(ndim + 1);

  index_products[ndim] = 1;

  for (uint32_t i = ndim; i-- > 0;) {
    index_products[i] = index_products[i + 1] * inshape[i];
  }

  return index_products;
}

inline void IndexArrayBuildSelectedAxesWorkspace(const TShape &axes,
                                                 const std::vector<int64_t> &index_products,
                                                 int64_t* workspace,
                                                 const uint32_t ndim) {
  for (uint32_t i = 0; i < axes.ndim(); i++) {
    // Make sure that the axis is between 0 and ndim.
    const dim_t axis = ((axes[i] % ndim) + ndim) % ndim;

    workspace[2 * i] = index_products[axis];
    workspace[2 * i + 1] = index_products[axis + 1];
  }
}

template<typename xpu>
void IndexArrayForward(const nnvm::NodeAttrs &attrs,
                       const OpContext &ctx,
                       const std::vector<TBlob> &inputs,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &outputs);

struct IndexArrayParam : public dmlc::Parameter<IndexArrayParam> {
  dmlc::optional<mxnet::TShape> axes;
  DMLC_DECLARE_PARAMETER(IndexArrayParam) {
    DMLC_DECLARE_FIELD(axes).set_default(dmlc::optional<mxnet::TShape>())
      .describe("The axes to include in the index array. Supports negative values.");
  }
};  // struct IndexArrayParam

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INDEX_ARRAY_INL_H_

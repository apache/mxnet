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
 * Copyright (c) 2019 by Contributors
 * \file np_unique_op.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_

#include <mxnet/operator_util.h>
#include <dmlc/optional.h>
#include <vector>
#include <numeric>
#include <set>
#include <string>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../contrib/boolean_mask-inl.h"
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#endif

namespace mxnet {
namespace op {

struct UniqueReturnInverseKernel {
  MSHADOW_XINLINE static void Map(dim_t i,
                                  dim_t* unique_inverse,
                                  const int32_t* prefix_sum,
                                  const dim_t* perm) {
      dim_t j = perm[i];
      unique_inverse[j] = prefix_sum[i] - 1;
  }
};

struct UniqueReturnCountsKernel {
  MSHADOW_XINLINE static void Map(dim_t i, dim_t* unique_counts, const dim_t* idx) {
      unique_counts[i] = idx[i + 1] - idx[i];
  }
};

struct NumpyUniqueParam : public dmlc::Parameter<NumpyUniqueParam> {
  bool return_index, return_inverse, return_counts;
  dmlc::optional<int> axis;
  DMLC_DECLARE_PARAMETER(NumpyUniqueParam) {
    DMLC_DECLARE_FIELD(return_index)
    .set_default(false)
    .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_inverse)
    .set_default(false)
    .describe("If true, return the indices of the input.");
    DMLC_DECLARE_FIELD(return_counts)
    .set_default(false)
    .describe("If true, return the number of times each unique item appears in input.");
    DMLC_DECLARE_FIELD(axis)
    .set_default(dmlc::optional<int>())
    .describe("An integer that represents the axis to operator on.");
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_

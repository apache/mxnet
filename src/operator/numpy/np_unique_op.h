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

struct UniqueComputeAuxCPUKernel {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // i is the index of out_data
  template<typename DType>
  MSHADOW_XINLINE static void Map(int64_t i, DType* out_data, const DType* in_data,
                                  const int64_t* idx, const int64_t M) {
    int64_t j = idx[i];
    std::memcpy(out_data + i * M, in_data + j * M, M * sizeof(DType));
  }
};

struct UniqueComputeAuxGPUKernel {
  // assume that idx have been flattened to a 1-D tensor (N,)
  // assume that out_data and in_data have been flattened to 2-D tensors, (N, M) and (K, M)
  // M is the number of columns of in_data and out_data
  // i is the index of out_data
  template<typename DType>
  MSHADOW_XINLINE static void Map(int64_t i, DType* out_data, const DType* in_data,
                                  const int64_t* idx, const int64_t M) {
    int64_t j = idx[i/M];
    out_data[i] = in_data[j * M + i % M];
  }
};

struct UniqueComputeMaskCPUKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int64_t i,
                                  int64_t* out_data,
                                  const DType* in_data,
                                  const int64_t numel) {
    if (i == 0) {
      out_data[i] = 1;
    } else {
      out_data[i] = (std::memcmp(in_data + i * numel,
                     in_data + (i - 1) * numel, numel * sizeof(DType)) == 0) ? 0 : 1;
    }
  }
};

struct UniqueComputeMaskGPUKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int64_t i,
                                  int64_t* out_data,
                                  const DType* in_data,
                                  const int64_t numel) {
    if (i == 0) {
      out_data[i] = 1;
    } else {
      out_data[i] = 0;
      for (int64_t j = 0; j < numel; ++j) {
        if (in_data[(i - 1) * numel + j] != in_data[i * numel + j]) {
          out_data[i] = 1;
          break;
        }
      }
    }
  }
};

struct UniqueReturnInverseKernel {
  MSHADOW_XINLINE static void Map(int i,
                                  int64_t* unique_inverse,
                                  const int32_t* prefix_sum,
                                  const int64_t* perm) {
      dim_t j = perm[i];
      unique_inverse[j] = prefix_sum[i] - 1;
  }
};

struct UniqueReturnCountsKernel {
  MSHADOW_XINLINE static void Map(int i, int64_t* unique_counts, const int32_t* idx) {
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

inline bool NumpyUniqueType(const nnvm::NodeAttrs& attrs,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  for (size_t i = 1; i < out_attrs->size(); ++i) {
    TYPE_ASSIGN_CHECK(*out_attrs, i, mshadow::kInt64);
  }
  return out_attrs->at(0) != -1;
}

inline bool NumpyUniqueStorageType(const nnvm::NodeAttrs& attrs,
                            const int dev_mask,
                            DispatchMode* dispatch_mode,
                            std::vector<int> *in_attrs,
                            std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  // CHECK_EQ(out_attrs->size(), 1U);
  for (int &attr : *in_attrs) {
    CHECK_EQ(attr, kDefaultStorage) << "Only default storage is supported";
  }
  for (int &attr : *out_attrs) {
    attr = kDefaultStorage;
  }
  *dispatch_mode = DispatchMode::kFComputeEx;
  return true;
}

template<typename xpu>
void NumpyUniqueForward(const nnvm::NodeAttrs& attrs,
                        const OpContext &ctx,
                        const std::vector<NDArray> &inputs,
                        const std::vector<OpReqType> &req,
                        const std::vector<NDArray> &outputs);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_UNIQUE_OP_H_

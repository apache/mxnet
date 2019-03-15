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
 * Copyright (c) 2018 by Contributors
 * \file boolean_mask-inl.h
*/

#ifndef MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_
#define MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/ndarray.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include "../operator_common.h"
#include "../mxnet_op.h"
#include "../tensor/init_op.h"
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct BooleanMaskParam : public dmlc::Parameter<BooleanMaskParam> {
  int axis;
  DMLC_DECLARE_PARAMETER(BooleanMaskParam) {
    DMLC_DECLARE_FIELD(axis).set_default(0)
    .describe("An integer that represents the axis in NDArray to mask from.");
  }
};

struct BooleanMaskForwardKernel {
  template<typename DType>
  static void MSHADOW_XINLINE Map(int i,
                                  DType* out,
                                  const DType* data,
                                  const int32_t* idx,
                                  const size_t col_size) {
    int row_id = i / col_size;
    int col_id = i % col_size;
    int32_t prev = (row_id == 0) ? 0 : idx[row_id - 1];
    int32_t curr = idx[row_id];
    if (prev != curr) {
      out[prev * col_size + col_id] = data[i];
    }
  }
};

struct BooleanMaskBackwardKernel {
  template<typename DType>
  static void MSHADOW_XINLINE Map(int i,
                                  DType* igrad,
                                  const DType* ograd,
                                  const int32_t* idx,
                                  const size_t col_size) {
    int row_id = i / col_size;
    int col_id = i % col_size;
    int32_t prev = (row_id == 0) ? 0 : idx[row_id - 1];
    int32_t curr = idx[row_id];
    if (prev != curr) {
      igrad[i] = ograd[prev * col_size + col_id];
    }
  }
};

template<typename xpu>
inline void BooleanMaskForward(const nnvm::NodeAttrs& attrs,
                               const OpContext &ctx,
                               const std::vector<NDArray> &inputs,
                               const std::vector<OpReqType> &req,
                               const std::vector<NDArray> &outputs);

template<typename xpu>
inline void BooleanMaskBackward(const nnvm::NodeAttrs& attrs,
                                const OpContext &ctx,
                                const std::vector<NDArray> &inputs,
                                const std::vector<OpReqType> &req,
                                const std::vector<NDArray> &outputs);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_BOOLEAN_MASK_INL_H_

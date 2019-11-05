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
 * \file fold-inl.h
 * \brief CPU implementation of unfold operator
 * \author Istvan Fehervari
*/
#ifndef MXNET_OPERATOR_NN_FOLD_INL_H_
#define MXNET_OPERATOR_NN_FOLD_INL_H_
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include "../operator_common.h"
#include "./broadcast_reduce_op.h"

namespace mxnet {
namespace op {

struct UnfoldParam : public dmlc::Parameter<UnfoldParam> {
  int kernel_size;
  int dim;
  int stride;
  DMLC_DECLARE_PARAMETER(UnfoldParam) {
    DMLC_DECLARE_FIELD(kernel_size)
      .set_lower_bound(1)
      .describe("Size of each unfolded block");
    DMLC_DECLARE_FIELD(dim)
      .set_default(-1)
      .describe("Dimension to unfold");
    DMLC_DECLARE_FIELD(stride)
      .set_lower_bound(1)
      .set_default(1)
      .describe("Stride of the sliding window");
  }
};  // struct UnfoldParam

inline mxnet::TShape UnfoldShapeImpl(const mxnet::TShape& ishape, const int k,
                            const int32_t dim, const int32_t stride) {
  int32_t axis = CheckAxis(dim, ishape.ndim());

  auto num_target_dim = ishape[axis];

  CHECK_LE(k, num_target_dim) << "kernel size " << k << "must be less than or equal"
                             " to the number of elements in the target dimension";

  int num_windows = num_target_dim / k;

  auto o_ndim = ishape.ndim() + 1;
  mxnet::TShape oshape(o_ndim, -1);

  for (auto i = 0; i < ishape.ndim(); i++) {
    oshape[i] = ishape[i];
  }

  oshape[o_ndim - 1] = k;
  oshape[axis] = num_windows;
  return oshape;
}

inline bool UnfoldOpShape(const nnvm::NodeAttrs& attrs,
                          mxnet::ShapeVector* in_attrs,
                          mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const mxnet::TShape& ishape = (*in_attrs)[0];
  if (!mxnet::ndim_is_known(ishape)) {
    return false;
  }

  const UnfoldParam& param = nnvm::get<UnfoldParam>(attrs.parsed);

  mxnet::TShape oshape = UnfoldShapeImpl(ishape,
                                  param.kernel_size,
                                  param.dim,
                                  param.stride);
  if (shape_is_none(oshape)) {
    LOG(FATAL) << "Failed to infer shape for unfold.";
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, oshape);

  return shape_is_known(out_attrs->at(0));
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_FOLD_INL_H_

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
 * \file prepare_b_op-inl.h
 * \brief Quantize B to int8 and permute to a CPU-dependent format in preparation for multiplication.
 * This 
 */
#ifndef MXNET_OPERATOR_CONTRIB_INTGEMM_PREPARE_B_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_INTGEMM_PREPARE_B_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../../mshadow_op.h"
#include "../../mxnet_op.h"
#include "../../operator_common.h"
#include "../../elemwise_op_common.h"
#include "../../tensor/init_op.h"

namespace mxnet {
namespace op {

struct PrepareBParam : public dmlc::Parameter<PrepareBParam> {
  float multiplier;
  DMLC_DECLARE_PARAMETER(PrepareBParam) {
    DMLC_DECLARE_FIELD(multiplier)
      .describe("Multiply floats by this constant before casting to int8.  Typically you would set this to 127.0 / max absolute value in B.");
  }
};

inline bool PrepareBOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  // One in, one out.
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));

  const mxnet::TShape &shape = in_attrs->front();
  if (mxnet::ndim_is_known(shape)) {
    CHECK_GE(shape.ndim(), 2) << "Matrices have at least two dimensions.";
  }
  return !mxnet::op::shape_is_none(out_attrs->at(0));
}

inline bool PrepareBOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  // This routine converts from float to int8.
  TYPE_ASSIGN_CHECK(*out_attrs, 0, mshadow::kInt8);
  TYPE_ASSIGN_CHECK(*in_attrs, 0, mshadow::kFloat32);
  return true;
}

inline bool PrepareBOpStorageType(const nnvm::NodeAttrs& attrs,
                                   const int dev_mask,
                                   DispatchMode* dispatch_mode,
                                   std::vector<int>* in_attrs,
                                   std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  // Dense storage only.
  return storage_type_assign(&out_attrs->front(), kDefaultStorage, dispatch_mode, DispatchMode::kFCompute) &&
    storage_type_assign(&in_attrs->front(), kDefaultStorage, dispatch_mode, DispatchMode::kFCompute);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_INTGEMM_PREPARE_B_INL_H_

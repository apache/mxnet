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
 *  Copyright (c) 2018 by Contributors
 * \file optimizer_op.cc
 * \brief Optimizer operators
 * \author Leonard Lausen
 */
#include "../elemwise_op_common.h"
#include "./optimizer_op-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(GroupAdagradParam);

/*!
 * \brief Shape inference function for Group AdaGrad.
 */
inline bool GroupAdagradShape(const nnvm::NodeAttrs &attrs,
                              mxnet::ShapeVector *in_attrs,
                              mxnet::ShapeVector *out_attrs) {
  CHECK_EQ(in_attrs->size(), 3U);
  CHECK_EQ(out_attrs->size(), 1U);

  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(1));
  SHAPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  SHAPE_ASSIGN_CHECK(*in_attrs, 1, out_attrs->at(0));

  return !mxnet::op::shape_is_none(out_attrs->at(0)) &&
         (in_attrs->at(0)[0] == in_attrs->at(1)[0]) &&
         (in_attrs->at(0)[0] == in_attrs->at(2)[0]);
}

NNVM_REGISTER_OP(_contrib_group_adagrad_update)
.describe(R"code(Update function for Group AdaGrad optimizer.

Referenced from *Adaptive Subgradient Methods for Online Learning and Stochastic Optimization*,
and available at http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf but
uses only a single learning rate for every row of the parameter array.

Updates are applied by::

    grad = clip(grad * rescale_grad, clip_gradient)
    history += mean(square(grad), axis=1, keepdims=True)
    div = grad / sqrt(history + float_stable_eps)
    weight -= div * lr

Weights are updated lazily if the gradient is sparse.

Note that non-zero values for the weight decay option are not supported.

)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr_parser(ParamParser<GroupAdagradParam>)
.set_attr<mxnet::FInferShape>("FInferShape", GroupAdagradShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<3, 1>)
.set_attr<FInferStorageType>("FInferStorageType", GroupAdagradStorageType)
.set_attr<nnvm::FMutateInputs>("FMutateInputs",
  [](const nnvm::NodeAttrs& attrs) {
    return std::vector<uint32_t>{2};
  })
.set_attr<FComputeEx>("FComputeEx<cpu>", GroupAdagradUpdateEx<cpu>)
.add_argument("weight", "NDArray-or-Symbol", "Weight")
.add_argument("grad", "NDArray-or-Symbol", "Gradient")
.add_argument("history", "NDArray-or-Symbol", "History")
.add_arguments(GroupAdagradParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

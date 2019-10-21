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
 *  Copyright (c) 2019 by Contributors
 * \file np_broadcast_reduce_op_index.cc
 * \brief CPU Implementation of broadcast and reduce functions based on index.
 */
#include "./np_broadcast_reduce_op.h"

namespace mxnet {
namespace op {

bool NumpyReduceAxisShape(const nnvm::NodeAttrs& attrs,
                          std::vector<TShape> *in_attrs,
                          std::vector<TShape> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  if (!shape_is_known(in_attrs->at(0))) {
    return false;
  }
  const ReduceAxisParam& param = nnvm::get<ReduceAxisParam>(attrs.parsed);
  dmlc::optional<mxnet::Tuple<int>> axes;
  if (param.axis.has_value()) {
    mxnet::Tuple<int> t({param.axis.value()});
    axes = dmlc::optional<mxnet::Tuple<int>>(t);
  }
  SHAPE_ASSIGN_CHECK(*out_attrs, 0,
                     NumpyReduceAxesShapeImpl((*in_attrs)[0], axes, param.keepdims));
  return shape_is_known(out_attrs->at(0));
}

NNVM_REGISTER_OP(_npi_argmax)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<ReduceAxisParam>)
.set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxisShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.add_argument("data", "NDArray-or-Symbol", "The input")
.set_attr<FCompute>("FCompute<cpu>", SearchAxisCompute<cpu, mshadow::red::maximum>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_arguments(ReduceAxisParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

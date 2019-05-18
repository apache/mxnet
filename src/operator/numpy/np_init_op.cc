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
 * \file np_init_op.cc
 * \brief CPU Implementation of numpy init op
 */
#include "../tensor/init_op.h"
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_zeros)
.describe("Return a new array of given shape, type, and context, filled with zeros.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_npi_ones)
.describe("Return a new array of given shape, type, and context, filled with ones.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_np_zeros_like)
.describe(R"code(Return an array of zeros with the same shape and type as a given array.

Examples::

  x = [[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]

  zeros_like(x) = [[ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol",
              "The shape and data-type of a define these same attributes of the returned array.");

NNVM_REGISTER_OP(_np_ones_like)
.describe(R"code(Return an array of ones with the same shape and type as a given array.

Examples::

  x = [[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]]

  ones_like(x) = [[ 1.,  1.,  1.],
                  [ 1.,  1.,  1.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
  [](const NodeAttrs& attrs) {
    return std::vector<uint32_t>(1, 0);
  })
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"a"};
  })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("a", "NDArray-or-Symbol",
              "The shape and data-type of a define these same attributes of the returned array.");

}  // namespace op
}  // namespace mxnet

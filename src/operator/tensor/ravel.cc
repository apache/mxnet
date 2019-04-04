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
 * \file ravel.cc
 * \brief CPU implementation of operators for ravel/unravel.
 */
#include "./ravel.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(RavelParam);

NNVM_REGISTER_OP(_ravel_multi_index)
.add_alias("ravel_multi_index")
.describe(R"code(Converts a batch of index arrays into an array of flat indices. The operator follows numpy conventions so a single multi index is given by a column of the input matrix. 

Examples::
   
   A = [[3,6,6],[4,5,1]]
   ravel(A, shape=(7,6)) = [22,41,37]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RavelParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"data"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", RavelOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", RavelForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Batch of multi-indices")
.add_arguments(RavelParam::__FIELDS__());

NNVM_REGISTER_OP(_unravel_index)
.add_alias("unravel_index")
.describe(R"code(Converts an array of flat indices into a batch of index arrays. The operator follows numpy conventions so a single multi index is given by a column of the output matrix.

Examples::

   A = [22,41,37]
   unravel(A, shape=(7,6)) = [[3,6,6],[4,5,1]]

)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RavelParam>)
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& attrs)
  { return std::vector<ResourceRequest>{ResourceRequest::kTempSpace}; })
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs)
  { return std::vector<std::string>{"data"}; } )
.set_attr<nnvm::FInferShape>("FInferShape", UnravelOpShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FCompute>("FCompute<cpu>", UnravelForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "Array of flat indices")
.add_arguments(RavelParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

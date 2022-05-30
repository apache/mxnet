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
 * \file swapaxis.cc
 * \brief
 * \author Ming Zhang
 */

#include "./swapaxis-inl.h"
#include "operator/elemwise_op_common.h"
#include "operator/operator_common.h"
#include "operator/tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(SwapAxisParam);

NNVM_REGISTER_OP(SwapAxis)
    .add_alias("swapaxes")
    .add_alias("_npi_swapaxes")
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<SwapAxisParam>)
    .describe(R"code(Interchanges two axes of an array.

Examples::

  x = [[1, 2, 3]])
  swapaxes(x, 0, 1) = [[ 1],
                       [ 2],
                       [ 3]]

  x = [[[ 0, 1],
        [ 2, 3]],
       [[ 4, 5],
        [ 6, 7]]]  // (2,2,2) array

 swapaxes(x, 0, 2) = [[[ 0, 4],
                       [ 2, 6]],
                      [[ 1, 5],
                       [ 3, 7]]]
)code" ADD_FILELINE)
    .set_attr<FCompute>("FCompute<cpu>", SwapAxisCompute<cpu>)
    .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
    .set_attr<mxnet::FInferShape>("FInferShape", SwapAxisShape)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_SwapAxis"})
    .add_argument("data", "NDArray-or-Symbol", "Input array.")
    .add_arguments(SwapAxisParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_SwapAxis)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<SwapAxisParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int>>{{0, 0}};
                                    })
    .set_attr<FCompute>("FCompute<cpu>", SwapAxisGrad<cpu>)
    .set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity", [](const NodeAttrs& attrs) {
      return std::vector<bool>{true};
    });

}  // namespace op
}  // namespace mxnet

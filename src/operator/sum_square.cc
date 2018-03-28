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
 * \file sum_square.cc
 * \brief 
 * \author Hang Zhang
*/
#include "sum_square-inl.h"
#include "elemwise_op_common.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(SumSquare)
.describe(R"code(In-device Sum and Sum of Square.
)code" ADD_FILELINE)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::FInferShape>("FInferShape", SumSquareInferShape)
.set_attr<nnvm::FInferType>("FInferType", SumSquareInferType)
.set_attr<FInferStorageType>("FInferStorageType", SumSquareStorageType)
.set_attr<FCompute>("FCompute<cpu>", SumSquareForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_SumSquare"})
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
;

NNVM_REGISTER_OP(_backward_SumSquare)
.set_num_outputs(1)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", backward_SumSquareStorageType)
.set_attr<FCompute>("FCompute<cpu>", SumSquareBackward<cpu>);
;

}  // namespace op
}  // namespace mxnet

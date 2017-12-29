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
 *  Copyright (c) 2016 by Contributors
 * \file init_op.cc
 * \brief CPU Implementation of init op
 */
#include "./init_op.h"
#include "./elemwise_unary_op.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(InitOpParam);
DMLC_REGISTER_PARAMETER(InitOpWithScalarParam);
DMLC_REGISTER_PARAMETER(RangeParam);


NNVM_REGISTER_OP(_zeros)
.describe("fill target with zeros")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FillComputeZerosEx<cpu>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_ones)
.describe("fill target with ones")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_full)
  .describe("fill target with a scalar value")
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr_parser(ParamParser<InitOpWithScalarParam>)
  .set_attr<nnvm::FInferShape>("FInferShape", InitShape<InitOpWithScalarParam>)
  .set_attr<nnvm::FInferType>("FInferType", InitType<InitOpWithScalarParam>)
  .set_attr<FCompute>("FCompute<cpu>", InitFillWithScalarCompute<cpu>)
.add_arguments(InitOpWithScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_arange)
.describe("Return evenly spaced values within a given interval. Similar to Numpy")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(RangeParamParser)
.set_attr<nnvm::FInferShape>("FInferShape", RangeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<RangeParam>)
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(zeros_like)
.add_alias("_sparse_zeros_like")
.describe(R"code(Return an array of zeros with the same shape and type
as the input array.

The storage type of ``zeros_like`` output depends on the storage type of the input

- zeros_like(row_sparse) = row_sparse
- zeros_like(csr) = csr
- zeros_like(default) = default

Examples::

  x = [[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]]

  zeros_like(x) = [[ 0.,  0.,  0.],
                   [ 0.,  0.,  0.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 1, false, true, true>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FillComputeZerosEx<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

NNVM_REGISTER_OP(ones_like)
.describe(R"code(Return an array of ones with the same shape and type
as the input array.

Examples::

  x = [[ 0.,  0.,  0.],
       [ 0.,  0.,  0.]]

  ones_like(x) = [[ 1.,  1.,  1.],
                  [ 1.,  1.,  1.]]

)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

}  // namespace op
}  // namespace mxnet

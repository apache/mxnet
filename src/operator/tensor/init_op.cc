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
DMLC_REGISTER_PARAMETER(InitOpWithoutDTypeParam);
DMLC_REGISTER_PARAMETER(RangeParam);
DMLC_REGISTER_PARAMETER(RangeLikeParam);
DMLC_REGISTER_PARAMETER(EyeParam);
DMLC_REGISTER_PARAMETER(LinspaceParam);

NNVM_REGISTER_OP(_zeros_without_dtype)
.describe("fill target with zeros without default dtype")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpWithoutDTypeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpWithoutDTypeParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpWithoutDTypeParam>)
.set_attr<FInferStorageType>("FInferStorageType",
  InitStorageType<InitOpWithoutDTypeParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FillComputeZerosEx<cpu>)
.add_arguments(InitOpWithoutDTypeParam::__FIELDS__());

NNVM_REGISTER_OP(_zeros)
.describe("fill target with zeros")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<FComputeEx>("FComputeEx<cpu>", FillComputeZerosEx<cpu>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_eye)
.describe("Return a 2-D array with ones on the diagonal and zeros elsewhere.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<EyeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitEyeShape<EyeParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<EyeParam>)
.set_attr<FCompute>("FCompute<cpu>", EyeFill<cpu>)
.add_arguments(EyeParam::__FIELDS__());

NNVM_REGISTER_OP(_ones)
.describe("fill target with ones")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_full)
  .describe("fill target with a scalar value")
  .set_num_inputs(0)
  .set_num_outputs(1)
  .set_attr_parser(ParamParser<InitOpWithScalarParam>)
  .set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpWithScalarParam>)
  .set_attr<nnvm::FInferType>("FInferType", InitType<InitOpWithScalarParam>)
  .set_attr<FCompute>("FCompute<cpu>", InitFillWithScalarCompute<cpu>)
.add_arguments(InitOpWithScalarParam::__FIELDS__());

NNVM_REGISTER_OP(_arange)
.describe("Return evenly spaced values within a given interval. Similar to Numpy")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(RangeParamParser)
.set_attr<mxnet::FInferShape>("FInferShape", RangeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<RangeParam>)
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu, RangeParam>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(_contrib_arange_like)
.describe(R"code(Return an array with evenly spaced values. If axis is not given, the output will 
have the same shape as the input array. Otherwise, the output will be a 1-D array with size of 
the specified axis in input shape.

Examples::

  x = [[0.14883883 0.7772398  0.94865847 0.7225052 ]
       [0.23729339 0.6112595  0.66538996 0.5132841 ]
       [0.30822644 0.9912457  0.15502319 0.7043658 ]]
       <NDArray 3x4 @cpu(0)>

  out = mx.nd.contrib.arange_like(x, start=0)

    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]
     <NDArray 3x4 @cpu(0)>

  out = mx.nd.contrib.arange_like(x, start=0, axis=-1)

    [0. 1. 2. 3.]
    <NDArray 4 @cpu(0)>
)code")
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr_parser(ParamParser<RangeLikeParam>)
.set_attr<mxnet::FInferShape>("FInferShape", RangeLikeShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<RangeLikeParam, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", RangeCompute<cpu, RangeLikeParam>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

NNVM_REGISTER_OP(_linspace)
.add_alias("_npi_linspace")
.describe("Return evenly spaced numbers over a specified interval. Similar to Numpy")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LinspaceParam>)
.set_attr<mxnet::FInferShape>("FInferShape", LinspaceShape)
.set_attr<nnvm::FInferType>("FInferType", InitType<LinspaceParam>)
.set_attr<FCompute>("FCompute<cpu>", LinspaceCompute<cpu>)
.add_arguments(RangeParam::__FIELDS__());

NNVM_REGISTER_OP(zeros_like)
MXNET_ADD_SPARSE_OP_ALIAS(zeros_like)
.describe(R"code(Return an array of zeros with the same shape, type and storage type
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
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
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
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<1, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)
.set_attr<nnvm::FIgnoreInputs>("FIgnoreInputs",
    [](const NodeAttrs& attrs) { return std::vector<uint32_t>(1, 0); })
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("data", "NDArray-or-Symbol", "The input");

}  // namespace op
}  // namespace mxnet

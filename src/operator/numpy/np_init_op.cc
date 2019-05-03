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

NNVM_REGISTER_OP(_np_zeros)
.describe("Return a new array of given shape, type, and context, filled with zeros.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FInferStorageType>("FInferStorageType", InitStorageType<InitOpParam, true, true>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 0>)
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.add_arguments(InitOpParam::__FIELDS__());

NNVM_REGISTER_OP(_np_ones)
.describe("Return a new array of given shape, type, and context, filled with ones.")
.set_num_inputs(0)
.set_num_outputs(1)
.set_attr_parser(ParamParser<InitOpParam>)
.set_attr<mxnet::FInferShape>("FInferShape", InitShape<InitOpParam>)
.set_attr<nnvm::FInferType>("FInferType", InitType<InitOpParam>)
.set_attr<FCompute>("FCompute<cpu>", FillCompute<cpu, 1>)
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true)
.add_arguments(InitOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

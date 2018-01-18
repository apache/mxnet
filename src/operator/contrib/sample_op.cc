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
 *  Copyright (c) 2017 by Contributors
 * \file sample_op.cc
 * \brief
 */
#include "./sample_op-inl.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_accidental_hits)
.describe(R"code(Compute accidental hits
)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", AccidentalHitShape)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<2, 1>)
.set_attr<FInferStorageType>("FInferStorageType", AccidentalHitStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", AccidentalHitComputeEx<cpu>)
.add_argument("label", "NDArray-or-Symbol", "Label")
.add_argument("sample", "NDArray-or-Symbol", "Sample");

}  // namespace op
}  // namespace mxnet

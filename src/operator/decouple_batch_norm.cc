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
 * Copyright (c) 2015 by Contributors
 * \file sync_batch_norm.cc
 * \brief cpu sync BN operator
 * \author Hang Zhang
 * Adapted from BatchNormV1
*/
#include "decouple_batch_norm-inl.h"
#include "elemwise_op_common.h"

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(DecoupleBatchNorm)
.describe(R"code(Synchronized Cross-GPU Batch normalization.
)code" ADD_FILELINE)
.set_num_inputs(5)
.set_num_outputs(1)
.set_attr<nnvm::FInferShape>("FInferShape", DecoupleBNInferShape)
.set_attr<nnvm::FInferType>("FInferType", DecoupleBNInferType)
.set_attr<FInferStorageType>("FInferStorageType", DecoupleBNStorageType)
.set_attr<FCompute>("FCompute<cpu>", DecoupleBNForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseInOut{"_backward_DecoupleBatchNorm"})
.add_argument("data", "NDArray-or-Symbol", "Input data to batch normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_argument("mean", "NDArray-or-Symbol", "mean array")
.add_argument("std", "NDArray-or-Symbol", "std array")
;

NNVM_REGISTER_OP(_backward_DecoupleBatchNorm)
.set_num_outputs(5)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<FInferStorageType>("FInferStorageType", backward_DecoupleBNStorageType)
.set_attr<FCompute>("FCompute<cpu>", DecoupleBNBackward<cpu>);
;

}  // namespace op
}  // namespace mxnet

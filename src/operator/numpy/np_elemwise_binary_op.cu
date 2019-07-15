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
 * \file np_elemwise_binary_op.cu
 * \brief GPU Implementation of numpy-compatible bitwise_xor
 */



#include <mxnet/base.h>
#include "../mshadow_op.h"  // mshadow operations
#include "../operator_common.h"  // MakeZeroGradNodes
#include "../tensor/elemwise_binary_op.h"  // ElemwiseShape, ElemwiseType
#include "../tensor/elemwise_binary_broadcast_op.h"  // BinaryBroadcastCompute

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_np_bitwise_xor)
.set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastCompute<gpu,
    mshadow_op::bitwise_xor>);
}   // namespace op
}   // namespace mxnet

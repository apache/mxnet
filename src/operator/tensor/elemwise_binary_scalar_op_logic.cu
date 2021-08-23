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
 * \file elemwise_binary_scalar_op_logic.cu
 * \brief GPU Implementation of binary scalar logic functions.
 */
#include "elemwise_binary_scalar_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"equal"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"equal"});

NNVM_REGISTER_OP(_not_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"not_equal"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"not_equal"});

NNVM_REGISTER_OP(_greater_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"greater"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"greater"});

NNVM_REGISTER_OP(_greater_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"greater_equal"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"greater_equal"});

NNVM_REGISTER_OP(_lesser_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"less"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"less"});

NNVM_REGISTER_OP(_lesser_equal_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"less_equal"})
.set_attr<FComputeEx>("FComputeEx<gpu>", BinaryScalarRTCCompute{"less_equal"});

NNVM_REGISTER_OP(_logical_and_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"logical_and"});

NNVM_REGISTER_OP(_logical_or_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"logical_or"});

NNVM_REGISTER_OP(_logical_xor_scalar)
.set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"logical_xor"});

}  // namespace op
}  // namespace mxnet

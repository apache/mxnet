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
 * \file np_elemwise_broadcast_op_extended_thi.cu
 * \brief GPU Implementation of extended functions for elementwise binary broadcast operator. (Third
 * extended file)
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_bitwise_left_shift)
    .set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"bitwise_left_shift"});

NNVM_REGISTER_OP(_npi_bitwise_left_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"bitwise_left_shift"});

NNVM_REGISTER_OP(_npi_rbitwise_left_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rbitwise_left_shift"});

NNVM_REGISTER_OP(_backward_npi_bitwise_left_shift)
    .set_attr<FCompute>("FCompute<gpu>",
                        BinaryBroadcastRTCBackwardUseIn{"bitwise_left_shift_grad",
                                                        "bitwise_left_shift_rgrad"});

NNVM_REGISTER_OP(_backward_npi_bitwise_left_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"bitwise_left_shift_grad"});

NNVM_REGISTER_OP(_backward_npi_rbitwise_left_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rbitwise_left_shift_grad"});

NNVM_REGISTER_OP(_npi_bitwise_right_shift)
    .set_attr<FCompute>("FCompute<gpu>", BinaryBroadcastRTCCompute{"bitwise_right_shift"});

NNVM_REGISTER_OP(_npi_bitwise_right_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"bitwise_right_shift"});

NNVM_REGISTER_OP(_npi_rbitwise_right_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCCompute{"rbitwise_right_shift"});

NNVM_REGISTER_OP(_backward_npi_bitwise_right_shift)
    .set_attr<FCompute>("FCompute<gpu>",
                        BinaryBroadcastRTCBackwardUseIn{"bitwise_right_shift_grad",
                                                        "bitwise_right_shift_rgrad"});

NNVM_REGISTER_OP(_backward_npi_bitwise_right_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"bitwise_right_shift_grad"});

NNVM_REGISTER_OP(_backward_npi_rbitwise_right_shift_scalar)
    .set_attr<FCompute>("FCompute<gpu>", BinaryScalarRTCBackward{"rbitwise_right_shift_grad"});

}  // namespace op
}  // namespace mxnet

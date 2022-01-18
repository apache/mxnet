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
 * \file np_elemwise_binary_op_mul.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary multiply.
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_multiply)
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBroadcastComputeWithBool<cpu,
                                                            op::mshadow_op::mul,
                                                            op::mshadow_op::mixed_mul,
                                                            op::mshadow_op::mixed_mul>)
#if MXNET_USE_ONEDNN == 1
    .set_attr<FComputeEx>("FComputeEx<cpu>", NumpyBinaryOperatorComputeExCPU<op::mshadow_op::mul>)
    .set_attr<FInferStorageType>("FInferStorageType", NumpyBinaryBroadcastStorageType)
#endif  // MXNET_USE_ONEDNN
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_mul"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mul)
    .set_num_inputs(3)
    .set_num_outputs(2)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 1}};
                                    })
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBackwardUseIn<cpu, mshadow_op::right, mshadow_op::left>);

}  // namespace op
}  // namespace mxnet

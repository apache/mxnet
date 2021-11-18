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
 * \file np_elemwise_binary_op_mod.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary mod.
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_NP_BINARY_MIXED_PRECISION(_npi_mod)
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryBroadcastCompute<cpu,
                                                    op::mshadow_op::mod,
                                                    op::mshadow_op::mixed_mod,
                                                    op::mshadow_op::mixed_rmod>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_broadcast_mod"});

NNVM_REGISTER_OP(_backward_npi_broadcast_mod)
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
                        NumpyBinaryBackwardUseIn<cpu, mshadow_op::mod_grad, mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet

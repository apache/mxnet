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
 * \file np_elemwise_broadcast_op_lae.cc
 * \brief CPU Implementation of basic functions for elementwise numpy binary logaddexp.
 */

#include "./np_elemwise_broadcast_op.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_logaddexp)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"x1", "x2"};
                                     })
    .set_attr<mxnet::FInferShape>("FInferShape", BinaryBroadcastShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyBinaryMixedFloatingType)
    .set_attr<FCompute>("FCompute<cpu>",
                        NumpyBinaryMixedFloatingCompute<cpu, mshadow_op::logaddexp>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_logaddexp"})
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<nnvm::FInplaceOption>("FInplaceOption",
                                    [](const NodeAttrs& attrs) {
                                      return std::vector<std::pair<int, int> >{{0, 0}};
                                    })
    .add_argument("x1", "NDArray-or-Symbol", "The input array")
    .add_argument("x2", "NDArray-or-Symbol", "The input array");

MXNET_OPERATOR_REGISTER_NP_BINARY_SCALAR(_npi_logaddexp_scalar)
    .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Compute<cpu, mshadow_op::logaddexp>)
    .set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_npi_logaddexp_scalar"});

NNVM_REGISTER_OP(_backward_npi_logaddexp)
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
    .set_attr<FCompute>(
        "FCompute<cpu>",
        NumpyBinaryBackwardUseIn<cpu, mshadow_op::logaddexp_grad, mshadow_op::logaddexp_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_backward_npi_logaddexp_scalar)
    .add_arguments(NumpyBinaryScalarParam::__FIELDS__())
    .set_attr_parser(ParamParser<NumpyBinaryScalarParam>)
    .set_attr<FCompute>("FCompute<cpu>", BinaryScalarOp::Backward<cpu, mshadow_op::logaddexp_grad>);

}  // namespace op
}  // namespace mxnet

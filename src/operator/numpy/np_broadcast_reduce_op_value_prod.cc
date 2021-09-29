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
 * \file np_broadcast_reduce_op_value_prod.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */

#if MXNET_USE_TVM_OP
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "np_broadcast_reduce_op_value.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_prod)
    .add_alias("_np_product")
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyReduceAxesParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpySumType)
    .add_arguments(NumpyReduceAxesParam::__FIELDS__())
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "The input")
    .set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesCompute<cpu, mshadow_op::product, true>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_npi_prod"});

NNVM_REGISTER_OP(_backward_npi_prod)
    .set_num_inputs(3)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyReduceAxesParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesBackwardUseInOut<cpu, mshadow_op::rdiv>);

}  // namespace op
}  // namespace mxnet

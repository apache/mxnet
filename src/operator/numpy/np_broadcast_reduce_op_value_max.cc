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
 * \file np_broadcast_reduce_op_value_max.cc
 * \brief CPU Implementation of broadcast and reduce functions based on value.
 */

#if MXNET_USE_TVM_OP
#include "../tvmop/op_module.h"
#endif  // MXNET_USE_TVM_OP

#include "np_broadcast_reduce_op_value.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_npi_max)
    .add_alias("_npi_amax")
    .describe(R"code()code" ADD_FILELINE)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
    .set_attr<mxnet::FInferShape>("FInferShape", NumpyReduceAxesNoDTypeShape)
    .set_attr<nnvm::FInferType>("FInferType", NumpyReduceAxesNoDTypeType)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
                                     [](const NodeAttrs& attrs) {
                                       return std::vector<std::string>{"a"};
                                     })
    .add_argument("a", "NDArray-or-Symbol", "The input")
    .add_arguments(NumpyReduceAxesNoDTypeParam::__FIELDS__())
    .set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeCompute<cpu, mshadow::red::maximum>)
    .set_attr<FResourceRequest>("FResourceRequest",
                                [](const NodeAttrs& attrs) {
                                  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
                                })
    .set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
    .set_attr<nnvm::FGradient>("FGradient", ReduceGrad{"_backward_npi_max"});

NNVM_REGISTER_OP(_backward_npi_max)
    .set_num_outputs(1)
    .set_attr_parser(ParamParser<NumpyReduceAxesNoDTypeParam>)
    .set_attr<nnvm::TIsBackward>("TIsBackward", true)
    .set_num_inputs(3)
    .set_attr<FCompute>("FCompute<cpu>", NumpyReduceAxesNoDTypeBackward<cpu, mshadow_op::eq>);

}  // namespace op
}  // namespace mxnet

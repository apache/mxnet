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

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>

namespace mxnet {
namespace op {

static void ForeachComputeExCPU(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<NDArray>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<NDArray>& outputs) {
  CHECK(attrs.g != nullptr);
}

NNVM_REGISTER_OP(Foreach)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"fn", "data1", "data2"};
})
.set_attr<nnvm::FInputGraph>("FInputGraph",
    [](const NodeAttrs& attrs) {
  return 0;
})
//.set_attr<nnvm::FInferShape>("FInferShape", ConvolutionShape)
//.set_attr<nnvm::FInferType>("FInferType", ConvolutionType)
.describe(R"code(test)code" ADD_FILELINE)
//.set_attr_parser(ParamParser<ActivationParam>)
//.set_attr<FInferStorageType>("FInferStorageType", ActivationStorageType)
.set_attr<FComputeEx>("FComputeEx<cpu>", ForeachComputeExCPU)
.add_argument("fn", "Symbol", "Input graph.")
.add_argument("data1", "NDArray-or-Symbol", "Input1.")
.add_argument("data2", "NDArray-or-Symbol", "Input2.");
//.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

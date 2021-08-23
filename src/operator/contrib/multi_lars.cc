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
 *  Copyright (c) 2019 by Contributors
 * \file multi_lars.cc
 * \brief vectorized LARS coefficient computed from sums of squared weights and grads
 * \author Clement Fuji Tsang
 */

#include "./multi_lars-inl.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LARSParam);

NNVM_REGISTER_OP(multi_lars)
.describe(R"code(Compute the LARS coefficients of multiple weights and grads from their sums of square"
)code" ADD_FILELINE)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<LARSParam>)
.set_attr<mxnet::FInferShape>("FInferShape", ElemwiseShape<4, 1>)
.set_attr<nnvm::FInferType>("FInferType", ElemwiseType<4, 1>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<4, 1, false, false, false>)
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const nnvm::NodeAttrs& attrs) {
    std::vector<std::string> list_input_names = {"lrs", "weights_sum_sq", "grads_sum_sq", "wds"};
    return list_input_names;
  })
.set_attr<FCompute>("FCompute<cpu>", MultiLARS<cpu>)
.add_argument("lrs", "NDArray-or-Symbol", "Learning rates to scale by LARS coefficient")
.add_argument("weights_sum_sq", "NDArray-or-Symbol", "sum of square of weights arrays")
.add_argument("grads_sum_sq", "NDArray-or-Symbol", "sum of square of gradients arrays")
.add_argument("wds", "NDArray-or-Symbol", "weight decays")
.add_arguments(LARSParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

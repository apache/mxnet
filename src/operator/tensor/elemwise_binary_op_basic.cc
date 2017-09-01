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
 * \file elemwise_binary_scalar_op.cc
 * \brief CPU Implementation of unary function.
 */
#include "./elemwise_unary_op.h"
#include "./elemwise_binary_op.h"

namespace mxnet {
namespace op {
MXNET_OPERATOR_REGISTER_BINARY(elemwise_add)
.add_alias("_add").add_alias("_plus").add_alias("_Plus").add_alias("_sparse_elemwise_add")
.describe(R"code(Adds arguments element-wise.

The storage type of ``elemwise_add`` output depends on storage types of inputs

- elemwise_add(row_sparse, row_sparse) = row_sparse
- otherwise, ``elemwise_add`` generates output with default storage

)code")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::plus>)
.set_attr<nnvm::FGradient>("FGradient", CloneGradient{"_backward_add"})
.set_attr<FComputeEx>("FComputeEx<cpu>", BinaryComputeEx<cpu, mshadow::op::plus>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<2, 1>);

// specialized gradient add function to do add to optimization
// this must differ from elemwise_add to prevent add to optimization in forward pass.
MXNET_OPERATOR_REGISTER_BINARY(_grad_add)
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::plus>);

NNVM_REGISTER_OP(_backward_add)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::identity>)
.set_attr<FComputeEx>("FComputeEx<cpu>",
                      BinaryBackwardUseNoneEx<cpu, mshadow_op::identity, mshadow_op::identity>)
.set_attr<FInferStorageType>("FInferStorageType", ElemwiseStorageType<1, 2>);

MXNET_OPERATOR_REGISTER_BINARY(_sub)
.add_alias("_minus").add_alias("_Minus")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::minus>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_backward_sub"});

NNVM_REGISTER_OP(_backward_sub)
.set_num_inputs(1)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 0}, {0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBackwardUseNone<cpu, mshadow_op::identity,
                                                                mshadow_op::negation>);

MXNET_OPERATOR_REGISTER_BINARY(_mul)
.add_alias("_Mul")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::mul>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mul"});

NNVM_REGISTER_OP(_backward_mul)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBackwardUseIn<cpu, mshadow_op::right,
                                                              mshadow_op::left>);

MXNET_OPERATOR_REGISTER_BINARY(_div)
.add_alias("_Div")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow::op::div>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_div"});

NNVM_REGISTER_OP(_backward_div)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBackwardUseIn<cpu, mshadow_op::div_grad,
                                                              mshadow_op::div_rgrad>);

MXNET_OPERATOR_REGISTER_BINARY(_mod)
.add_alias("_Mod")
.set_attr<FCompute>("FCompute<cpu>", BinaryCompute<cpu, mshadow_op::mod>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseIn{"_backward_mod"});

NNVM_REGISTER_OP(_backward_mod)
.set_num_inputs(3)
.set_num_outputs(2)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs){
    return std::vector<std::pair<int, int> >{{0, 1}};
  })
.set_attr<FCompute>("FCompute<cpu>", BinaryBackwardUseIn<cpu, mshadow_op::mod_grad,
                                                         mshadow_op::mod_rgrad>);

}  // namespace op
}  // namespace mxnet

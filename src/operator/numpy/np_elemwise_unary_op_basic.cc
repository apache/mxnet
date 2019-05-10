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
 * \file np_elemwise_unary_op_basic.cc
 * \brief CPU Implementation of numpy elementwise unary function.
 */
#include <mxnet/base.h>
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

MXNET_OPERATOR_REGISTER_UNARY(_numpy__ext_relu)
.describe(R"code(Computes rectified linear activation.

.. math::
   max(features, 0)

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::relu>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_relu"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true);

MXNET_OPERATOR_REGISTER_UNARY(_numpy__ext_sigmoid)
.describe(R"code(Computes sigmoid of x element-wise.

.. math::
   y = 1 / (1 + exp(-x))

)code" ADD_FILELINE)
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::Compute<cpu, mshadow_op::sigmoid>)
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseOut{"_backward_sigmoid"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true);

MXNET_OPERATOR_REGISTER_UNARY(_np_copy)
.MXNET_DESCRIBE("Returns a copy of the input.")
.set_attr<FCompute>("FCompute<cpu>", UnaryOp::IdentityCompute<cpu>)
.set_attr<nnvm::FInplaceIdentity>("FInplaceIdentity",
  [](const NodeAttrs& attrs){
    return std::vector<bool>{true};
  })
.set_attr<nnvm::FGradient>("FGradient", ElemwiseGradUseNone{"_copy"})
.set_attr<mxnet::TIsNumpyCompatible>("TIsNumpyCompatible", true);

}  // namespace op
}  // namespace mxnet

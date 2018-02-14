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
 * \file regression_ouput.cu
 * \brief Regression output operator.
*/
#include "./regression_output-inl.h"


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(LinearRegressionOutput)
.set_attr<FCompute>("FCompute<gpu>", RegressionForward<gpu, mshadow_op::identity>);

NNVM_REGISTER_OP(_backward_linear_reg_out)
.set_attr<FCompute>("FCompute<gpu>", RegressionBackward<gpu, mshadow_op::minus>);

NNVM_REGISTER_OP(MAERegressionOutput)
.set_attr<FCompute>("FCompute<gpu>", RegressionForward<gpu, mshadow_op::identity>);

NNVM_REGISTER_OP(_backward_mae_reg_out)
.set_attr<FCompute>("FCompute<gpu>", RegressionBackward<gpu, mshadow_op::minus_sign>);

NNVM_REGISTER_OP(LogisticRegressionOutput)
.set_attr<FCompute>("FCompute<gpu>", RegressionForward<gpu, mshadow_op::sigmoid>);

NNVM_REGISTER_OP(_backward_logistic_reg_out)
.set_attr<FCompute>("FCompute<gpu>", RegressionBackward<gpu, mshadow_op::minus>);

}  // namespace op
}  // namespace mxnet

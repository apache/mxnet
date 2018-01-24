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
 * \file regression_ouput.cc
 * \brief Regression output operator.
*/

#include "./regression_output-inl.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(RegressionOutputParam);

NNVM_REGISTER_OP(LinearRegressionOutput)
.describe(R"code(Computes and optimizes for squared loss during backward propagation.
Just outputs ``data`` during forward propagation.

If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
then the squared loss estimated over :math:`n` samples is defined as

:math:`\text{SquaredLoss}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert  \textbf{y}_i - \hat{\textbf{y}}_i  \rVert_2`

.. note::
   Use the LinearRegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of dimensions of a training example.
The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "label"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", RegressionOpShape)
.set_attr<nnvm::FGradient>("FGradient", RegressionOpGrad{"_backward_linear_reg_out"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionForward<cpu, mshadow_op::identity>)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_linear_reg_out)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<RegressionOutputParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  // inputs are in_label and out_data, outputs are data_grad and label_grad
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionBackward<cpu, mshadow_op::minus>);

NNVM_REGISTER_OP(MAERegressionOutput)
.describe(R"code(Computes mean absolute error of the input.

MAE is a risk metric corresponding to the expected value of the absolute error.

If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
then the mean absolute error (MAE) estimated over :math:`n` samples is defined as

:math:`\text{MAE}(\textbf{Y}, \hat{\textbf{Y}} ) = \frac{1}{n} \sum_{i=0}^{n-1} \lVert \textbf{y}_i - \hat{\textbf{y}}_i \rVert_1`

.. note::
   Use the MAERegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of dimensions of a training example.
The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "label"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", RegressionOpShape)
.set_attr<nnvm::FGradient>("FGradient", RegressionOpGrad{"_backward_mae_reg_out"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionForward<cpu, mshadow_op::identity>)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_mae_reg_out)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<RegressionOutputParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  // inputs are in_label and out_data, outputs are data_grad and label_grad
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionBackward<cpu, mshadow_op::minus_sign>);


NNVM_REGISTER_OP(LogisticRegressionOutput)
.describe(R"code(Applies a logistic function to the input.

The logistic function, also known as the sigmoid function, is computed as
:math:`\frac{1}{1+exp(-\textbf{x})}`.

Commonly, the sigmoid is used to squash the real-valued output of a linear model
:math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
It is suitable for binary classification or probability prediction tasks.

.. note::
   Use the LogisticRegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/m`, where m is the number of dimensions of a training example.
The parameter `grad_scale` can be used to change this scale to `grad_scale/m`.

)code" ADD_FILELINE)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data", "label"};
  })
.set_attr<nnvm::FInferShape>("FInferShape", RegressionOpShape)
.set_attr<nnvm::FGradient>("FGradient", RegressionOpGrad{"_backward_logistic_reg_out"})
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionForward<cpu, mshadow_op::sigmoid>)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

NNVM_REGISTER_OP(_backward_logistic_reg_out)
.set_num_inputs(2)
.set_num_outputs(2)
.set_attr_parser(ParamParser<RegressionOutputParam>)
.set_attr<nnvm::TIsBackward>("TIsBackward", true)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
[](const NodeAttrs& attrs){
  // inputs are in_label and out_data, outputs are data_grad and label_grad
  return std::vector<std::pair<int, int> >{{1, 0}};
})
.set_attr<FCompute>("FCompute<cpu>", RegressionBackward<cpu, mshadow_op::minus>);



}  // namespace op
}  // namespace mxnet

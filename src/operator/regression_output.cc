/*!
 * Copyright (c) 2015 by Contributors
 * \file regression_output.cc
 * \brief regression output operator
*/
#include "./regression_output-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {

template<>
Operator *CreateRegressionOutputOp<cpu>(reg_enum::RegressionOutputType type,
                                        RegressionOutputParam param) {
  switch (type) {
    case reg_enum::kLinear:
      return new RegressionOutputOp<cpu, mshadow::op::identity, mshadow::op::minus>(param);
    case reg_enum::kLogistic:
      return new RegressionOutputOp<cpu, mshadow_op::sigmoid, mshadow::op::minus>(param);
    case reg_enum::kMAE:
      return new RegressionOutputOp<cpu, mshadow::op::identity, mshadow_op::minus_sign>(param);
    default:
      LOG(FATAL) << "unknown activation type " << type;
  }
  return nullptr;
}

// DO_BIND_DISPATCH comes from operator_common.h
template<reg_enum::RegressionOutputType type>
Operator *RegressionOutputProp<type>::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateRegressionOutputOp, type, param_);
}

DMLC_REGISTER_PARAMETER(RegressionOutputParam);

MXNET_REGISTER_OP_PROPERTY(LinearRegressionOutput, RegressionOutputProp<reg_enum::kLinear>)
.describe(R"code(Computes and optimizes for squared loss.

.. note::
   Use the LinearRegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.
The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(MAERegressionOutput, RegressionOutputProp<reg_enum::kMAE>)
.describe(R"code(Computes mean absolute error of the input.

MAE is a risk metric corresponding to the expected value of the absolute error.

If :math:`\hat{y}_i` is the predicted value of the i-th sample, and :math:`y_i` is the corresponding target value,
then the mean absolute error (MAE) estimated over :math:`n` samples is defined as

:math:`\text{MAE}(y, \hat{y} ) = \frac{1}{n} \sum_{i=0}^{n-1} \left| y_i - \hat{y}_i \right|`

.. note::
   Use the MAERegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.
The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

MXNET_REGISTER_OP_PROPERTY(LogisticRegressionOutput, RegressionOutputProp<reg_enum::kLogistic>)
.describe(R"code(Applies a logistic function to the input.

The logistic function, also known as the sigmoid function, is computed as
:math:`\frac{1}{1+exp(-x)}`.

Commonly, the sigmoid is used to squash the real-valued output of a linear model
:math:wTx+b into the [0,1] range so that it can be interpreted as a probability.
It is suitable for binary classification or probability prediction tasks.

.. note::
   Use the LogisticRegressionOutput as the final output layer of a net.

By default, gradients of this loss function are scaled by factor `1/n`, where n is the number of training examples.
The parameter `grad_scale` can be used to change this scale to `grad_scale/n`.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_argument("label", "NDArray-or-Symbol", "Input label to the function.")
.add_arguments(RegressionOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

/*!
 * Copyright (c) 2015 by Contributors
 * \file make_loss.cc
 * \brief special layer for propagating loss
*/
#include "./make_loss-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MakeLossParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MakeLossOp<cpu, DType>(param);
  });
  return op;
}

Operator *MakeLossProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                         std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MakeLossParam);

MXNET_REGISTER_OP_PROPERTY(MakeLoss, MakeLossProp)
.describe(R"code(Make your own loss function in network construction.
This operator accepts a customized loss function symbol as a terminal loss and
the symbol should be an operator with no backward dependency.
The output of this function is the gradient of loss with respect to the input data.

For example, if you are a making a weighted cross entropy loss function.

.. math::
  \sum_i w_i * (y_i * \log \hat{y_i} + (1 - y_i) * \log(1 - \hat{y_i}))


The following is a pseudocode snippet to create the customized loss::

  y = Variable('y')
  w = Variable('w')
  out = Activation(data = data, act_type = 'sigmoid')
  cross_entropy = y * log(out) + (1 - y) * log(1 - out)
  loss = MakeLoss(w * cross_entropy)

Notice: ``This operator is only useful as a Symbol instead of NDArray``

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(MakeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

/*!
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief softmax_activation op
 * \author Junyuan Xie
*/
#include "./softmax_activation-inl.h"
#include "./mshadow_op.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SoftmaxActivationParam param) {
  return new SoftmaxActivationOp<cpu>(param);
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxActivationProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(SoftmaxActivationParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxActivation, SoftmaxActivationProp)
.describe("Apply softmax activation to input. This is intended for internal layers. "
          "For output (loss layer) please use SoftmaxOutput. This operator takes "
          "an input tensor of dims [batch, d_1, ..., d_n]. If mode=instance, "
          "this operator will compute a softmax and normalize over the d_n "
          "dimension, treating all other dims as batch dims; this is the "
          "default mode. If mode=channel, this operator will compute a softmax "
          "and normalize over the d_1 dimension; this can be used for fully convolutional "
          "network, image segmentation, etc. The rank of the input "
          "tensor must be > 2 for this case.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(SoftmaxActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


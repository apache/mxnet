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
          "For output (loss layer) please use SoftmaxOutput. If mode=instance, "
          "this operator will compute a softmax for each instance in the batch; "
          "this is the default mode. If mode=channel, this operator will compute "
          "a num_channel-class softmax at each position of each instance; this can "
          "be used for fully convolutional network, image segmentation, etc.")
.add_argument("data", "Symbol", "Input data to activation function.")
.add_arguments(SoftmaxActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet


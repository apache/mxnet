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
 * Copyright (c) 2015 by Contributors
 * \file activation.cc
 * \brief softmax_activation op
 * \author Junyuan Xie
*/
#include "./softmax_activation-inl.h"
#include "../mshadow_op.h"

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
.describe(R"code(Applies softmax activation to input. This is intended for internal layers.

.. note::

  This operator has been deprecated, please use `softmax`.

If `mode` = ``instance``, this operator will compute a softmax for each instance in the batch.
This is the default mode.

If `mode` = ``channel``, this operator will compute a k-class softmax at each position
of each instance, where `k` = ``num_channel``. This mode can only be used when the input array
has at least 3 dimensions.
This can be used for `fully convolutional network`, `image segmentation`, etc.

Example::

  >>> input_array = mx.nd.array([[3., 0.5, -0.5, 2., 7.],
  >>>                            [2., -.4, 7.,   3., 0.2]])
  >>> softmax_act = mx.nd.SoftmaxActivation(input_array)
  >>> print softmax_act.asnumpy()
  [[  1.78322066e-02   1.46375655e-03   5.38485940e-04   6.56010211e-03   9.73605454e-01]
   [  6.56221947e-03   5.95310994e-04   9.73919690e-01   1.78379621e-02   1.08472735e-03]]

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.")
.add_arguments(SoftmaxActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

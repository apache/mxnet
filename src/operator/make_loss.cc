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

For example, if you are a making a cross entropy loss function. Assume ``out`` is the
predicted output and ``label`` is the true label, then the cross entropy can be defined as::

  cross_entropy = label * log(out) + (1 - label) * log(1 - out)
  loss = MakeLoss(cross_entropy)

We will need to use ``MakeLoss`` when we are creating our own loss function or we want to
combine multiple loss functions. Also we may want to stop some variables' gradients
from backpropagation. See more detail in ``BlockGrad`` or ``stop_gradient``.

In addition, we can give a scale to the loss by setting ``grad_scale``,
so that the gradient of the loss will be rescaled in the backpropagation.

.. note:: This operator should be used as a Symbol instead of NDArray.

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array.")
.add_arguments(MakeLossParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

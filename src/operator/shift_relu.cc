
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
 * \file shift_relu.cc
 * \brief
 * \author Kang Liang
*/

#include "./shift_relu-inl.h"

namespace mxnet {
namespace op {
template <>
Operator *CreateOp<cpu>(ShiftReLUParam param, int dtype, const TShape& shape) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType,
                           { op = new ShiftReLUOp<cpu, DType>(param); });
  return op;
}

Operator *ShiftReLUProp::CreateOperatorEx(Context ctx,
                                          std::vector<TShape> *in_shape,
                                          std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;

  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], (*in_shape)[0]);
}

DMLC_REGISTER_PARAMETER(ShiftReLUParam);

MXNET_REGISTER_OP_PROPERTY(ShiftReLU, ShiftReLUProp)
.describe(R"code(Applies Shift Leaky rectified linear unit activation element-wise to the input.

Shift ReLUs attempt to fix the "dying ReLU" problem by allowing a small `slope` and respawn the "died ReLU" by range.
when the input is smaller than shift and has a slope of one when input is positive.

The following Shift ReLU Activation function are supported:

srelu: Shift Linear Unit. `y = x > 0 ? x : (x - shift) * slope + shift`
       shift = rand(-0.5, 0.5) * scale * range
       range is set by user
       if v1:
        scale is the L1-norm/sizeof(Blob) of whole Blob
       if v2:
        scale is a vector of the L1-norm/sizeof(feature map) of every feature map in Blob

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to activation function.")
.add_arguments(ShiftReLUParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

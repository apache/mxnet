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
 * \file crop.cc
 * \brief
 * \author Wei Wu
*/

#include "./crop-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(CropParam param) {
  return new CropOp<cpu>(param);
}

Operator* CropProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(CropParam);

MXNET_REGISTER_OP_PROPERTY(Crop, CropProp)
.describe(R"code(

.. note:: `Crop` is deprecated. Use `slice` instead.

Crop the 2nd and 3rd dim of input data, with the corresponding size of h_w or
with width and height of the second input symbol, i.e., with one input, we need h_w to
specify the crop height and width, otherwise the second input symbol's size will be used
)code" ADD_FILELINE)

.add_argument("data", "Symbol or Symbol[]", "Tensor or List of Tensors, the second input "
"will be used as crop_like shape reference")
.add_arguments(CropParam::__FIELDS__())
.set_key_var_num_args("num_args");
}  // namespace op
}  // namespace mxnet

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
 * Copyright (c) 2017 by Contributors
 * \file grid_generator.cc
 * \brief
 * \author Xu Dong
*/

#include "./grid_generator-inl.h"

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(GridGeneratorParam param, int dtype) {
  Operator *op = NULL;
  if (dtype == mshadow::kFloat32) {
    op = new GridGeneratorOp<cpu, float>(param);
  } else {
    LOG(FATAL) << "Other DTypes are not supported!";
  }
  return op;
}

Operator *GridGeneratorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(GridGeneratorParam);

MXNET_REGISTER_OP_PROPERTY(GridGenerator, GridGeneratorProp)
.add_argument("data", "NDArray-or-Symbol", "Input data to the function.")
.add_arguments(GridGeneratorParam::__FIELDS__())
.describe("Generates 2D sampling grid for bilinear sampling.");

}  // namespace op
}  // namespace mxnet

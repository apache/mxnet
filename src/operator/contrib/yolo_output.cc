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
 * \file yolo_output.cc
 * \brief Yolo2Output op
 * \author Joshua Zhang
*/
#include "./yolo_output-inl.h"

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(Yolo2OutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_SGL_DBL_TYPE_SWITCH(dtype, DType, {
    op = new Yolo2OutputOp<cpu, DType>(param);
  });
  return op;
}

Operator *Yolo2OutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                    std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(Yolo2OutputParam);

MXNET_REGISTER_OP_PROPERTY(_contrib_Yolo2Output, Yolo2OutputProp)
.describe(R"code(Yolo v2 output layer.  This is a convolutional version as described in YOLO 9000 paper.

  Examples::

    x = mx.nd.random.uniform(shape=(1, 75, 20, 20))
    y = mx.nd.zeros((1, 1, 5))
    beta = mx.nd.zeros((1))
    output = Yolo2Output(x, y, beta, num_class=10)
    output.shape = (1, 2000, 5)

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the Yolo2OutputOp.")
.add_argument("label", "NDArray-or-Symbol", "Object detection labels.")
.add_argument("beta", "NDArray-or-Symbol", "Warm up counting buffer.")
.add_arguments(Yolo2OutputParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet

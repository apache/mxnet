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
 * \brief activation op
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "../mshadow_op.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "../mkl/mkl_memory-inl.h"
#include "../mkl/mkl_relu-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(ActivationParam param, int dtype, const TShape& dshape) {
  Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
  if (param.act_type == activation::kReLU && dshape.ndim() <= 4) {
      switch (dtype) {
      case mshadow::kFloat32:
          return new MKLReluOp<cpu, float>();
      case mshadow::kFloat64:
          return new MKLReluOp<cpu, double>();
      default:
          break;
      }
  }
  if (enableMKLWarnGenerated())
    LOG(INFO) << MKLReluOp<cpu, float>::getName() << " Skip MKL optimization";
#endif
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    switch (param.act_type) {
      case activation::kReLU:
        op = new ActivationOp<cpu, mshadow_op::relu, mshadow_op::relu_grad, DType>();
        break;
      case activation::kSigmoid:
        op = new ActivationOp<cpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad, DType>();
        break;
      case activation::kTanh:
        op = new ActivationOp<cpu, mshadow_op::tanh, mshadow_op::tanh_grad, DType>();
        break;
      case activation::kSoftReLU:
        op = new ActivationOp<cpu, mshadow_op::softrelu, mshadow_op::softrelu_grad, DType>();
        break;
      default:
        LOG(FATAL) << "unknown activation type";
    }
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *ActivationProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], (*in_shape)[0]);
}

DMLC_REGISTER_PARAMETER(ActivationParam);

MXNET_REGISTER_OP_PROPERTY(Activation, ActivationProp)
.describe(R"code(Applies an activation function element-wise to the input.

The following activation functions are supported:

- `relu`: Rectified Linear Unit, :math:`y = max(x, 0)`
- `sigmoid`: :math:`y = \frac{1}{1 + exp(-x)}`
- `tanh`: Hyperbolic tangent, :math:`y = \frac{exp(x) - exp(-x)}{exp(x) + exp(-x)}`
- `softrelu`: Soft ReLU, or SoftPlus, :math:`y = log(1 + exp(x))`

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input array to activation function.")
.add_arguments(ActivationParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

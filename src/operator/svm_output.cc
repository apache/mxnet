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
 * \file svm_output.cc
 * \brief
 * \author Jonas Amaro
*/
#include "./svm_output-inl.h"
#include "./mshadow_op.h"

namespace mshadow {
  template<typename DType>
  inline void L1_SVM(const DType & margin,
                     const DType & reg_coef,
                     Tensor<cpu, 2, DType> dst,
                     const Tensor<cpu, 1, DType> & label,
                     const Tensor<cpu, 2, DType> & src) {
    for (index_t y = 0; y < dst.size(0); y++) {
      const index_t k = static_cast<int>(label[y]);
      for (index_t x = 0; x < dst.size(1); x++) {
        if (x == k) {
          dst[y][k] = -DType(margin > src[y][k]) * reg_coef;
        } else {
          dst[y][x] = DType(margin > -src[y][x]) * reg_coef;
        }
      }
    }
  }


  template<typename DType>
  inline void L2_SVM(const DType & margin,
                     const DType & reg_coef,
                     Tensor<cpu, 2, DType> dst,
                     const Tensor<cpu, 1, DType> & label,
                     const Tensor<cpu, 2, DType> & src) {
    for (index_t y = 0; y < dst.size(0); y++) {
      const index_t k = static_cast<int>(label[y]);
      for (index_t x = 0; x < dst.size(1); x++) {
        if (x == k) {
          dst[y][k] = margin > src[y][k] ?  2*(margin - src[y][k]) : DType(0.0f);
          dst[y][k] *= -reg_coef;
        } else {
          dst[y][x] = margin > -src[y][x] ? (-2)*(margin + src[y][x]) : DType(0.0f);
          dst[y][x] *= -reg_coef;
        }
      }
    }
  }
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(SVMOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SVMOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SVMOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SVMOutputParam);

MXNET_REGISTER_OP_PROPERTY(SVMOutput, SVMOutputProp)
.describe(R"code(Computes support vector machine based transformation of the input.

This tutorial demonstrates using SVM as output layer for classification instead of softmax:
https://github.com/dmlc/mxnet/tree/master/example/svm_mnist.

)code")
.add_argument("data", "NDArray-or-Symbol", "Input data for SVM transformation.")
.add_argument("label", "NDArray-or-Symbol", "Class label for the input data.")
.add_arguments(SVMOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

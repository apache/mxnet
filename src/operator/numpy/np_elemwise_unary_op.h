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
 *  Copyright (c) 2019 by Contributors
 * \file np_elementwise_unary_op.h
 * \brief Function definition of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_H_

#include <vector>
#include "../tensor/elemwise_unary_op.h"

namespace mxnet {
namespace op {

template<int req>
struct exp2_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* in_data1, 
                                  const DType* in_data2, const DType* out_grad) {
    DType grad = in_grad[i] * in_data2[i] * in_data1[i] / DType(2);
    KERNEL_ASSIGN(in_grad[i], req, grad);
  }
};

template<typename xpu>
void Exp2Backward(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mshadow;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data1 = inputs[1];
  const TBlob& in_data2 = inputs[2];
  const TBlob& in_grad = outputs[0];
  using namespace mxnet_op;
  MSHADOW_REAL_TYPE_SWITCH(in_data1.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
        Kernel<exp2_backward<req_type>, xpu>::Launch(
            s, in_grad.Size(), in_grad.dptr<DType>(), in_data1.dptr<DType>(),
            in_data2.dptr<DType>(), out_grad.dptr<DType>());
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_ELEMWISE_UNARY_OP_H_

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
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./fully_connected-inl.h"
namespace mxnet {
namespace op {

template<>
void FullyConnectedCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t in_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), in_expected);
  CHECK_EQ(outputs.size(), 1U);
  int dtype = inputs[0].type_flag_;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FullyConnectedOp<gpu, DType>::get_op(param).Forward(ctx, inputs,
        req, outputs);
  });
}

template<>
void FullyConnectedGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                    const OpContext& ctx,
                                    const std::vector<TBlob>& inputs,
                                    const std::vector<OpReqType>& req,
                                    const std::vector<TBlob>& outputs) {
  const FullyConnectedParam& param = nnvm::get<FullyConnectedParam>(attrs.parsed);
  uint32_t out_expected = param.no_bias ? 2 : 3;
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), out_expected);
  CHECK_EQ(req.size(), out_expected);

  std::vector<TBlob> out_grad{inputs[0]};
  std::vector<TBlob> in_data(inputs.begin() + 1, inputs.end());
  int dtype = inputs[0].type_flag_;

  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    FullyConnectedOp<gpu, DType>::get_op(param).Backward(ctx, out_grad, in_data,
        req, outputs);
  });
}

NNVM_REGISTER_OP(FullyConnected)
.set_attr<FCompute>("FCompute<gpu>", FullyConnectedCompute<gpu>);

NNVM_REGISTER_OP(_backward_FullyConnected)
.set_attr<FCompute>("FCompute<gpu>", FullyConnectedGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

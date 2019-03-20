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
 * \file rnn.cu
 * \brief
 * \author Sebastian Bodenstein
*/

#include "./rnn-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_rnn-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

namespace mxnet {
namespace op {

template<typename xpu>
void RNNStatefulCompute(const OpStatePtr& state,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  int dtype = inputs[rnn_enum::kData].type_flag_;
  #if  MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      CuDNNRNNOp<DType>& op = state.get_state<CuDNNRNNOp<DType>>();
      op.Forward(ctx, inputs, req, outputs);
    });
  #else
    LOG(FATAL) << "RNN on GPU is only available for cuDNN at the moment.";
  #endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
}

template<typename xpu>
void RNNStatefulGradCompute(const OpStatePtr& state,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs) {
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> out_data{inputs[3]};
  std::vector<TBlob> out_grad{inputs[4]};
  const std::vector<TBlob> &in_grad = outputs;
  int dtype = inputs[rnn_enum::kData].type_flag_;
  #if  MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      CuDNNRNNOp<DType>& op = state.get_state<CuDNNRNNOp<DType>>();
      const RNNParam& param = op.param_;
      int index = 5;
      if (param.state_outputs) {
        out_data.push_back(inputs[index++]);
        out_grad.push_back(inputs[index++]);
      }

      if (param.mode == rnn_enum::kLstm) {
        in_data.push_back(inputs[index++]);
          if (param.state_outputs) {
          out_data.push_back(inputs[index++]);
          out_grad.push_back(inputs[index]);
        }
      }
      op.Backward(ctx, out_grad, in_data, out_data, req, in_grad);
    });
  #else
    LOG(FATAL) << "RNN on GPU is only available for cuDNN at the moment.";
  #endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
}

NNVM_REGISTER_OP(RNN)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", RNNStatefulCompute<gpu>);

NNVM_REGISTER_OP(_backward_RNN)
.set_attr<FStatefulCompute>("FStatefulCompute<gpu>", RNNStatefulGradCompute<gpu>);
}  // namespace op
}  // namespace mxnet

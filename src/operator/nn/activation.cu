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
 * \file activation.cu
 * \brief
 * \author Bing Xu
*/
#include "./activation-inl.h"
#include "../mshadow_op.h"
#if MXNET_USE_CUDNN == 1
#include "./cudnn/cudnn_activation-inl.h"
#endif

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1

template<typename DType>
static CuDNNActivationOp<DType> &get_cudnn_op(const ActivationParam& param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local CuDNNActivationOp<DType> cudnn_op;
#else
  static MX_THREAD_LOCAL CuDNNActivationOp<DType> cudnn_op;
#endif
  cudnn_op.Init(param);
  return cudnn_op;
}

template<>
void ActivationCompute<gpu>(const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>& inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  const int act_type = param.act_type;

  // SoftReLU, SoftSign, Log_Sigmoid and Mish are not supported by CUDNN yet
  if (act_type == activation::kSoftReLU) {
    ActivationForward<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>(ctx,
      inputs[0], req[0], outputs[0]);
  } else if (act_type == activation::kSoftSign) {
    ActivationForward<gpu, mshadow_op::softsign, mshadow_op::softsign_grad>(ctx,
      inputs[0], req[0], outputs[0]);
  } else if (act_type == activation::kLogSigmoid) {
    ActivationForward<gpu, mshadow_op::log_sigmoid, mshadow_op::log_sigmoid_grad>(ctx,
      inputs[0], req[0], outputs[0]);
  } else if (act_type == activation::kMish) {
    ActivationForward<gpu, mshadow_op::mish, mshadow_op::mish_grad>(ctx,
      inputs[0], req[0], outputs[0]);
  } else {
    MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
      get_cudnn_op<DType>(param).Forward(ctx, inputs[0], req[0], outputs[0]);
    });
  }
}

template<>
void ActivationGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                                const OpContext& ctx,
                                const std::vector<TBlob>& inputs,
                                const std::vector<OpReqType>& req,
                                const std::vector<TBlob>& outputs) {
  const ActivationParam& param = nnvm::get<ActivationParam>(attrs.parsed);
  const int act_type = param.act_type;
  CHECK_EQ(inputs.size(), activation::GradNumInputs(act_type));
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  bool do_memory_opt = dmlc::GetEnv("MXNET_MEMORY_OPT", 0);

  // SoftReLU, SoftSign, Log_Sigmoid and Mish not supported by CUDNN yet
  if (act_type == activation::kSoftReLU) {
    ActivationBackward<gpu, mshadow_op::softrelu, mshadow_op::softrelu_grad>(
      ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
  } else if (act_type == activation::kLogSigmoid) {
    ActivationBackward<gpu, mshadow_op::log_sigmoid, mshadow_op::log_sigmoid_grad>(
      ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
  } else if (act_type == activation::kMish) {
    ActivationBackward<gpu, mshadow_op::mish, mshadow_op::mish_grad>(
      ctx, inputs.at(0), inputs.at(2), req[0], outputs[0]);
  } else if (act_type == activation::kSoftSign) {
    if (do_memory_opt) {
      ActivationBackward<gpu, mshadow_op::softsign, mshadow_op::softsign_grad>(
        ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
    } else {
      ActivationBackward<gpu, mshadow_op::softsign, mshadow_op::softsign_grad>(
        ctx, inputs.at(0), inputs.at(2), req[0], outputs[0]);
    }
  } else if (act_type == activation::kReLU) {
    if (do_memory_opt) {
      ActivationBackward<gpu, mshadow_op::relu, mshadow_op::relu_grad>(
        ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
    } else {
      MSHADOW_REAL_TYPE_SWITCH(inputs.at(0).type_flag_, DType, {
        // XXX: for y = relu(x), y is passed as "in_data" to Backward()
        get_cudnn_op<DType>(param).Backward(ctx, inputs.at(0), inputs.at(1),
                                            inputs.at(1), req[0], outputs[0]);
      });
    }
  } else {
    if (do_memory_opt) {
      if (act_type == activation::kTanh) {
        ActivationBackward<gpu, mshadow_op::tanh, mshadow_op::tanh_grad>(
          ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
      } else if (act_type == activation::kSigmoid) {
        ActivationBackward<gpu, mshadow_op::sigmoid, mshadow_op::sigmoid_grad>(
          ctx, inputs.at(0), inputs.at(1), req[0], outputs[0]);
      } else {
        LOG(FATAL) << "unknown activation type";
      }
    } else {
      MSHADOW_REAL_TYPE_SWITCH(inputs.at(0).type_flag_, DType, {
        get_cudnn_op<DType>(param).Backward(ctx, inputs.at(0), inputs.at(2),
                                            inputs.at(1), req[0], outputs[0]);
      });
    }  // if (do_memory_opt)
  }
}
#endif

NNVM_REGISTER_OP(Activation)
.set_attr<FCompute>("FCompute<gpu>", ActivationCompute<gpu>);

NNVM_REGISTER_OP(_backward_Activation)
.set_attr<FCompute>("FCompute<gpu>", ActivationGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet

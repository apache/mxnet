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
 * \author Shu Zhang
*/
/*
#include "./rnn-inl.h"
#include <algorithm>
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
#include "./cudnn_rnn-inl.h"
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
template<typename DType>
static CuDNNRNNOp<DType> &GetCuDNNRNNOp(const RNNParam &param) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<RNNSignature,
                                       std::shared_ptr<CuDNNRNNOp<DType> >,
                                       OpHash> ops;

#else
  static MX_THREAD_LOCAL std::unordered_map<RNNSignature,
                                       std::shared_ptr<CuDNNRNNOp<DType> >,
                                       OpHash> ops;
#endif
  RNNSignature key(param);
  auto it = ops.find(key);
  if (it == ops.end()) {
    std::shared_ptr<CuDNNRNNOp<DType>> op(new CuDNNRNNOp<DType>(param));
    auto ins_ret = ops.insert(std::pair<RNNSignature, std::shared_ptr<CuDNNRNNOp<DType>>>(
                              key, op));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return *it->second;
}
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR

template<>
void RNNCompute<gpu>(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(inputs[rnn_enum::kData].type_flag_, DType, {
    GetCuDNNRNNOp<DType>(param).Forward(ctx, inputs, req, outputs);
  });
#else
  LOG(FATAL) << "RNN is only available for cuDNN at the moment.";
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
}


template<>
void RNNGradCompute<gpu>(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  const RNNParam& param = nnvm::get<RNNParam>(attrs.parsed);
  std::vector<TBlob> in_data(inputs.begin(), inputs.begin() + 3);
  std::vector<TBlob> out_data{inputs[3]};
  std::vector<TBlob> out_grad{inputs[4]};

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
  const std::vector<TBlob> &in_grad = outputs;
#if MXNET_USE_CUDNN == 1 && CUDNN_MAJOR >= 5
  MSHADOW_REAL_TYPE_SWITCH(inputs[0].type_flag_, DType, {
    GetCuDNNRNNOp<DType>(param).Backward(ctx, out_grad, in_data, out_data, req, in_grad);
  });
#else
  LOG(FATAL) << "RNN is only available for cuDNN at the moment.";
#endif  // MXNET_USE_CUDNN && CUDNN_MAJOR
}
NNVM_REGISTER_OP(RNN)
.set_attr<FCompute>("FCompute<gpu>", RNNCompute<gpu>);

NNVM_REGISTER_OP(_backward_RNN)
.set_attr<FCompute>("FCompute<gpu>", RNNGradCompute<gpu>);

}  // namespace op
}  // namespace mxnet
*/

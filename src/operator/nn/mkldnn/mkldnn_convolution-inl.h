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
 * \file mkldnn_convolution-inl.h
 * \brief
*/

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include "../convolution-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

mkldnn::convolution_forward::primitive_desc GetConvFwdImpl(
    const ConvolutionParam& param, bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output);

class MKLDNNConvForward {
  std::shared_ptr<mkldnn::convolution_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> weight;
  std::shared_ptr<mkldnn::memory> bias;
  std::shared_ptr<mkldnn::memory> out;

 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param, bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output);

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd;
  }
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

inline MKLDNNConvForward &GetConvFwd(
    const nnvm::NodeAttrs& attrs, bool is_train,
    const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNConvSignature, MKLDNNConvForward, OpHash> fwds;
  const ConvolutionParam& param = nnvm::get<ConvolutionParam>(attrs.parsed);
  MKLDNNConvSignature key(param);
  key.AddSign(is_train);
  // Here we can sign the conv op with NDArray because conv primitive will
  // decide the right layout for the, so we only need to get the shape and the
  // data type of the arrays.
  key.AddSign(data);
  key.AddSign(weights);
  key.AddSign(output);
  if (bias)
    key.AddSign(*bias);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNConvForward fwd(param, is_train, data, weights, bias, output);
    auto ins_ret = fwds.insert(
        std::pair<MKLDNNConvSignature, MKLDNNConvForward>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

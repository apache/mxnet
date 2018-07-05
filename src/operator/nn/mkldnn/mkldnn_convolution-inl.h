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
    const ConvolutionParam& param, const bool is_train, const NDArray &data,
    const NDArray &weights, const NDArray *bias, const NDArray &output);

class MKLDNNConvForward {
 public:
  mkldnn::convolution_forward::primitive_desc fwd_pd;

  MKLDNNConvForward(const ConvolutionParam& param, const bool is_train,
                    const NDArray &data, const NDArray &weights,
                    const NDArray *bias, const NDArray &output): fwd_pd(
                        GetConvFwdImpl(param, is_train, data, weights, bias, output)) {
  }

  void SetNewMem(const mkldnn::memory &data, const mkldnn::memory &weight,
                 const mkldnn::memory *bias, const mkldnn::memory &output);

  const mkldnn::convolution_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::convolution_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> weight_;
  std::shared_ptr<mkldnn::memory> bias_;
  std::shared_ptr<mkldnn::memory> out_;
};

typedef ParamOpSign<ConvolutionParam> MKLDNNConvSignature;

MKLDNNConvForward &GetConvFwd(const nnvm::NodeAttrs& attrs,
    const bool is_train, const NDArray &data, const NDArray &weights,
    const NDArray *bias, const NDArray &output);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_CONVOLUTION_INL_H_

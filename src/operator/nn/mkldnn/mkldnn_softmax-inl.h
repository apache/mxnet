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
 * \file mkldnn_softmax-inl.h
 */

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_SOFTMAX_INL_H_

#if MXNET_USE_ONEDNN == 1
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

#include "../softmax-inl.h"

namespace mxnet {
namespace op {

using softmax_fwd_t    = mkldnn::softmax_forward;
using softmax_fwd_pd_t = mkldnn::softmax_forward::primitive_desc;

using linear_t    = mkldnn::eltwise_forward;
using linear_pd_t = mkldnn::eltwise_forward::primitive_desc;

class MKLDNNSoftmaxFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& data, const NDArray& out);

    const NDArray& data;
    const NDArray& out;
  };

  static MKLDNNSoftmaxFwd& GetCached(const SoftmaxParam& param,
                                     const Tensors& tensors,
                                     const bool is_train);

  static softmax_fwd_pd_t GetSoftmaxFwdPd(const mkldnn::memory& input_mem,
                                          const int axis,
                                          const bool is_train);

  static linear_pd_t GetTemperaturePd(const mkldnn::memory& input_mem, const float temperature);

  MKLDNNSoftmaxFwd(const SoftmaxParam& param, const Tensors& tensors, const bool is_train);
  void Execute(const Tensors& tensors) const;

 private:
  std::shared_ptr<softmax_fwd_pd_t> softmax_pd;
  std::shared_ptr<softmax_fwd_t> softmax_fwd;
  std::shared_ptr<linear_pd_t> temperature_pd;
  std::shared_ptr<linear_t> temperature_fwd;
};

MKLDNNSoftmaxFwd::Tensors::Tensors(const NDArray& data, const NDArray& output)
    : data(data), out(output){};

MKLDNNSoftmaxFwd::MKLDNNSoftmaxFwd(const SoftmaxParam& param,
                                   const Tensors& tensors,
                                   const bool is_train) {
  float temperature = param.temperature.has_value() ? param.temperature.value() : 1.0f;
  int axis          = CheckAxis(param.axis, tensors.data.shape().ndim());
  auto input_mem    = tensors.data.GetMKLDNNData();
  softmax_pd      = std::make_shared<softmax_fwd_pd_t>(GetSoftmaxFwdPd(*input_mem, axis, is_train));
  softmax_fwd     = std::make_shared<softmax_fwd_t>(*softmax_pd);
  temperature_pd  = std::make_shared<linear_pd_t>(GetTemperaturePd(*input_mem, temperature));
  temperature_fwd = std::make_shared<linear_t>(*temperature_pd);
}
}  // namespace op
}  // namespace mxnet
#endif
#endif
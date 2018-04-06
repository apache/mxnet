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
 * \file mkldnn_quantized_pooling-inl.h
 * \brief
*/
#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_POOLING_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_POOLING_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <utility>
#include <vector>
#include <mkldnn.hpp>
#include "../pooling-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNQuantizedPoolingFwd {
 public:
  MKLDNNQuantizedPoolingFwd(const mxnet::NDArray &input,
                            const mxnet::NDArray &output,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int padding_t, const int padding_b,
                            const int padding_l, const int padding_r,
                            const mkldnn::algorithm alg_kind) :
                            alg_kind_(alg_kind),
                            fwd_(nullptr), data_(nullptr), out_(nullptr) {
    Init(input, output,
         kernel_h, kernel_w, stride_h, stride_w,
         padding_t, padding_b, padding_l, padding_r);
  }

  ~MKLDNNQuantizedPoolingFwd() {}
  void SetDataHandle(const mxnet::NDArray &data,
                     const mxnet::NDArray &output);
  void Execute();

 private:
  mkldnn::algorithm alg_kind_;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::pooling_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;

 private:
  void Init(const mxnet::NDArray &input,
            const mxnet::NDArray &output,
            const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w,
            const int padding_t, const int padding_b,
            const int padding_l, const int padding_r);
};

inline bool SupportMKLDNNQuantizedPooling(const PoolingParam &param) {
  return param.kernel.ndim() == 2 &&
         (param.pool_type == pool_enum::kMaxPooling ||
          param.pool_type == pool_enum::kAvgPooling);
}

typedef ParamOpSign<PoolingParam> MKLDNNPoolingSignature;
void MKLDNNQuantizedPoolingForward(const nnvm::NodeAttrs& attrs,
                                   const OpContext &ctx,
                                   const std::vector<NDArray> &in_data,
                                   const std::vector<OpReqType> &req,
                                   const std::vector<NDArray> &out_data);
}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_POOLING_INL_H_

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
 * \file dnnl_quantized_rnn-inl.h
 * \brief Common functions for quantized recurrent neural network
 * \author Zixuan Wei
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RNN_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RNN_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <vector>
#include "operator/nn/dnnl/dnnl_rnn-inl.h"
#include "operator/rnn-inl.h"
#include "operator/quantization/quantized_rnn-inl.h"

namespace mxnet {
namespace op {

class DNNLQuantizedRnnOp {
 public:
  explicit DNNLQuantizedRnnOp(const nnvm::NodeAttrs& attrs,
                              const int seq_len,
                              const int batch_size,
                              const int input_size)
      : initialized_(false),
        weights_ver_(0),
        rnn_attr_(new dnnl::primitive_attr),
        full_param_(DNNLRnnFullParamParser(attrs, seq_len, batch_size, input_size)) {}

  void Forward(const OpContext& op_ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  bool initialized_;
  size_t weights_ver_;
  shared_dnnl_attr_t rnn_attr_;
  DNNLRnnFullParam full_param_;
  DNNLRnnMemMgr mgr_;
  std::vector<DNNLRnnForward> fwd_inf_vec_;  // forward inference layers

  // Used to store the intermediate results of multi-layer
  std::vector<dnnl::memory*> dst_;
  // According to
  // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html, the
  // non-symmetric quantization is assumed by LSTM primitive. Namely, the
  // formula is:
  //                    data_f32 = (data_u8 - shift) / scale
  float cached_data_shift_{0.0};
  float cached_data_scale_{0.0};
  void Init(const OpContext& ctx,
            const std::vector<NDArray>& inputs,
            const std::vector<OpReqType>& req,
            const std::vector<NDArray>& outputs);
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_DNNL_DNNL_QUANTIZED_RNN_INL_H_

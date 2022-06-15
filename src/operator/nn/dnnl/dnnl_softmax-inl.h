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
 * \file dnnl_softmax-inl.h
 * Naming convention:
 *                 ________
 *                |Softmax|
 *  data  ------->|  FWD  |---> out
 *                |_______|
 *                 ________
 *                |Softmax|<--- out
 *  data_grad <---|  BWD  |
 *                |_______|<--- out_grad
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_SOFTMAX_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_SOFTMAX_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <vector>

#include "dnnl_base-inl.h"

#include "operator/nn/softmax-inl.h"

namespace mxnet {
namespace op {

using softmax_fwd_t    = dnnl::softmax_forward;
using softmax_fwd_pd_t = dnnl::softmax_forward::primitive_desc;

using softmax_bwd_t    = dnnl::softmax_backward;
using softmax_bwd_pd_t = dnnl::softmax_backward::primitive_desc;

using linear_t    = dnnl::eltwise_forward;
using linear_pd_t = dnnl::eltwise_forward::primitive_desc;

class DNNLSoftmaxFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& data, const NDArray& out);

    const NDArray& data;
    const NDArray& out;
  };

  static DNNLSoftmaxFwd& GetCached(const SoftmaxParam& param,
                                   const Tensors& tensors,
                                   const bool is_train);

  static softmax_fwd_pd_t GetSoftmaxFwdPd(const dnnl::memory& input_mem,
                                          const int axis,
                                          const bool is_train);

  static linear_pd_t GetTemperaturePd(const dnnl::memory& input_mem, const float temperature);

  DNNLSoftmaxFwd(const SoftmaxParam& param, const Tensors& tensors, const bool is_train);
  void Execute(const Tensors& tensors) const;

  std::shared_ptr<softmax_fwd_pd_t> softmax_pd;
  std::shared_ptr<softmax_fwd_t> softmax_fwd;
  std::shared_ptr<linear_pd_t> temperature_pd;
  std::shared_ptr<linear_t> temperature_fwd;
};

class DNNLSoftmaxBwd {
 public:
  struct Tensors {
    Tensors(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs);
    const NDArray& out_grad;
    const NDArray& out;
    const NDArray& data_grad;
  };
  static DNNLSoftmaxBwd& GetCached(const SoftmaxParam& param, const Tensors& tensors);

  static softmax_bwd_pd_t GetSoftmaxBwdPd(const dnnl::memory& out_grad_mem,
                                          const dnnl::memory& out_mem,
                                          const int axis,
                                          const softmax_fwd_pd_t& hint_fwd_pd);

  DNNLSoftmaxBwd(const SoftmaxParam& param, const Tensors& tensors);
  void Execute(const Tensors& tensors, const std::vector<OpReqType>& req) const;

 private:
  std::shared_ptr<softmax_bwd_pd_t> softmax_bwd_pd;
  std::shared_ptr<softmax_bwd_t> softmax_bwd;
  std::shared_ptr<linear_pd_t> temperature_pd;
  std::shared_ptr<linear_t> temperature_fwd;
};

void DNNLSoftmaxForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& in_data,
                        const OpReqType& req,
                        const NDArray& out_data);

void DNNLSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& out_data);

void DNNLLogSoftmaxForward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const NDArray& in_data,
                           const OpReqType& req,
                           const NDArray& out_data);

void DNNLLogSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& in_data,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& out_data);

void DNNLMaskedSoftmaxForward(const nnvm::NodeAttrs& attrs,
                              const OpContext& ctx,
                              const std::vector<NDArray>& inputs,
                              const std::vector<OpReqType>& req,
                              const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_SOFTMAX_INL_H_

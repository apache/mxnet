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
 * \file dnnl_layer_norm-inl.h
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */
#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_LAYER_NORM_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_LAYER_NORM_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <utility>
#include <vector>

#include "operator/nn/layer_norm-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

using layernorm_fwd_t    = dnnl::layer_normalization_forward;
using layernorm_fwd_pd_t = dnnl::layer_normalization_forward::primitive_desc;

using layernorm_bwd_t    = dnnl::layer_normalization_backward;
using layernorm_bwd_pd_t = dnnl::layer_normalization_backward::primitive_desc;

typedef ParamOpSign<LayerNormParam> LayerNormSignature;

class DNNLLayerNormFwd {
 public:
  static DNNLLayerNormFwd& GetCached(const LayerNormParam& param,
                                     const OpContext& ctx,
                                     const NDArray& data);

  DNNLLayerNormFwd(const LayerNormParam& param, const NDArray& data);

  static std::shared_ptr<layernorm_fwd_pd_t> CreatePrimitiveDesc(const LayerNormParam& param,
                                                                 const dnnl::memory::desc& src_md);

  void Execute(const LayerNormParam& param,
               const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const OpReqType& req,
               const std::vector<NDArray>& outputs) const;

  ~DNNLLayerNormFwd() {}

 private:
  std::shared_ptr<layernorm_fwd_t> fwd;
  std::shared_ptr<layernorm_fwd_pd_t> fwd_pd;
};

class DNNLLayerNormBwd {
 public:
  static DNNLLayerNormBwd& GetCached(const LayerNormParam& param,
                                     const std::vector<NDArray>& inputs);

  DNNLLayerNormBwd(const LayerNormParam& param,
                   const std::vector<NDArray>& inputs,
                   const dnnl::memory::desc& data_md,
                   const dnnl::memory::desc& diff_md);

  static std::shared_ptr<layernorm_bwd_pd_t> CreatePrimitiveDesc(
      const LayerNormParam& param,
      const dnnl::memory::desc& data_md,
      const dnnl::memory::desc& diff_md,
      const layernorm_fwd_pd_t& layernorm_fwd_pd);

  void Execute(const std::vector<NDArray>& inputs,
               const std::vector<NDArray>& outputs,
               const std::vector<OpReqType>& req) const;

  ~DNNLLayerNormBwd() {}

 private:
  std::shared_ptr<layernorm_bwd_t> bwd;
  std::shared_ptr<layernorm_fwd_pd_t> fwd_pd;
  std::shared_ptr<layernorm_bwd_pd_t> bwd_pd;
};

void DNNLLayerNormForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs);

void DNNLLayerNormBackward(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx,
                           const std::vector<NDArray>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_LAYER_NORM_INL_H__

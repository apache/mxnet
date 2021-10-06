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
 * \file dnnl_pooling-inl.h
 * \brief
 */
#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_POOLING_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_POOLING_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <dnnl.hpp>
#include <utility>

#include "../pooling-inl.h"
#include "./dnnl_base-inl.h"

namespace mxnet {
namespace op {

class DNNLPoolingFwd {
 public:
  DNNLPoolingFwd(const mxnet::NDArray& input,
                 const mxnet::NDArray& output,
                 const dnnl::memory::dims& kernel,
                 const dnnl::memory::dims& strides,
                 const dnnl::memory::dims& pad_l,
                 const dnnl::memory::dims& pad_r,
                 const dnnl::algorithm alg_kind,
                 const bool with_workspace,
                 const bool is_train)
      : with_workspace_(with_workspace), fwd_(nullptr) {
    Init(input, output, kernel, strides, pad_l, pad_r, is_train, alg_kind);
  }

  ~DNNLPoolingFwd() {}
  void Execute(const NDArray& in_data,
               const OpReqType req,
               const NDArray& out_data,
               const NDArray* workspace);

 private:
  bool with_workspace_;

  std::shared_ptr<dnnl::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<dnnl::pooling_forward> fwd_;

 private:
  void Init(const mxnet::NDArray& input,
            const mxnet::NDArray& output,
            const dnnl::memory::dims& kernel,
            const dnnl::memory::dims& strides,
            const dnnl::memory::dims& pad_l,
            const dnnl::memory::dims& pad_r,
            const bool is_train,
            const dnnl::algorithm alg_kind);
};

class DNNLPoolingBwd {
  std::shared_ptr<const dnnl::pooling_backward> bwd;
  bool with_workspace;

 public:
  const dnnl::pooling_backward::primitive_desc pd;

  DNNLPoolingBwd(const dnnl::pooling_backward::primitive_desc& pdesc, bool with_ws);

  ~DNNLPoolingBwd() {}
  const dnnl::pooling_backward& GetBwd();
  const dnnl::pooling_backward::primitive_desc& GetPd();
};

inline int GetPaddingSizeFull(dim_t x, int padl, int padr, int k, int s) {
  if ((x + padl + padr - k) % s != 0) {
    return (padr + s - ((x + padl + padr - k) % s));
  } else {
    return padr;
  }
}

inline bool SupportDNNLPooling(const PoolingParam& param) {
  return (param.kernel.ndim() == 1 || param.kernel.ndim() == 2 || param.kernel.ndim() == 3) &&
         (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) &&
         (!param.layout.has_value() ||
          (param.layout.value() == mshadow::kNCW || param.layout.value() == mshadow::kNCHW ||
           param.layout.value() == mshadow::kNCDHW));
}

inline bool SupportDNNLPooling(const PoolingParam& param, const NDArray& input) {
  const auto dshape = input.shape();
  const auto ndim   = dshape.ndim();
  const auto dtype  = input.dtype();

  if (!(SupportStorageDNNL(input.storage_type()) && (ndim == 3 || ndim == 4 || ndim == 5) &&
        (dtype == mshadow::kFloat32 || dtype == mshadow::kBfloat16)))
    return false;

  if (!SupportDNNLPooling(param))
    return false;

  if (param.pooling_convention == pool_enum::kValid) {
    return true;
  } else {
    if (param.pool_type == pool_enum::kAvgPooling) {
      // dnnl works differently when padding is asymmetric, so let's skip this case.
      bool is_symmetric = true;
      switch (ndim) {
        case 5:
          is_symmetric =
              is_symmetric &&
              (param.pad[2] ==
               GetPaddingSizeFull(
                   dshape[4], param.pad[2], param.pad[2], param.kernel[2], param.stride[2]));
        case 4:
          is_symmetric =
              is_symmetric &&
              (param.pad[1] ==
               GetPaddingSizeFull(
                   dshape[3], param.pad[1], param.pad[1], param.kernel[1], param.stride[1]));
        case 3:
          is_symmetric =
              is_symmetric &&
              (param.pad[0] ==
               GetPaddingSizeFull(
                   dshape[2], param.pad[0], param.pad[0], param.kernel[0], param.stride[0]));
      }
      return is_symmetric;
    }
    return param.pool_type == pool_enum::kMaxPooling;
  }
}

inline bool DNNLRequireWorkspace(const PoolingParam& param) {
  return param.pool_type != pool_enum::kAvgPooling;
}

typedef ParamOpSign<PoolingParam> DNNLPoolingSignature;
void DNNLPoolingCompute(const OpContext& ctx,
                        const PoolingParam& param,
                        const NDArray& in_data,
                        const OpReqType req,
                        const NDArray& out_data,
                        const NDArray* workspace);

void DNNLPoolingGradCompute(const OpContext& ctx,
                            const PoolingParam& param,
                            const NDArray& out_grad,
                            const NDArray& in_data,
                            const NDArray* workspace,
                            const OpReqType req,
                            const NDArray& in_grad);
DNNLPoolingFwd& GetPoolingFwd(const PoolingParam& param,
                              const bool is_train,
                              const NDArray& data,
                              const NDArray& output);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_POOLING_INL_H_
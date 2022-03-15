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
 * \file mkldnn_pooling-inl.h
 * \brief
 */
#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <mkldnn.hpp>
#include <utility>
#include <vector>

#include "../pooling-inl.h"
#include "./mkldnn_base-inl.h"

#define DIV_ROUND_UP(a, b) (((a) + (b - 1)) / (b))

namespace mxnet {
namespace op {

class MKLDNNPoolingFwd {
 public:
  MKLDNNPoolingFwd(const mxnet::NDArray& input,
                   const mxnet::NDArray& output,
                   const mkldnn::memory::dims& kernel,
                   const mkldnn::memory::dims& strides,
                   const mkldnn::memory::dims& pad_l,
                   const mkldnn::memory::dims& pad_r,
                   const mkldnn::algorithm alg_kind,
                   const bool with_workspace,
                   const bool is_train)
      : with_workspace_(with_workspace), fwd_(nullptr) {
    Init(input, output, kernel, strides, pad_l, pad_r, is_train, alg_kind);
  }

  ~MKLDNNPoolingFwd() {}
  void Execute(const NDArray& in_data,
               const OpReqType req,
               const NDArray& out_data,
               const NDArray* workspace,
               const bool use_adaptive_pooling);

 private:
  bool with_workspace_;

  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::pooling_forward> fwd_;

 private:
  void Init(const mxnet::NDArray& input,
            const mxnet::NDArray& output,
            const mkldnn::memory::dims& kernel,
            const mkldnn::memory::dims& strides,
            const mkldnn::memory::dims& pad_l,
            const mkldnn::memory::dims& pad_r,
            const bool is_train,
            const mkldnn::algorithm alg_kind);
};

class MKLDNNPoolingBwd {
  std::shared_ptr<const mkldnn::pooling_backward> bwd;
  bool with_workspace;

 public:
  const mkldnn::pooling_backward::primitive_desc pd;

  MKLDNNPoolingBwd(const mkldnn::pooling_backward::primitive_desc& pdesc, bool with_ws);

  ~MKLDNNPoolingBwd() {}
  const mkldnn::pooling_backward& GetBwd();
  const mkldnn::pooling_backward::primitive_desc& GetPd();
};

inline int ComputeStrides(const int inner, const int outer) {
  return ((inner << 1) / outer) - (inner / outer);
}

inline int ComputeKernel(const int inner, const int outer) {
  return DIV_ROUND_UP((inner << 1) / outer, 1) - (inner / outer);
}

inline int ComputePadding(const int inner, const int outer, const int strides,
                          const int kernel) {
  return (strides * (outer - 1) + kernel - inner) / 2;
}

template <typename T = mkldnn::memory::dims>
void UseAdaptivePaddingKernel(T* kernel,
                              T* strides,
                              T* pad_l,
                              T* pad_r,
                              const mxnet::TShape& input_shape,
                              const mxnet::TShape& output_shape) {
  const int IH = input_shape[2];
  const int IW = input_shape[3];
  const int OH = output_shape[2];
  const int OW = output_shape[3];

  strides->at(0) = ComputeStrides(IH, OH);
  strides->at(1) = ComputeStrides(IW, OW);
  kernel->at(0) = ComputeKernel(IH, OH);
  kernel->at(1) = ComputeKernel(IW, OW);
  pad_l->at(0) = ComputePadding(IH, OH, strides->at(0), kernel->at(0));
  pad_l->at(1) = ComputePadding(IW, OW, strides->at(1), kernel->at(1));
}

inline int GetPaddingSizeFull(dim_t x, int padl, int padr, int k, int s) {
  if ((x + padl + padr - k) % s != 0) {
    return (padr + s - ((x + padl + padr - k) % s));
  } else {
    return padr;
  }
}

inline bool SupportMKLDNNPooling(const PoolingParam& param) {
  return (param.kernel.ndim() == 1 || param.kernel.ndim() == 2 || param.kernel.ndim() == 3) &&
         (param.pool_type == pool_enum::kMaxPooling || param.pool_type == pool_enum::kAvgPooling) &&
         (!param.layout.has_value() ||
          (param.layout.value() == mshadow::kNCW || param.layout.value() == mshadow::kNCHW ||
           param.layout.value() == mshadow::kNCDHW));
}

inline bool SupportMKLDNNPooling(const PoolingParam& param, const NDArray& input) {
  const auto dshape = input.shape();
  const auto ndim   = dshape.ndim();
  const auto dtype  = input.dtype();

  if (!(SupportStorageMKLDNN(input.storage_type()) && (ndim == 3 || ndim == 4 || ndim == 5) &&
        (dtype == mshadow::kFloat32 || dtype == mshadow::kBfloat16)))
    return false;

  if (!SupportMKLDNNPooling(param))
    return false;

  if (param.pooling_convention == pool_enum::kValid) {
    return true;
  } else {
    if (param.pool_type == pool_enum::kAvgPooling) {
      // mkldnn works differently when padding is asymmetric, so let's skip this
      // case.
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

inline bool MKLDNNRequireWorkspace(const PoolingParam& param) {
  return param.pool_type != pool_enum::kAvgPooling && !param.IsAdaptivePooling();
}

typedef ParamOpSign<PoolingParam> MKLDNNPoolingSignature;

MKLDNNPoolingFwd& GetPoolingFwd(const PoolingParam& param,
                                const bool is_train,
                                const NDArray& data,
                                const NDArray& output,
                                const bool use_adaptive_pooling);

MKLDNNPoolingBwd& GetPoolingBwd(const PoolingParam& param,
                                const NDArray& in_data,
                                const NDArray& in_grad,
                                const NDArray& out_grad,
                                const bool use_adaptive_pooling);

void MKLDNNPoolingCompute(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& in_data,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& out_data);

void MKLDNNPoolingGradCompute(const nnvm::NodeAttrs &attrs,
                              const OpContext &ctx,
                              const std::vector<NDArray> &inputs,
                              const std::vector<OpReqType> &req,
                              const std::vector<NDArray> &outputs);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_

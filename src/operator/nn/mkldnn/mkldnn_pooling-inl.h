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

#include <utility>
#include <mkldnn.hpp>
#include "../pooling-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

class MKLDNNPoolingFwd {
 public:
  MKLDNNPoolingFwd(const mxnet::NDArray &input,
                   const mxnet::NDArray &output,
                   const int kernel_h, const int kernel_w,
                   const int stride_h, const int stride_w,
                   const int padding_t, const int padding_b,
                   const int padding_l, const int padding_r,
                   const mkldnn::algorithm alg_kind,
                   const bool with_workspace, const bool is_train) :
                   is_train_(is_train),
                   with_workspace_(with_workspace),
                   alg_kind_(alg_kind),
                   fwd_(nullptr), data_(nullptr), out_(nullptr), workspace_(nullptr) {
    Init(input, output,
         kernel_h, kernel_w, stride_h, stride_w,
         padding_t, padding_b, padding_l, padding_r);
  }

  ~MKLDNNPoolingFwd() {}
  void SetNewMem(const NDArray& in_data,
                 const NDArray& out_data,
                 const OpReqType& req,
                 const mxnet::NDArray *workspace = nullptr);
  void Execute(const NDArray& out_data);

 private:
  bool is_train_;
  bool with_workspace_;
  mkldnn::algorithm alg_kind_;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd_;
  std::shared_ptr<mkldnn::pooling_forward> fwd_;
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory> workspace_;
  mkldnn_output_t output_mem_t_;

 private:
  void Init(const mxnet::NDArray &input,
            const mxnet::NDArray &output,
            const int kernel_h, const int kernel_w,
            const int stride_h, const int stride_w,
            const int padding_t, const int padding_b,
            const int padding_l, const int padding_r);
};

inline bool SupportMKLDNNPooling(const PoolingParam &param) {
  return param.kernel.ndim() == 2 &&
         (param.pool_type == pool_enum::kMaxPooling ||
          param.pool_type == pool_enum::kAvgPooling);
}

inline bool SupportMKLDNNPooling(const PoolingParam &param,
                                 const TShape &dshape) {
  bool ret = SupportMKLDNNPooling(param);
  if (!ret)
    return false;

  if (param.pooling_convention == pool_enum::kValid)
    return true;
  else
    return false;

// need to support pooling convention full
// https://issues.apache.org/jira/browse/MXNET-33
#if 0
  if (((dshape[2] + 2 * param.pad[0] - param.kernel[0]) % param.stride[0] == 0) &&
      ((dshape[3] + 2 * param.pad[1] - param.kernel[1]) % param.stride[1] == 0))
    return true;
  else
    return false;
#endif
}

inline bool MKLDNNRequireWorkspace(const PoolingParam &param) {
  return param.pool_type != pool_enum::kAvgPooling;
}

typedef ParamOpSign<PoolingParam> MKLDNNPoolingSignature;
void MKLDNNPoolingCompute(const OpContext &ctx, const PoolingParam &param,
                          const NDArray &in_data, const OpReqType req,
                          const NDArray &out_data, const NDArray *workspace);

void MKLDNNPoolingGradCompute(const OpContext &ctx, const PoolingParam &param,
                              const NDArray &out_grad, const NDArray &in_data,
                              const NDArray *workspace, const OpReqType req,
                              const NDArray &in_grad);
MKLDNNPoolingFwd &GetPoolingFwd(const PoolingParam &param,
                                const bool is_train,
                                const NDArray &data,
                                const NDArray &output);
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_

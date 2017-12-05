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
 * \file mkldnn_pooling.cc
 * \brief
*/

#if MXNET_USE_MKLDNN == 1
#include <mkldnn.hpp>
#include "../pooling-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {

static inline bool SupportMKLDNNPooling(const PoolingParam &param) {
  return param.kernel.ndim() == 2
      && (param.pool_type == pool_enum::kMaxPooling
          || param.pool_type == pool_enum::kAvgPooling);
}

static inline bool SupportMKLDNNPooling(const PoolingParam &param,
                                        const TShape &dshape) {
  auto ret = SupportMKLDNNPooling(param);
  if (!ret)
    return false;
  if (param.pooling_convention == pool_enum::kValid)
    return true;
  if ((dshape[2] + 2 * param.pad[0] - param.kernel[0]) % param.stride[0] == 0
      && (dshape[3] + 2 * param.pad[1] - param.kernel[1]) % param.stride[1] == 0)
    return true;
  else
    return false;
}

static inline algorithm GetMKLDNNPoolAlgo(const PoolingParam &param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      return algorithm::pooling_avg;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
      return algorithm::pooling_max;
  }
}

inline static pooling_forward::primitive_desc GetPoolingFwd(
    const PoolingParam &param, bool is_train, const memory::desc &data_md,
    const memory::desc &out_md) {
  CHECK_EQ(param.kernel.ndim(), 2) << "Not Implemented";
  int kernel_h_, kernel_w_;
  if (param.global_pool) {
    kernel_h_ = data_md.data.dims[2];
    kernel_w_ = data_md.data.dims[3];
  } else {
    kernel_h_ = param.kernel[0];
    kernel_w_ = param.kernel[1];
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";

  auto pad_t_ = param.pad[0], pad_b_ = param.pad[0];
  auto pad_l_ = param.pad[1], pad_r_ = param.pad[1];
  auto stride_h_ = param.stride[0], stride_w_ = param.stride[1];

  auto engine = CpuEngine::Instance().get_engine();
  if (param.global_pool) {
    CHECK(pad_t_ == 0 && pad_l_ == 0 && stride_h_ == 1 && stride_w_ == 1)
        << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_t_ != 0 || pad_l_ != 0) {
    CHECK(param.pool_type == pool_enum::kAvgPooling ||
          param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l_, kernel_w_);
    CHECK_LT(pad_t_, kernel_h_);
  }
  auto alg = GetMKLDNNPoolAlgo(param);
  auto kind = prop_kind::forward_scoring;
  if (is_train && alg != algorithm::pooling_avg) {
    kind = prop_kind::forward_training;
  }
  pooling_forward::desc poolingFwd_desc(
      kind, alg, data_md, out_md, {(int)stride_h_, (int)stride_w_},
      {kernel_h_, kernel_w_}, {(int)pad_t_, (int)pad_l_}, {(int)pad_b_, (int)pad_r_},
      padding_kind::zero);
  return mkldnn::pooling_forward::primitive_desc(poolingFwd_desc, engine);
}

inline bool MKLDNNRequireWorkspace(const PoolingParam &param) {
  return param.pool_type != pool_enum::kAvgPooling;
}

void MKLDNNPooling_Forward(const OpContext &ctx, const PoolingParam &param,
                           const NDArray &in_data, const OpReqType &req,
                           const NDArray &out_data, const NDArray *workspace) {
  std::shared_ptr<const mkldnn::memory> input_mem = in_data.GetMKLDNNData();
  auto data_mpd = input_mem->get_primitive_desc();
  auto data_md = data_mpd.desc();

  memory::dims dims = {data_md.data.dims[0], data_md.data.dims[1],
                       (int)out_data.shape()[2], (int)out_data.shape()[3]};
  memory::desc out_md({dims},
                      static_cast<memory::data_type>(data_md.data.data_type),
                      static_cast<memory::format>(data_md.data.format));

  auto pdesc = GetPoolingFwd(param, ctx.is_train, data_md, out_md);

  std::shared_ptr<const mkldnn::memory> output_memory =
      const_cast<NDArray &>(out_data).CreateMKLDNNData(
          pdesc.dst_primitive_desc());
  std::shared_ptr<const mkldnn::memory> workspace_mem;

  if (ctx.is_train && MKLDNNRequireWorkspace(param)) {
    CHECK(workspace != nullptr);
    workspace_mem = workspace->GetMKLDNNData();
    MKLDNNStream::Instance().RegisterPrim(
        pooling_forward(pdesc, *input_mem, *output_memory, *workspace_mem));
  } else {
    MKLDNNStream::Instance().RegisterPrim(
        pooling_forward(pdesc, *input_mem, *output_memory));
  }
  MKLDNNStream::Instance().Submit();
}

void MKLDNNPooling_Backward(const OpContext &ctx, const PoolingParam &param,
                            const NDArray &out_grad, const NDArray &in_data,
                            const NDArray *workspace, const OpReqType &req,
                            const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }

  std::shared_ptr<const mkldnn::memory> diff_dst_mem = out_grad.GetMKLDNNData();
  std::shared_ptr<const mkldnn::memory> input_mem = in_data.GetMKLDNNData();
  mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  memory::dims dims = {data_md.data.dims[0], data_md.data.dims[1],
                       (int)out_grad.shape()[2], (int)out_grad.shape()[3]};
  memory::desc out_md({dims},
                      static_cast<memory::data_type>(data_md.data.data_type),
                      static_cast<memory::format>(data_md.data.format));
  auto pdesc_fwd = GetPoolingFwd(param, ctx.is_train, data_md, out_md);

  mkldnn::memory::desc diff_md = diff_dst_mem->get_primitive_desc().desc();
  memory::dims dims1 = {diff_md.data.dims[0], diff_md.data.dims[1],
                        (int)in_grad.shape()[2], (int)in_grad.shape()[3]};
  memory::desc diff_in_md(
      {dims1}, static_cast<memory::data_type>(diff_md.data.data_type),
      static_cast<memory::format>(diff_md.data.format));
  auto cpu_engine = data_mpd.get_engine();

  auto alg = GetMKLDNNPoolAlgo(param);

  int kernel_h_, kernel_w_;
  if (param.global_pool) {
    kernel_h_ = data_md.data.dims[2];
    kernel_w_ = data_md.data.dims[3];
  } else {
    kernel_h_ = param.kernel[0];
    kernel_w_ = param.kernel[1];
  }
  pooling_backward::desc desc(
      alg, diff_in_md, diff_md, {(int)param.stride[0], (int)param.stride[1]},
      {kernel_h_, kernel_w_}, {(int)param.pad[0], (int)param.pad[1]},
      {(int)param.pad[0], (int)param.pad[1]}, padding_kind::zero);
  pooling_backward::primitive_desc pdesc(desc, cpu_engine, pdesc_fwd);

  auto diff_src_mem =
      CreateMKLDNNMem(in_grad, pdesc.diff_src_primitive_desc(), req);
  std::shared_ptr<const mkldnn::memory> workspace_mem;

  if (MKLDNNRequireWorkspace(param)) {
    CHECK(workspace != nullptr);
    workspace_mem = workspace->GetMKLDNNData();
    MKLDNNStream::Instance().RegisterPrim(
        pooling_backward(pdesc, *diff_dst_mem, primitive::at(*workspace_mem),
                         *diff_src_mem.second));
  } else {
    MKLDNNStream::Instance().RegisterPrim(
        pooling_backward(pdesc, *diff_dst_mem, *diff_src_mem.second));
  }
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Instance().Submit();
}
}
}
#endif  // MXNET_USE_MKLDNN == 1

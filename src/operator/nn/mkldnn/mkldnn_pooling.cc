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
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1

#include "./mkldnn_pooling-inl.h"

namespace mxnet {
namespace op {

void MKLDNNPoolingFwd::Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
                            const int kernel_h,  const int kernel_w,
                            const int stride_h,  const int stride_w,
                            const int padding_t, const int padding_b,
                            const int padding_l, const int padding_r) {
  // mkldnn::memory::desc
  auto src_md = input.GetMKLDNNData()->get_primitive_desc().desc();
  mkldnn::memory::dims dims = {src_md.data.dims[0],
                               src_md.data.dims[1],
                               static_cast<int>(output.shape()[2]),
                               static_cast<int>(output.shape()[3])};
  auto dst_md = mkldnn::memory::desc({dims},
                                     static_cast<mkldnn::memory::data_type>(src_md.data.data_type),
                                     static_cast<mkldnn::memory::format>(src_md.data.format));
  const mkldnn::engine engine = CpuEngine::Get()->get_engine();
  const mkldnn::algorithm alg_kind = this->alg_kind_;
  if (alg_kind != mkldnn::algorithm::pooling_max &&
      alg_kind != mkldnn::algorithm::pooling_avg &&
      alg_kind != mkldnn::algorithm::pooling_avg_include_padding &&
      alg_kind != mkldnn::algorithm::pooling_avg_exclude_padding) {
    LOG(FATAL) << "MKLDNN Pooling: algorithm is not supported";
  }

  mkldnn::prop_kind prop = mkldnn::prop_kind::forward_scoring;
  if (this->is_train_ && alg_kind != mkldnn::algorithm::pooling_avg) {
    prop = mkldnn::prop_kind::forward_training;
  }
  if (this->is_train_ && prop == mkldnn::prop_kind::forward_scoring) {
    LOG(INFO) << "MKLDNN Pooling: training with prop_kind is forward_scoring";
  }

  const mkldnn::memory::dims strides = {stride_h,  stride_w  };
  const mkldnn::memory::dims pad_l   = {padding_t, padding_l };
  const mkldnn::memory::dims pad_r   = {padding_b, padding_r };
  const mkldnn::memory::dims kernel  = {kernel_h,  kernel_w  };
  // mkldnn::pooling_forward::desc
  const auto fwd_desc = mkldnn::pooling_forward::desc(prop, alg_kind, src_md, dst_md,
                                                      strides, kernel, pad_l, pad_r,
                                                      mkldnn::padding_kind::zero);
  this->fwd_pd_.reset(new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
  this->data_.reset(new mkldnn::memory(input.GetMKLDNNData()->get_primitive_desc()));
  this->out_.reset(new mkldnn::memory(this->fwd_pd_->dst_primitive_desc()));
  if (this->with_workspace_) {
    this->workspace_.reset(new mkldnn::memory(this->fwd_pd_->workspace_primitive_desc()));
    this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_),
                                                 mkldnn::primitive::at(*(this->data_)),
                                                 *(this->out_),
                                                 *(this->workspace_)));
  } else {
    this->fwd_.reset(new mkldnn::pooling_forward(*(this->fwd_pd_),
                                                 mkldnn::primitive::at(*(this->data_)),
                                                 *(this->out_)));
  }
  return;
}

void MKLDNNPoolingFwd::SetNewMem(const NDArray& in_data,
                                 const NDArray& out_data,
                                 const OpReqType& req,
                                 const mxnet::NDArray *workspace) {
  auto input_mem = in_data.GetMKLDNNData();
  output_mem_t_ = CreateMKLDNNMem(out_data, fwd_pd_->dst_primitive_desc(), req);
  // mkldnn::memory
  this->data_->set_data_handle(input_mem->get_data_handle());
  this->out_->set_data_handle(output_mem_t_.second->get_data_handle());
  if (this->with_workspace_ && workspace == nullptr) {
    LOG(FATAL) << "MKLDNN Pooling: incorrect workspace input";
  }

  if (this->with_workspace_) {
    // mkldnn::memory
    auto ws_mem = workspace->GetMKLDNNData();
    this->workspace_->set_data_handle(ws_mem->get_data_handle());
  }
}

void MKLDNNPoolingFwd::Execute(const NDArray& out_data) {
  if (this->fwd_) {
    MKLDNNStream::Get()->RegisterPrim(*(this->fwd_));
    CommitOutput(out_data, this->output_mem_t_);
    MKLDNNStream::Get()->Submit();
  } else {
    LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
  }
}

mkldnn::algorithm GetMKLDNNPoolAlgo(const PoolingParam &param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return mkldnn::algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      if (param.count_include_pad.has_value() && !param.count_include_pad.value()) {
        return mkldnn::algorithm::pooling_avg_exclude_padding;
      } else {
        return mkldnn::algorithm::pooling_avg_include_padding;
      }
      break;
    default:
      LOG(FATAL) << "MKLDNN Pooling: Unknown pooling method.";
      return mkldnn::algorithm::pooling_max;
  }
}

static inline int GetPaddingSizeFull(int x, int padl, int padr, int k, int s) {
  if ((x + padl + padr - k) % s != 0) {
    return (padr + s - ((x + padl + padr - k) % s));
  } else {
    return padr;
  }
}

mkldnn::pooling_forward::primitive_desc GetPoolingFwdPdesc(
    const PoolingParam &param, const bool is_train, const memory::desc &data_md,
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

  int pad_t_ = param.pad[0], pad_b_ = param.pad[0];
  int pad_l_ = param.pad[1], pad_r_ = param.pad[1];
  int stride_h_ = param.stride[0], stride_w_ = param.stride[1];

  if (param.pooling_convention == pool_enum::kFull) {
    pad_b_ = GetPaddingSizeFull(data_md.data.dims[2], pad_t_, pad_b_, kernel_h_, stride_h_);
    pad_r_ = GetPaddingSizeFull(data_md.data.dims[3], pad_l_, pad_r_, kernel_w_, stride_w_);
  }

  const mkldnn::engine engine = CpuEngine::Get()->get_engine();
  if (param.global_pool) {
    pad_t_ = pad_b_ = pad_l_ = pad_r_ = 0;
    stride_h_ = stride_w_ = 1;
  }

  if (pad_t_ != 0 || pad_l_ != 0) {
    CHECK(param.pool_type == pool_enum::kAvgPooling ||
          param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l_, kernel_w_);
    CHECK_LT(pad_t_, kernel_h_);
  }

  const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
  mkldnn::prop_kind kind = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg != algorithm::pooling_avg) {
    kind = mkldnn::prop_kind::forward_training;
  }

  const pooling_forward::desc poolingFwd_desc(kind, alg, data_md, out_md,
                                              {static_cast<int>(stride_h_),
                                               static_cast<int>(stride_w_)},
                                              {kernel_h_, kernel_w_},
                                              {static_cast<int>(pad_t_),
                                               static_cast<int>(pad_l_)},
                                              {static_cast<int>(pad_b_),
                                               static_cast<int>(pad_r_)},
                                              padding_kind::zero);
  return mkldnn::pooling_forward::primitive_desc(poolingFwd_desc, engine);
}

MKLDNNPoolingFwd &GetPoolingFwd(const PoolingParam &param,
                                const bool is_train,
                                const NDArray &data,
                                const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNPoolingFwd,
                                         OpHash> pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNPoolingSignature,
                                            MKLDNNPoolingFwd,
                                            OpHash> pooling_fwds;
#endif

  bool with_workspace = is_train && MKLDNNRequireWorkspace(param);
  MKLDNNPoolingSignature key(param);
  key.AddSign(is_train);
  key.AddSign(with_workspace);
  key.AddSign(data);
  key.AddSign(output);

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    CHECK_EQ(param.kernel.ndim(), 2) << "Not Implemented";
    auto data_md = data.GetMKLDNNData()->get_primitive_desc().desc();
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

    int pad_t_ = param.pad[0], pad_b_ = param.pad[0];
    int pad_l_ = param.pad[1], pad_r_ = param.pad[1];
    int stride_h_ = param.stride[0], stride_w_ = param.stride[1];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_b_ = GetPaddingSizeFull(data_md.data.dims[2], pad_t_, pad_b_, kernel_h_, stride_h_);
      pad_r_ = GetPaddingSizeFull(data_md.data.dims[3], pad_l_, pad_r_, kernel_w_, stride_w_);
    }

    if (param.global_pool) {
      pad_t_ = pad_b_ = pad_l_ = pad_r_ = 0;
      stride_h_ = stride_w_ = 1;
    }

    if (pad_t_ != 0 || pad_l_ != 0) {
      CHECK(param.pool_type == pool_enum::kAvgPooling ||
            param.pool_type == pool_enum::kMaxPooling)
            << "Padding implemented only for average and max pooling.";
      CHECK_LT(pad_l_, kernel_w_);
      CHECK_LT(pad_t_, kernel_h_);
    }

    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
    MKLDNNPoolingFwd fwd(data, output, kernel_h_, kernel_w_, stride_h_, stride_w_,
                         pad_t_, pad_b_, pad_l_, pad_r_, alg, with_workspace, is_train);
    it = AddToCache(&pooling_fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNPoolingCompute(const OpContext &ctx, const PoolingParam &param,
                          const NDArray &in_data, const OpReqType req,
                          const NDArray &out_data, const NDArray *workspace) {
  auto &fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
  fwd.SetNewMem(in_data, out_data, req, workspace);
  fwd.Execute(out_data);
}

MKLDNNPoolingBwd::MKLDNNPoolingBwd(
    const pooling_backward::primitive_desc &pdesc, bool with_ws)
    : with_workspace(with_ws), pd(pdesc) {}

void MKLDNNPoolingBwd::SetNewMem(const mxnet::NDArray *workspace,
                                 const mxnet::NDArray &out_grad,
                                 const mkldnn::memory *diff_src_mem) {
  if (bwd == nullptr) {
    diff_dst.reset(
        new mkldnn::memory(out_grad.GetMKLDNNData()->get_primitive_desc(),
                           out_grad.GetMKLDNNData()->get_data_handle()));
    diff_src.reset(new mkldnn::memory(pd.diff_src_primitive_desc(),
                                      diff_src_mem->get_data_handle()));
    if (with_workspace) {
      CHECK(workspace != nullptr);
      ws.reset(
          new mkldnn::memory(workspace->GetMKLDNNData()->get_primitive_desc(),
                             workspace->GetMKLDNNData()->get_data_handle()));
      bwd.reset(
          new pooling_backward(pd, *diff_dst, primitive::at(*ws), *diff_src));
    } else {
      bwd.reset(new pooling_backward(pd, *diff_dst, *diff_src));
    }
  } else {
    diff_dst->set_data_handle(out_grad.GetMKLDNNData()->get_data_handle());
    diff_src->set_data_handle(diff_src_mem->get_data_handle());
    if (with_workspace) {
      CHECK(workspace != nullptr);
      ws->set_data_handle(workspace->GetMKLDNNData()->get_data_handle());
    }
  }
}

const mkldnn::pooling_backward &MKLDNNPoolingBwd::GetBwd() {
  return *this->bwd;
}

MKLDNNPoolingBwd &GetPoolingBwd(const PoolingParam &param,
                                const NDArray &in_data,
                                const NDArray &in_grad,
                                const NDArray &out_grad) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local
      std::unordered_map<MKLDNNPoolingSignature,
                         MKLDNNPoolingBwd, OpHash> pooling_bwds;
#else
  static MX_THREAD_LOCAL
      std::unordered_map<MKLDNNPoolingSignature,
                         MKLDNNPoolingBwd, OpHash> pooling_bwds;
#endif

  bool with_workspace = MKLDNNRequireWorkspace(param);
  MKLDNNPoolingSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);

  auto it = pooling_bwds.find(key);
  if (it == pooling_bwds.end()) {
    auto diff_dst_mem = out_grad.GetMKLDNNData();
    auto input_mem = in_data.GetMKLDNNData();
    mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
    const mkldnn::memory::desc data_md = data_mpd.desc();
    const memory::dims dims = {data_md.data.dims[0], data_md.data.dims[1],
                               static_cast<int>(out_grad.shape()[2]),
                               static_cast<int>(out_grad.shape()[3])};
    const memory::desc out_md(
        {dims}, static_cast<memory::data_type>(data_md.data.data_type),
        static_cast<memory::format>(data_md.data.format));
    auto fwd_pd = GetPoolingFwdPdesc(param, true, data_md, out_md);

    const mkldnn::memory::desc diff_md =
        diff_dst_mem->get_primitive_desc().desc();
    const memory::dims dims1 = {diff_md.data.dims[0], diff_md.data.dims[1],
                                static_cast<int>(in_grad.shape()[2]),
                                static_cast<int>(in_grad.shape()[3])};
    const memory::desc diff_in_md(
        {dims1}, static_cast<memory::data_type>(diff_md.data.data_type),
        static_cast<memory::format>(diff_md.data.format));
    const mkldnn::engine cpu_engine = data_mpd.get_engine();
    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);

    int kernel_h_, kernel_w_;
    if (param.global_pool) {
      kernel_h_ = data_md.data.dims[2];
      kernel_w_ = data_md.data.dims[3];
    } else {
      kernel_h_ = param.kernel[0];
      kernel_w_ = param.kernel[1];
    }

    int pad_t_ = param.pad[0], pad_b_ = param.pad[0];
    int pad_l_ = param.pad[1], pad_r_ = param.pad[1];
    int stride_h_ = param.stride[0], stride_w_ = param.stride[1];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_b_ = GetPaddingSizeFull(data_md.data.dims[2], pad_t_, pad_b_, kernel_h_, stride_h_);
      pad_r_ = GetPaddingSizeFull(data_md.data.dims[3], pad_l_, pad_r_, kernel_w_, stride_w_);
    }

    if (param.global_pool) {
      pad_t_ = pad_b_ = pad_l_ = pad_r_ = 0;
      stride_h_ = stride_w_ = 1;
    }

    const pooling_backward::desc desc(
        alg, diff_in_md, diff_md, {stride_h_, stride_w_},
        {kernel_h_, kernel_w_}, {pad_t_, pad_l_}, {pad_b_, pad_r_},
        mkldnn::padding_kind::zero);
    const auto pdesc = pooling_backward::primitive_desc(desc, cpu_engine, fwd_pd);
    MKLDNNPoolingBwd bwd(pdesc, with_workspace);
    it = AddToCache(&pooling_bwds, key, bwd);
  }
  return it->second;
}

void MKLDNNPoolingGradCompute(const OpContext &ctx, const PoolingParam &param,
                              const NDArray &out_grad, const NDArray &in_data,
                              const NDArray *workspace, const OpReqType req,
                              const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }
  TmpMemMgr::Get()->Init(ctx.requested[0]);

  auto &bwd = GetPoolingBwd(param, in_data, in_grad, out_grad);
  auto diff_src_mem =
      CreateMKLDNNMem(in_grad, bwd.pd.diff_src_primitive_desc(), req);

  bwd.SetNewMem(workspace, out_grad, diff_src_mem.second);
  MKLDNNStream::Get()->RegisterPrim(bwd.GetBwd());
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Get()->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1

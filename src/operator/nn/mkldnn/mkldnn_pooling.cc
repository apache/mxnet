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
                            const mkldnn::memory::dims &kernel,
                            const mkldnn::memory::dims &strides,
                            const mkldnn::memory::dims &pad_l,
                            const mkldnn::memory::dims &pad_r,
                            const bool is_train, const mkldnn::algorithm alg_kind) {
  const auto src_md = input.GetMKLDNNData()->get_desc();
  const auto dst_md = GetMemDesc(output);
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

void InitPoolingPrimitiveParams(const PoolingParam &param,
                                const mkldnn::memory::desc &data_md,
                                const mkldnn::memory::dims &new_kernel,
                                const mkldnn::memory::dims &new_strides,
                                const mkldnn::memory::dims &new_pad_l,
                                const mkldnn::memory::dims &new_pad_r) {
  const int kernel_ndims = param.kernel.ndim();
  mkldnn::memory::dims& kernel = const_cast<mkldnn::memory::dims&>(new_kernel);
  mkldnn::memory::dims& strides = const_cast<mkldnn::memory::dims&>(new_strides);
  mkldnn::memory::dims& pad_l = const_cast<mkldnn::memory::dims&>(new_pad_l);
  mkldnn::memory::dims& pad_r = const_cast<mkldnn::memory::dims&>(new_pad_r);
  if (kernel_ndims == 1) {
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.stride.ndim(), 1);
    kernel[0] = param.kernel[0];
    pad_l[0] = param.pad[0];
    pad_r[0] = param.pad[0];
    strides[0] = param.stride[0];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] =
        GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
    }

    if (param.global_pool) {
      kernel[0] = data_md.data.dims[2];
      strides[0] = 1;
      pad_l[0] = pad_r[0] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
  } else if (kernel_ndims == 2) {
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.stride.ndim(), 2);
    kernel[0] = param.kernel[0];
    kernel[1] = param.kernel[1];
    pad_l[0] = param.pad[0];
    pad_l[1] = param.pad[1];
    pad_r[0] = param.pad[0];
    pad_r[1] = param.pad[1];
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] =
        GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
      pad_r[1] =
        GetPaddingSizeFull(data_md.data.dims[3], pad_l[1], pad_r[1], kernel[1], strides[1]);
    }

    if (param.global_pool) {
      kernel[0] = data_md.data.dims[2];
      kernel[1] = data_md.data.dims[3];
      strides[0] = strides[1] = 1;
      pad_l[0] = pad_l[1] = pad_r[0] = pad_r[1] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[1], 0) << "Filter dimensions cannot be zero.";
  } else {
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.stride.ndim(), 3);
    kernel[0] = param.kernel[0];
    kernel[1] = param.kernel[1];
    kernel[2] = param.kernel[2];
    pad_l[0] = param.pad[0];
    pad_l[1] = param.pad[1];
    pad_l[2] = param.pad[2];
    pad_r[0] = param.pad[0];
    pad_r[1] = param.pad[1];
    pad_r[2] = param.pad[2];
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];
    strides[2] = param.stride[2];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] =
        GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
      pad_r[1] =
        GetPaddingSizeFull(data_md.data.dims[3], pad_l[1], pad_r[1], kernel[1], strides[1]);
      pad_r[2] =
        GetPaddingSizeFull(data_md.data.dims[4], pad_l[2], pad_r[2], kernel[2], strides[2]);
    }

    if (param.global_pool) {
      kernel[0] = data_md.data.dims[2];
      kernel[1] = data_md.data.dims[3];
      kernel[2] = data_md.data.dims[4];
      strides[0] = strides[1] = strides[2] = 1;
      pad_l[0] = pad_l[1] = pad_l[2] = pad_r[0] = pad_r[1] = pad_r[2] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[1], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[2], 0) << "Filter dimensions cannot be zero.";
  }

  if (pad_l[0] != 0 || (kernel_ndims == 2 && pad_l[1] != 0) ||
     (kernel_ndims == 3 && pad_l[2] != 0)) {
    CHECK(param.pool_type == pool_enum::kAvgPooling ||
          param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l[0], kernel[0]);
    if (kernel_ndims > 1)
      CHECK_LT(pad_l[1], kernel[1]);
    if (kernel_ndims > 2)
      CHECK_LT(pad_l[2], kernel[2]);
  }
}

mkldnn::pooling_forward::primitive_desc GetPoolingFwdPdesc(
    const PoolingParam &param, const bool is_train, const mkldnn::memory::desc &data_md,
    const mkldnn::memory::desc &out_md) {
  CHECK(param.kernel.ndim() == 1 || param.kernel.ndim() == 2 || param.kernel.ndim() == 3)
        << "Not Implemented";

  const int kernel_ndims = param.kernel.ndim();
  mkldnn::memory::dims kernel(kernel_ndims);
  mkldnn::memory::dims strides(kernel_ndims);
  mkldnn::memory::dims pad_l(kernel_ndims);
  mkldnn::memory::dims pad_r(kernel_ndims);

  InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

  const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
  mkldnn::prop_kind kind = mkldnn::prop_kind::forward_scoring;
  if (is_train && alg != algorithm::pooling_avg) {
    kind = mkldnn::prop_kind::forward_training;
  }

  const mkldnn::pooling_forward::desc poolingFwd_desc(kind, alg, data_md, out_md, strides,
                                                      kernel, pad_l, pad_r);
  return mkldnn::pooling_forward::primitive_desc(poolingFwd_desc, CpuEngine::Get()->get_engine());
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
    CHECK(param.kernel.ndim() == 1 || param.kernel.ndim() == 2 || param.kernel.ndim() == 3)
          << "Not Implemented";
    auto data_md = data.GetMKLDNNData()->get_desc();

    const auto kernel_ndims = param.kernel.ndim();
    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    mkldnn::memory::dims pad_l(kernel_ndims);
    mkldnn::memory::dims pad_r(kernel_ndims);
    InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);
    MKLDNNPoolingFwd fwd(data, output, kernel, strides,
                         pad_l, pad_r, alg, with_workspace, is_train);
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
    const mkldnn::memory::desc data_md = input_mem->get_desc();
    const mkldnn::memory::desc out_md = GetMemDesc(out_grad);
    auto fwd_pd = GetPoolingFwdPdesc(param, true, data_md, out_md);
    const mkldnn::memory::desc diff_md = diff_dst_mem->get_desc();

    const mkldnn::memory::desc diff_in_md = GetMemDesc(in_grad);
    const mkldnn::engine cpu_engine = CpuEngine::Get()->get_engine();
    const mkldnn::algorithm alg = GetMKLDNNPoolAlgo(param);

    const int kernel_ndims = param.kernel.ndim();
    mkldnn::memory::dims kernel(kernel_ndims);
    mkldnn::memory::dims strides(kernel_ndims);
    mkldnn::memory::dims pad_l(kernel_ndims);
    mkldnn::memory::dims pad_r(kernel_ndims);

    InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);

    const mkldnn::pooling_backward::desc desc(
                alg, diff_in_md, diff_md, strides, kernel, pad_l, pad_r);
    const auto pdesc = mkldnn::pooling_backward::primitive_desc(desc, cpu_engine, fwd_pd);
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

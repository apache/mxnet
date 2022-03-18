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
 * \file dnnl_pooling.cc
 * \brief
 * \author Tao Lv
 */

#if MXNET_USE_ONEDNN == 1

#include "dnnl_pooling-inl.h"

namespace mxnet {
namespace op {

static inline dnnl::memory::data_type get_data_type(const dnnl::memory::desc& md) {
  return static_cast<dnnl::memory::data_type>(md.data_type());
}

void DNNLPoolingFwd::Init(const mxnet::NDArray& input,
                          const mxnet::NDArray& output,
                          const dnnl::memory::dims& kernel,
                          const dnnl::memory::dims& strides,
                          const dnnl::memory::dims& pad_l,
                          const dnnl::memory::dims& pad_r,
                          const bool is_train,
                          const dnnl::algorithm alg_kind) {
  const auto src_md         = input.GetDNNLData()->get_desc();
  const auto dst_md         = GetMemDesc(output);
  const dnnl::engine engine = CpuEngine::Get()->get_engine();
  if (alg_kind != dnnl::algorithm::pooling_max && alg_kind != dnnl::algorithm::pooling_avg &&
      alg_kind != dnnl::algorithm::pooling_avg_include_padding &&
      alg_kind != dnnl::algorithm::pooling_avg_exclude_padding) {
    LOG(FATAL) << "oneDNN Pooling: algorithm is not supported";
  }

  dnnl::prop_kind prop = dnnl::prop_kind::forward_scoring;
  if (is_train && alg_kind != dnnl::algorithm::pooling_avg) {
    prop = dnnl::prop_kind::forward_training;
  }
  if (is_train && prop == dnnl::prop_kind::forward_scoring) {
    LOG(INFO) << "oneDNN Pooling: training with prop_kind is forward_scoring";
  }

  const auto fwd_desc =
      dnnl::pooling_forward::desc(prop, alg_kind, src_md, dst_md, strides, kernel, pad_l, pad_r);
  this->fwd_pd_.reset(new dnnl::pooling_forward::primitive_desc(fwd_desc, engine));
  this->fwd_.reset(new dnnl::pooling_forward(*(this->fwd_pd_)));

  return;
}

void DNNLPoolingFwd::Execute(const NDArray& in_data,
                             const OpReqType req,
                             const NDArray& out_data,
                             const NDArray* workspace,
                             const bool use_adaptive_pooling) {
  NDArray in_buffer = in_data;
  if (in_data.IsView() && in_data.IsDNNLData())
    in_buffer = in_data.Reorder2Default();

  auto input_mem     = in_buffer.GetDNNLData();
  auto output_mem_t_ = CreateDNNLMem(out_data, this->fwd_pd_->dst_desc(), req);

  dnnl_args_map_t args = {
      {DNNL_ARG_SRC, *input_mem},
      {DNNL_ARG_DST, *(output_mem_t_.second)},
  };

  if (this->with_workspace_ && !use_adaptive_pooling) {
    auto engine = CpuEngine::Get()->get_engine();

    if (workspace == nullptr) {
      LOG(FATAL) << "oneDNN Pooling: incorrect workspace input";
    }

    auto ws = std::make_shared<dnnl::memory>(
        (*(this->fwd_pd_)).workspace_desc(), engine, workspace->GetDNNLData()->get_data_handle());
    args[DNNL_ARG_WORKSPACE] = *ws;
  }
  if (this->fwd_) {
    DNNLStream::Get()->RegisterPrimArgs(*(this->fwd_), args);
    CommitOutput(out_data, output_mem_t_);
    DNNLStream::Get()->Submit();
  } else {
    LOG(FATAL) << "oneDNN Pooling: forward primitive is nullptr";
  }
}

dnnl::algorithm GetDNNLPoolingAlgorithm(const PoolingParam& param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return dnnl::algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      if (param.count_include_pad.has_value() && !param.count_include_pad.value()) {
        return dnnl::algorithm::pooling_avg_exclude_padding;
      } else {
        return dnnl::algorithm::pooling_avg_include_padding;
      }
      break;
    default:
      LOG(FATAL) << "oneDNN Pooling: Unknown pooling method.";
      return dnnl::algorithm::pooling_max;
  }
}

void InitPoolingPrimitiveParams(const PoolingParam& param,
                                const dnnl::memory::desc& data_md,
                                const dnnl::memory::dims& new_kernel,
                                const dnnl::memory::dims& new_strides,
                                const dnnl::memory::dims& new_pad_l,
                                const dnnl::memory::dims& new_pad_r) {
  const int kernel_ndims      = param.kernel.ndim();
  dnnl::memory::dims& kernel  = const_cast<dnnl::memory::dims&>(new_kernel);
  dnnl::memory::dims& strides = const_cast<dnnl::memory::dims&>(new_strides);
  dnnl::memory::dims& pad_l   = const_cast<dnnl::memory::dims&>(new_pad_l);
  dnnl::memory::dims& pad_r   = const_cast<dnnl::memory::dims&>(new_pad_r);
  if (kernel_ndims == 1) {
    CHECK_GE(param.pad.ndim(), 1);
    CHECK_GE(param.stride.ndim(), 1);
    kernel[0]  = param.kernel[0];
    pad_l[0]   = param.pad[0];
    pad_r[0]   = param.pad[0];
    strides[0] = param.stride[0];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] =
          GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
    }

    if (param.global_pool) {
      kernel[0]  = data_md.data.dims[2];
      strides[0] = 1;
      pad_l[0] = pad_r[0] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
  } else if (kernel_ndims == 2) {
    CHECK_GE(param.pad.ndim(), 2);
    CHECK_GE(param.stride.ndim(), 2);
    kernel[0]  = param.kernel[0];
    kernel[1]  = param.kernel[1];
    pad_l[0]   = param.pad[0];
    pad_l[1]   = param.pad[1];
    pad_r[0]   = param.pad[0];
    pad_r[1]   = param.pad[1];
    strides[0] = param.stride[0];
    strides[1] = param.stride[1];

    if (param.pooling_convention == pool_enum::kFull) {
      pad_r[0] =
          GetPaddingSizeFull(data_md.data.dims[2], pad_l[0], pad_r[0], kernel[0], strides[0]);
      pad_r[1] =
          GetPaddingSizeFull(data_md.data.dims[3], pad_l[1], pad_r[1], kernel[1], strides[1]);
    }

    if (param.global_pool) {
      kernel[0]  = data_md.data.dims[2];
      kernel[1]  = data_md.data.dims[3];
      strides[0] = strides[1] = 1;
      pad_l[0] = pad_l[1] = pad_r[0] = pad_r[1] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[1], 0) << "Filter dimensions cannot be zero.";
  } else {
    CHECK_GE(param.pad.ndim(), 3);
    CHECK_GE(param.stride.ndim(), 3);
    kernel[0]  = param.kernel[0];
    kernel[1]  = param.kernel[1];
    kernel[2]  = param.kernel[2];
    pad_l[0]   = param.pad[0];
    pad_l[1]   = param.pad[1];
    pad_l[2]   = param.pad[2];
    pad_r[0]   = param.pad[0];
    pad_r[1]   = param.pad[1];
    pad_r[2]   = param.pad[2];
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
      kernel[0]  = data_md.data.dims[2];
      kernel[1]  = data_md.data.dims[3];
      kernel[2]  = data_md.data.dims[4];
      strides[0] = strides[1] = strides[2] = 1;
      pad_l[0] = pad_l[1] = pad_l[2] = pad_r[0] = pad_r[1] = pad_r[2] = 0;
    }

    CHECK_GT(kernel[0], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[1], 0) << "Filter dimensions cannot be zero.";
    CHECK_GT(kernel[2], 0) << "Filter dimensions cannot be zero.";
  }

  if (pad_l[0] != 0 || (kernel_ndims == 2 && pad_l[1] != 0) ||
      (kernel_ndims == 3 && pad_l[2] != 0)) {
    CHECK(param.pool_type == pool_enum::kAvgPooling || param.pool_type == pool_enum::kMaxPooling)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_l[0], kernel[0]);
    if (kernel_ndims > 1)
      CHECK_LT(pad_l[1], kernel[1]);
    if (kernel_ndims > 2)
      CHECK_LT(pad_l[2], kernel[2]);
  }
}

dnnl::pooling_forward::primitive_desc GetPoolingFwdPdesc(const PoolingParam& param,
                                                         const bool is_train,
                                                         const dnnl::memory::desc& data_md,
                                                         const dnnl::memory::desc& out_md,
                                                         const bool use_adaptive_pooling) {
  CHECK((param.kernel.ndim() >= 1 && param.kernel.ndim() <= 3) || use_adaptive_pooling)
      << "Not Implemented";

  const int kernel_ndims =
      use_adaptive_pooling ? mxnet::TShape(data_md.dims()).ndim() : param.kernel.ndim();
  dnnl::memory::dims kernel(kernel_ndims);
  dnnl::memory::dims strides(kernel_ndims);
  dnnl::memory::dims pad_l(kernel_ndims);
  dnnl::memory::dims pad_r(kernel_ndims);

  const mxnet::TShape input_shape  = mxnet::TShape(data_md.dims());
  const mxnet::TShape output_shape = mxnet::TShape(out_md.dims());

  if (use_adaptive_pooling) {
    UseAdaptivePaddingKernel(&kernel, &strides, &pad_l, &pad_r, input_shape, output_shape);
    dnnl::memory::validate_dims(kernel);
    dnnl::memory::validate_dims(strides);
    dnnl::memory::validate_dims(pad_l);
    dnnl::memory::validate_dims(pad_r);
  } else {
    InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);
  }

  const dnnl::algorithm alg = GetDNNLPoolingAlgorithm(param);
  dnnl::prop_kind kind      = dnnl::prop_kind::forward_scoring;
  if (is_train && alg != dnnl::algorithm::pooling_avg) {
    kind = dnnl::prop_kind::forward_training;
  }

  const dnnl::pooling_forward::desc poolingFwd_desc(
      kind, alg, data_md, out_md, strides, kernel, pad_l, pad_r);
  return dnnl::pooling_forward::primitive_desc(poolingFwd_desc, CpuEngine::Get()->get_engine());
}

DNNLPoolingFwd& GetPoolingFwd(const PoolingParam& param,
                              const bool is_train,
                              const NDArray& data,
                              const NDArray& output,
                              const bool use_adaptive_pooling) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLPoolingSignature, DNNLPoolingFwd, OpHash> pooling_fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLPoolingSignature, DNNLPoolingFwd, OpHash>
      pooling_fwds;
#endif

  const bool with_workspace = is_train && DNNLRequireWorkspace(param);
  DNNLPoolingSignature key(param);
  key.AddSign(is_train);
  key.AddSign(with_workspace);
  key.AddSign(data);
  key.AddSign(output);

  if (use_adaptive_pooling) {
    key.AddSign(use_adaptive_pooling);
  }

  auto it = pooling_fwds.find(key);
  if (it == pooling_fwds.end()) {
    CHECK(use_adaptive_pooling || (param.kernel.ndim() >= 1 && param.kernel.ndim() <= 3))
        << "Not Implemented";
    auto data_md = data.GetDNNLData()->get_desc();

    const auto kernel_ndims = use_adaptive_pooling ? data.shape().ndim() : param.kernel.ndim();
    dnnl::memory::dims kernel(kernel_ndims);
    dnnl::memory::dims strides(kernel_ndims);
    dnnl::memory::dims pad_l(kernel_ndims);
    dnnl::memory::dims pad_r(kernel_ndims);

    if (use_adaptive_pooling) {
      UseAdaptivePaddingKernel(&kernel, &strides, &pad_l, &pad_r, data.shape(), output.shape());
      dnnl::memory::validate_dims(kernel);
      dnnl::memory::validate_dims(strides);
      dnnl::memory::validate_dims(pad_l);
      dnnl::memory::validate_dims(pad_r);
    } else {
      InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);
    }

    const dnnl::algorithm alg =
        use_adaptive_pooling ? dnnl::algorithm::pooling_avg : GetDNNLPoolingAlgorithm(param);

    DNNLPoolingFwd fwd(data, output, kernel, strides, pad_l, pad_r, alg, with_workspace, is_train);
    it = AddToCache(&pooling_fwds, key, fwd);
  }
  return it->second;
}

DNNLPoolingBwd::DNNLPoolingBwd(const dnnl::pooling_backward::primitive_desc& pdesc, bool with_ws)
    : with_workspace(with_ws), pd(pdesc) {
  bwd = std::make_shared<dnnl::pooling_backward>(pd);
}

const dnnl::pooling_backward& DNNLPoolingBwd::GetBwd() {
  return *this->bwd;
}

DNNLPoolingBwd& GetPoolingBwd(const PoolingParam& param,
                              const NDArray& in_data,
                              const NDArray& in_grad,
                              const NDArray& out_grad,
                              const bool use_adaptive_pooling) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLPoolingSignature, DNNLPoolingBwd, OpHash> pooling_bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLPoolingSignature, DNNLPoolingBwd, OpHash>
      pooling_bwds;
#endif

  const bool with_workspace = DNNLRequireWorkspace(param);
  DNNLPoolingSignature key(param);
  key.AddSign(in_data);
  key.AddSign(in_grad);
  key.AddSign(out_grad);
  if (use_adaptive_pooling) {
    key.AddSign(use_adaptive_pooling);
  }

  auto it = pooling_bwds.find(key);
  if (it == pooling_bwds.end()) {
    auto input_mem = in_data.GetDNNLData();
    auto data_md   = input_mem->get_desc();

    auto dst_dims = dnnl::memory::dims(out_grad.shape().begin(), out_grad.shape().end());
    auto any      = dnnl::memory::format_tag::any;
    auto dst_md   = dnnl::memory::desc(dst_dims, get_data_type(data_md), any);

    // fwd hint
    auto fwd_pd = GetPoolingFwdPdesc(param, true, data_md, dst_md, use_adaptive_pooling);

    // create bwd desc
    auto diff_src_dims = dnnl::memory::dims(in_grad.shape().begin(), in_grad.shape().end());
    auto diff_src_md   = dnnl::memory::desc(diff_src_dims, get_data_type(data_md), any);
    auto cpu_engine    = CpuEngine::Get()->get_engine();
    auto alg = use_adaptive_pooling ? dnnl::algorithm::pooling_avg : GetDNNLPoolingAlgorithm(param);

    const int kernel_ndims = use_adaptive_pooling ? in_grad.shape().ndim() : param.kernel.ndim();
    dnnl::memory::dims kernel(kernel_ndims);
    dnnl::memory::dims strides(kernel_ndims);
    dnnl::memory::dims pad_l(kernel_ndims);
    dnnl::memory::dims pad_r(kernel_ndims);

    if (use_adaptive_pooling) {
      UseAdaptivePaddingKernel(
          &kernel, &strides, &pad_l, &pad_r, in_grad.shape(), out_grad.shape());
      dnnl::memory::validate_dims(kernel);
      dnnl::memory::validate_dims(strides);
      dnnl::memory::validate_dims(pad_l);
      dnnl::memory::validate_dims(pad_r);
    } else {
      InitPoolingPrimitiveParams(param, data_md, kernel, strides, pad_l, pad_r);
    }

    // use dst_md as diff_dst_md with any format
    auto bwd_desc =
        dnnl::pooling_backward::desc(alg, diff_src_md, dst_md, strides, kernel, pad_l, pad_r);
    auto pdesc = dnnl::pooling_backward::primitive_desc(bwd_desc, cpu_engine, fwd_pd);

    DNNLPoolingBwd bwd(pdesc, with_workspace);
    it = AddToCache(&pooling_bwds, key, bwd);
  }
  return it->second;
}

void DNNLPoolingGradCompute(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const std::vector<NDArray>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<NDArray>& outputs) {
  if (req[0] == kNullOp) {
    return;
  }

  const PoolingParam& param = nnvm::get<PoolingParam>(attrs.parsed);

  const NDArray& out_grad  = inputs[0];
  const NDArray* workspace = nullptr;
  const NDArray* in_data   = nullptr;
  if (DNNLRequireWorkspace(param)) {
    // The first two elements are the gradients of the outputs in forward.
    // The third is the input of forward.
    // The fourth and the fifth are the outputs of forward.
    CHECK_EQ(inputs.size(), 5U);
    in_data   = &inputs[2];
    workspace = &inputs[4];
  } else if (!param.IsAdaptivePooling()) {
    CHECK_EQ(inputs.size(), 3U);
    in_data = &inputs[1];
  } else {
    in_data = &inputs[0];
  }
  const NDArray& in_grad = outputs[0];

  TmpMemMgr::Get()->Init(ctx.requested[0]);

  auto& bwd = GetPoolingBwd(param, *in_data, in_grad, out_grad, param.IsAdaptivePooling());
  auto bwd_diff_dst_desc = bwd.pd.diff_dst_desc();
  auto diff_dst_mem      = out_grad.GetDNNLDataReorder(&bwd_diff_dst_desc);
  auto diff_src_mem      = CreateDNNLMem(in_grad, bwd.pd.diff_src_desc(), req[0]);
  dnnl_args_map_t args   = {
      {DNNL_ARG_DIFF_DST, *diff_dst_mem},
      {DNNL_ARG_DIFF_SRC, *diff_src_mem.second},
  };
  if (DNNLRequireWorkspace(param) && workspace != nullptr) {
    args[DNNL_ARG_WORKSPACE] = *(workspace->GetDNNLData());
  }

  DNNLStream::Get()->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(in_grad, diff_src_mem);
  DNNLStream::Get()->Submit();
}

void DNNLPoolingCompute(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<NDArray>& in_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<NDArray>& out_data) {
  const PoolingParam& param      = nnvm::get<PoolingParam>(attrs.parsed);
  const NDArray* workspace       = nullptr;
  const bool is_adaptive_pooling = param.IsAdaptivePooling();
  if (DNNLRequireWorkspace(param) && !is_adaptive_pooling) {
    CHECK_GT(out_data.size(), 1U);
    workspace = &out_data[1];
  }
  auto& fwd = GetPoolingFwd(param, ctx.is_train, in_data[0], out_data[0], is_adaptive_pooling);
  fwd.Execute(in_data[0], req[0], out_data[0], workspace, is_adaptive_pooling);
}

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1

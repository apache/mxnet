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
                   int kernel_h, int kernel_w,
                   int stride_h, int stride_w,
                   int padding_t, int padding_b, int padding_l, int padding_r,
                   mkldnn::algorithm alg_kind,
                   bool with_workspace, bool is_train) :
                   _is_train(is_train),
                   _with_workspace(with_workspace),
                   _alg_kind(alg_kind),
                   fwd(nullptr), data(nullptr), out(nullptr), workspace(nullptr) {
    _Init(input, output,
          kernel_h, kernel_w, stride_h, stride_w,
          padding_t, padding_b, padding_l, padding_r);
  }

  ~MKLDNNPoolingFwd() {}
  void SetDataHandle(const mxnet::NDArray &data,
                     const mxnet::NDArray &output,
                     const mxnet::NDArray *workspace = nullptr);
  void Execute();

 private:
  bool _is_train;
  bool _with_workspace;
  mkldnn::algorithm _alg_kind;
  std::shared_ptr<mkldnn::pooling_forward::primitive_desc> fwd_pd;
  std::shared_ptr<mkldnn::pooling_forward> fwd;
  std::shared_ptr<mkldnn::memory> data;
  std::shared_ptr<mkldnn::memory> out;
  std::shared_ptr<mkldnn::memory> workspace;

 private:
  void _Init(const mxnet::NDArray &input,
             const mxnet::NDArray &output,
             int kernel_h, int kernel_w,
             int stride_h, int stride_w,
             int padding_t, int padding_b, int padding_l, int padding_r);
};

void MKLDNNPoolingFwd::_Init(const mxnet::NDArray &input, const mxnet::NDArray &output,
                             int kernel_h, int kernel_w, int stride_h, int stride_w,
                             int padding_t, int padding_b, int padding_l, int padding_r) {
    auto src_md = input.GetMKLDNNData()->get_primitive_desc().desc();
    mkldnn::memory::dims dims = {src_md.data.dims[0],
                                 src_md.data.dims[1],
                                 static_cast<int>(output.shape()[2]),
                                 static_cast<int>(output.shape()[3])};
    auto dst_md = mkldnn::memory::desc({dims},
                                static_cast<mkldnn::memory::data_type>(src_md.data.data_type),
                                static_cast<mkldnn::memory::format>(src_md.data.format));
    auto engine = CpuEngine::Get()->get_engine();
    auto alg_kind = this->_alg_kind;
    if (alg_kind != pooling_max &&
        alg_kind != pooling_avg &&
        alg_kind != pooling_avg_include_padding &&
        alg_kind != pooling_avg_exclude_padding) {
        LOG(FATAL) << "MKLDNN Pooling: algorithm is not supported";
    }

    auto prop = mkldnn::prop_kind::forward_scoring;
    if (this->_is_train && alg_kind != mkldnn::algorithm::pooling_avg) {
        prop = mkldnn::prop_kind::forward_training;
    }

    if (this->_is_train && prop == mkldnn::prop_kind::forward_scoring) {
        LOG(INFO) << "MKLDNN Pooling: training with prop_kind is forward_scoring";
    }

    mkldnn::memory::dims strides = {stride_h,  stride_w  };
    mkldnn::memory::dims pad_l   = {padding_t, padding_l };
    mkldnn::memory::dims pad_r   = {padding_b, padding_r };
    mkldnn::memory::dims kernel  = {kernel_h,  kernel_w  };

    auto fwd_desc = mkldnn::pooling_forward::desc(prop, alg_kind, src_md, dst_md,
                                                  strides, kernel, pad_l, pad_r,
                                                  mkldnn::padding_kind::zero);
    this->fwd_pd.reset(new mkldnn::pooling_forward::primitive_desc(fwd_desc, engine));
    this->data.reset(new mkldnn::memory(input.GetMKLDNNData()->get_primitive_desc()));
    this->out.reset(new mkldnn::memory(this->fwd_pd->dst_primitive_desc()));
    if (this->_with_workspace) {
        this->workspace.reset(new mkldnn::memory(this->fwd_pd->workspace_primitive_desc()));
        this->fwd.reset(new mkldnn::pooling_forward(*(this->fwd_pd),
                                                    mkldnn::primitive::at(*(this->data)),
                                                    *(this->out),
                                                    *(this->workspace)));
    } else {
        this->fwd.reset(new mkldnn::pooling_forward(*(fwd_pd),
                                                    mkldnn::primitive::at(*(this->data)),
                                                    *(this->out)));
    }
    return;
}

void MKLDNNPoolingFwd::SetDataHandle(const mxnet::NDArray &data,
                                     const mxnet::NDArray &output,
                                     const mxnet::NDArray *workspace) {
    auto data_mem = data.GetMKLDNNData();
    auto out_mem = const_cast<NDArray&>(output).CreateMKLDNNData(
                                                    this->fwd_pd->dst_primitive_desc());
    this->data->set_data_handle(data_mem->get_data_handle());
    this->out->set_data_handle(out_mem->get_data_handle());
    if (this->_with_workspace && workspace == nullptr) {
        LOG(FATAL) << "MKLDNN Pooling: incorrect workspace input";
    }

    if (this->_with_workspace) {
        // auto ws_mem = const_cast<mxnet::NDArray*>(workspace)->CreateMKLDNNData(
        //                                      this->fwd_pd->workspace_primitive_desc());
        auto ws_mem = workspace->GetMKLDNNData();
        this->workspace->set_data_handle(ws_mem->get_data_handle());
    }
}

void MKLDNNPoolingFwd::Execute() {
    if (this->fwd) {
        MKLDNNStream::Get()->RegisterPrim(*(this->fwd));
        MKLDNNStream::Get()->Submit();
    } else {
        LOG(FATAL) << "MKLDNN Pooling: forward primitive is nullptr";
    }
}

static inline bool SupportMKLDNNPooling(const PoolingParam &param) {
  return param.kernel.ndim() == 2
      && (param.pool_type == pool_enum::kMaxPooling ||
          param.pool_type == pool_enum::kAvgPooling);
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

static inline mkldnn::algorithm
GetMKLDNNPoolAlgo(const PoolingParam &param) {
  switch (param.pool_type) {
    case pool_enum::kMaxPooling:
      return mkldnn::algorithm::pooling_max;
      break;
    case pool_enum::kAvgPooling:
      return mkldnn::algorithm::pooling_avg;
      break;
    default:
      LOG(FATAL) << "MKLDNN Pooling: Unknown pooling method.";
      return mkldnn::algorithm::pooling_max;
  }
}

inline static mkldnn::pooling_forward::primitive_desc
GetPoolingFwd(const PoolingParam &param,
              bool is_train,
              const memory::desc &data_md,
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

  auto engine = CpuEngine::Get()->get_engine();
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

  pooling_forward::desc poolingFwd_desc(kind, alg, data_md, out_md,
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

inline bool MKLDNNRequireWorkspace(const PoolingParam &param) {
  return param.pool_type != pool_enum::kAvgPooling;
}

typedef MKLDNNParamOpSign<PoolingParam> MKLDNNPoolingSignature;

static inline MKLDNNPoolingFwd &GetPoolingFwd(const PoolingParam &param,
                                              bool is_train,
                                              const NDArray &data,
                                              const NDArray &output) {
  static thread_local std::unordered_map<MKLDNNPoolingSignature,
                                         MKLDNNPoolingFwd,
                                         MKLDNNOpHash> pooling_fwds;

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

    auto pad_t_ = param.pad[0], pad_b_ = param.pad[0];
    auto pad_l_ = param.pad[1], pad_r_ = param.pad[1];
    auto stride_h_ = param.stride[0], stride_w_ = param.stride[1];

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
    MKLDNNPoolingFwd fwd(data, output, kernel_h_, kernel_w_, stride_h_, stride_w_,
                         pad_t_, pad_b_, pad_l_, pad_r_, alg, with_workspace, is_train);
    auto ins_ret = pooling_fwds.insert(
                         std::pair<MKLDNNPoolingSignature, MKLDNNPoolingFwd>(key, fwd));
    CHECK(ins_ret.second);
    it = ins_ret.first;
  }
  return it->second;
}

void MKLDNNPoolingCompute(const OpContext &ctx, const PoolingParam &param,
                          const NDArray &in_data, const OpReqType &req,
                          const NDArray &out_data, const NDArray *workspace) {
  auto fwd = GetPoolingFwd(param, ctx.is_train, in_data, out_data);
  fwd.SetDataHandle(in_data, out_data, workspace);
  fwd.Execute();
}

void MKLDNNPoolingGradCompute(const OpContext &ctx, const PoolingParam &param,
                              const NDArray &out_grad, const NDArray &in_data,
                              const NDArray *workspace, const OpReqType &req,
                              const NDArray &in_grad) {
  if (req == kNullOp) {
    return;
  }

  TmpMemMgr::Get()->Init(ctx.requested[0]);
  auto diff_dst_mem = out_grad.GetMKLDNNData();
  auto input_mem = in_data.GetMKLDNNData();
  mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  memory::dims dims = {data_md.data.dims[0], data_md.data.dims[1],
                       static_cast<int>(out_grad.shape()[2]),
                       static_cast<int>(out_grad.shape()[3])};
  memory::desc out_md({dims},
                      static_cast<memory::data_type>(data_md.data.data_type),
                      static_cast<memory::format>(data_md.data.format));
  auto pdesc_fwd = GetPoolingFwd(param, ctx.is_train, data_md, out_md);

  mkldnn::memory::desc diff_md = diff_dst_mem->get_primitive_desc().desc();
  memory::dims dims1 = {diff_md.data.dims[0], diff_md.data.dims[1],
                        static_cast<int>(in_grad.shape()[2]),
                        static_cast<int>(in_grad.shape()[3])};
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
  pooling_backward::desc desc(alg, diff_in_md, diff_md,
                              {static_cast<int>(param.stride[0]),
                              static_cast<int>(param.stride[1])},
                              {kernel_h_, kernel_w_},
                              {static_cast<int>(param.pad[0]),
                              static_cast<int>(param.pad[1])},
                              {static_cast<int>(param.pad[0]),
                              static_cast<int>(param.pad[1])},
                              padding_kind::zero);
  pooling_backward::primitive_desc pdesc(desc, cpu_engine, pdesc_fwd);

  auto diff_src_mem =
      CreateMKLDNNMem(in_grad, pdesc.diff_src_primitive_desc(), req);

  if (MKLDNNRequireWorkspace(param)) {
    CHECK(workspace != nullptr);
    auto workspace_mem = workspace->GetMKLDNNData();
    MKLDNNStream::Get()->RegisterPrim(
        pooling_backward(pdesc, *diff_dst_mem, primitive::at(*workspace_mem),
                         *diff_src_mem.second));
  } else {
    MKLDNNStream::Get()->RegisterPrim(
        pooling_backward(pdesc, *diff_dst_mem, *diff_src_mem.second));
  }
  CommitOutput(in_grad, diff_src_mem);
  MKLDNNStream::Get()->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_POOLING_INL_H_

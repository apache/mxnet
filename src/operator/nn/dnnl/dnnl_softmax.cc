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
 * \file dnnl_softmax.cc
 * \brief
 * \author Da Zheng
 */

#include "../softmax-inl.h"
#include "./dnnl_base-inl.h"
#include "./dnnl_ops-inl.h"

#if MXNET_USE_ONEDNN == 1
namespace mxnet {
namespace op {

static dnnl::softmax_forward::primitive_desc GetSoftmaxFwdPd(bool is_train,
                                                             const int axis,
                                                             const dnnl::memory& input_mem) {
  dnnl::memory::desc data_md   = input_mem.get_desc();
  auto cpu_engine              = CpuEngine::Get()->get_engine();
  auto prop = is_train ? dnnl::prop_kind::forward_training : dnnl::prop_kind::forward_scoring;
  auto desc = dnnl::softmax_forward::desc(prop, data_md, axis);
  return dnnl::softmax_forward::primitive_desc(desc, cpu_engine);
}

static dnnl::softmax_backward::primitive_desc GetSoftmaxBwdPd(
    const dnnl::memory& diff_mem,
    const dnnl::memory& data_mem,
    const int axis,
    const dnnl::softmax_forward::primitive_desc& hint_fwd_pd) {
  dnnl::memory::desc diff_md   = diff_mem.get_desc();
  dnnl::memory::desc data_md   = data_mem.get_desc();
  auto cpu_engine              = CpuEngine::Get()->get_engine();
  auto desc                    = dnnl::softmax_backward::desc(diff_md, data_md, axis);
  return dnnl::softmax_backward::primitive_desc(desc, cpu_engine, hint_fwd_pd);
}

bool SupportDNNLSoftmax(const SoftmaxParam& param, const NDArray& data, const NDArray& output) {
  // DNNL does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  const int ndim      = data.shape().ndim();
  const int in_dtype  = data.dtype();
  const int out_dtype = output.dtype();
  const int axis      = CheckAxis(param.axis, ndim);
  // DNNL does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  // Currently, DNNL shows bad performance when softmax is not performed on the last dimension
  if (param.temperature.has_value() || in_dtype != mshadow::kFloat32 || in_dtype != out_dtype ||
      axis != (ndim - 1)) {
    return false;
  }

  // only supports ndim = 1, 2, 3, 4 for now
  return (ndim >= 1 && ndim <= 4);
}

class DNNLSoftmaxFwd {
 public:
  dnnl::softmax_forward::primitive_desc pd;

  DNNLSoftmaxFwd(const bool is_train, const int axis, const dnnl::memory& input)
      : pd(GetSoftmaxFwdPd(is_train, axis, input)) {
    fwd_ = std::make_shared<dnnl::softmax_forward>(pd);
  }

  const dnnl::softmax_forward& GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<dnnl::softmax_forward> fwd_;
};

typedef ParamOpSign<SoftmaxParam> DNNLSoftmaxSignature;

static DNNLSoftmaxFwd& GetSoftmaxFwd(const SoftmaxParam& param,
                                     const int real_axis,
                                     const bool is_train,
                                     const NDArray& data,
                                     const NDArray& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxFwd, OpHash> fwds;
#endif

  DNNLSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    DNNLSoftmaxFwd fwd(is_train, real_axis, *(data.GetDNNLData()));
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void DNNLSoftmaxForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& in_data,
                        const OpReqType& req,
                        const NDArray& out_data) {
  if (req == kNullOp)
    return;
  // same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now.
  CHECK_NE(req, kAddTo);

  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis                  = CheckAxis(param.axis, in_data.shape().ndim());
  auto fwd                  = GetSoftmaxFwd(param, axis, ctx.is_train, in_data, out_data);

  auto in_mem        = in_data.GetDNNLData();
  auto out_mem       = out_data.GetDNNLData(fwd.pd.dst_desc());
  DNNLStream* stream = DNNLStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), {{DNNL_ARG_SRC, *in_mem}, {DNNL_ARG_DST, *out_mem}});
  stream->Submit();
}

class DNNLSoftmaxBwd {
 public:
  dnnl::softmax_backward::primitive_desc pd;

  DNNLSoftmaxBwd(const dnnl::memory& diff_mem,
                 const dnnl::memory& data_mem,
                 const int axis,
                 const dnnl::softmax_forward::primitive_desc& hint_fwd_pd)
      : pd(GetSoftmaxBwdPd(diff_mem, data_mem, axis, hint_fwd_pd)) {
    bwd_ = std::make_shared<dnnl::softmax_backward>(pd);
  }

  const dnnl::softmax_backward& GetBwd() const {
    return *bwd_;
  }

 private:
  std::shared_ptr<dnnl::softmax_backward> bwd_;
};

static DNNLSoftmaxBwd& GetSoftmaxBwd(const SoftmaxParam& param,
                                     const int real_axis,
                                     const std::vector<NDArray>& data,
                                     const std::vector<NDArray>& output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxBwd, OpHash> bwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<DNNLSoftmaxSignature, DNNLSoftmaxBwd, OpHash> bwds;
#endif

  DNNLSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(data);
  key.AddSign(output);

  auto it = bwds.find(key);
  if (it == bwds.end()) {
    auto diff_mem = data[0].GetDNNLData();
    auto data_mem = data[1].GetDNNLData();
    auto fwd_pd   = GetSoftmaxFwdPd(true, real_axis, *data_mem);
    DNNLSoftmaxBwd bwd(*diff_mem, *data_mem, real_axis, fwd_pd);
    it = AddToCache(&bwds, key, bwd);
  }
  return it->second;
}

void DNNLSoftmaxBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<NDArray>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<NDArray>& out_data) {
  if (req[0] == kNullOp)
    return;
  CHECK_EQ(in_data.size(), 2U);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis                  = CheckAxis(param.axis, in_data[1].shape().ndim());
  auto diff_mem             = in_data[0].GetDNNLData();
  auto data_mem             = in_data[1].GetDNNLData();
  auto bwd                  = GetSoftmaxBwd(param, axis, in_data, out_data);

  auto out_mem         = CreateDNNLMem(out_data[0], bwd.pd.diff_src_desc(), req[0]);
  DNNLStream* stream   = DNNLStream::Get();
  dnnl_args_map_t args = {{DNNL_ARG_DST, *data_mem},
                          {DNNL_ARG_DIFF_DST, *diff_mem},
                          {DNNL_ARG_DIFF_SRC, *out_mem.second}};

  stream->RegisterPrimArgs(bwd.GetBwd(), args);
  CommitOutput(out_data[0], out_mem);
  stream->Submit();
}

}  // namespace op
}  // namespace mxnet
#endif
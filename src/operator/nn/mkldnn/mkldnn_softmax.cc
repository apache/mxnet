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
 * \file mkldnn_softmax.cc
 * \brief
 * \author Da Zheng
*/

#include "../softmax-inl.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

static mkldnn::softmax_forward::primitive_desc GetSoftmaxFwdPd(bool is_train,
                                                               const int axis,
                                                               const mkldnn::memory &input_mem) {
  mkldnn::memory::desc data_md = input_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training
                       : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::softmax_forward::desc(prop, data_md, axis);
  return mkldnn::softmax_forward::primitive_desc(desc, cpu_engine);
}


bool SupportMKLDNNSoftmax(const SoftmaxParam &param,
                          const NDArray &data,
                          const NDArray &output) {
  // MKLDNN does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  const int ndim = data.shape().ndim();
  const int in_dtype = data.dtype();
  const int out_dtype = output.dtype();
  const int axis = CheckAxis(param.axis, ndim);
  // MKLDNN does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  // Currently, MKLDNN shows bad performance when softmax is not performed on the last dimension
  if (param.temperature.has_value() ||
      in_dtype != mshadow::kFloat32 ||
      in_dtype != out_dtype ||
      axis != (ndim - 1)) {
    return false;
  }

  // only supports ndim = 1, 2, 3, 4 for now
  return (ndim >= 1 && ndim <= 4);
}

class MKLDNNSoftmaxFwd {
 public:
  mkldnn::softmax_forward::primitive_desc pd;

  MKLDNNSoftmaxFwd(const bool is_train,
                   const int axis,
                   const mkldnn::memory &input) : pd(GetSoftmaxFwdPd(is_train, axis, input)) {
    fwd_ = std::make_shared<mkldnn::softmax_forward>(pd);
  }

  const mkldnn::softmax_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::softmax_forward> fwd_;
};

typedef ParamOpSign<SoftmaxParam> MKLDNNSoftmaxSignature;

static MKLDNNSoftmaxFwd &GetSoftmaxFwd(const SoftmaxParam &param,
                                       const int real_axis,
                                       const bool is_train,
                                       const NDArray &data,
                                       const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNSoftmaxSignature, MKLDNNSoftmaxFwd, OpHash> fwds;
#endif

  MKLDNNSoftmaxSignature key(param);
  key.AddSign(real_axis);
  key.AddSign(is_train);
  key.AddSign(data);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSoftmaxFwd fwd(is_train, real_axis, *(data.GetMKLDNNData()));
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNSoftmaxForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const NDArray &in_data,
                          const OpReqType &req,
                          const NDArray &out_data) {
  if (req == kNullOp) return;
  // same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now.
  CHECK_NE(req, kAddTo);

  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, in_data.shape().ndim());
  auto fwd = GetSoftmaxFwd(param, axis, ctx.is_train, in_data, out_data);

  auto in_mem = in_data.GetMKLDNNData();
  auto out_mem = out_data.GetMKLDNNData(fwd.pd.dst_desc());
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), {{MKLDNN_ARG_SRC, *in_mem}, {MKLDNN_ARG_DST, *out_mem}});
  stream->Submit();
}

}   // namespace op
}   // namespace mxnet
#endif

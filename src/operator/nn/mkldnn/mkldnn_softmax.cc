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

bool SupportMKLDNNSoftmax(const SoftmaxParam &param,
                          const NDArray &data,
                          const NDArray &output) {
  int ndim = data.shape().ndim();
  int in_dtype = data.dtype();
  int out_dtype = output.dtype();

  // MKLDNN does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  if (param.temperature.has_value() ||
      in_dtype != mshadow::kFloat32 ||
      in_dtype != out_dtype) {
    return false;
  }
  // only support ndim = 1, 2, 3, 4 for now
  return (ndim >= 1 && ndim <= 4);
}

static mkldnn::softmax_forward::primitive_desc GetSoftmaxFwdPd(const int axis,
                                                               const bool is_train,
                                                               const mkldnn::memory &input) {
  auto data_md = input.get_primitive_desc().desc();
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::softmax_forward::desc(prop, data_md, axis);
  auto pd = mkldnn::softmax_forward::primitive_desc(desc, CpuEngine::Get()->get_engine());
  return pd;
}

/*
class MKLDNNSoftmaxFwd {
 public:
  mkldnn::softmax_forward::primitive_desc pd_;
  MKLDNNSoftmaxFwd(const int axis,
                   const bool is_train,
                   const mkldnn::memory &input) : pd_(GetSoftmaxFwdPd(axis, is_train, input)) {
    data_ = std::make_shared<mkldnn::memory>(pd_.src_primitive_desc(), nullptr);
    output_ = std::make_shared<mkldnn::memory>(pd_.dst_primitive_desc(), nullptr);
    fwd_ = std::make_shared<mkldnn::softmax_forward>(pd_, *data_, *output_);
  }

  void SetNewMem(const mkldnn::memory &data,
                 const mkldnn::memory &output) {
    data_->set_data_handle(data.get_data_handle());
    output_->set_data_handle(output.get_data_handle());
  }

  const mkldnn::softmax_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> output_;
  std::shared_ptr<mkldnn::softmax_forward> fwd_;
};


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
    MKLDNNSoftmaxFwd fwd(real_axis, is_train, *(data.GetMKLDNNData()));
    it = AddToCache(&fwd, key, fwd);
  }
  return it->second;
}
*/

void MKLDNNSoftmaxForward(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &in_data,
                          const OpReqType &req,
                          const NDArray &out_data) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo);
  // same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now.

  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  int axis = CheckAxis(param.axis, in_data.shape().ndim());

  NDArray data = in_data;
  if (in_data.IsView() && in_data.IsMKLDNNData())
    data = in_data.Reorder2Default();

  auto data_mem = data.GetMKLDNNData();
  auto pd = GetSoftmaxFwdPd(axis, ctx.is_train, *data_mem);
  CHECK(data_mem->get_primitive_desc() == pd.src_primitive_desc());
  auto out_mem = const_cast<NDArray &>(out_data).CreateMKLDNNData(pd.dst_primitive_desc());
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::softmax_forward(pd, *data_mem, *out_mem));
  stream->Submit();
}

}   // namespace op
}   // namespace mxnet
#endif

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
 * \file mkldnn_reshape.cc
 * \brief Implement reshape operator via MKL-DNN reorder primitive
 * \author Tao Lv
*/

#if MXNET_USE_MKLDNN == 1
#include "../../tensor/elemwise_unary_op.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_reshape-inl.h"

namespace mxnet {
namespace op {

MKLDNNReshapeFwd::MKLDNNReshapeFwd(const OpReqType &req, const NDArray &input,
                                   const NDArray &output) {
  const auto engine = CpuEngine::Get()->get_engine();
  data_ = std::make_shared<mkldnn::memory>(input.GetMKLDNNData()->get_primitive_desc(), nullptr);
  // Create temp memory
  auto temp_dims = mkldnn::memory::dims(input.shape().begin(), input.shape().end());
  auto temp_type = static_cast<mkldnn::memory::data_type>(get_mkldnn_type(input.dtype()));
  auto temp_fmt = static_cast<mkldnn::memory::format>(GetDefaultFormat(input.shape().ndim()));
  auto temp_desc = mkldnn::memory::desc(temp_dims, temp_type, temp_fmt);
  auto temp_pd = mkldnn::memory::primitive_desc(temp_desc, engine);
  out_ = std::make_shared<mkldnn::memory>(temp_pd, nullptr);
  if (req == kWriteInplace) {
    // If the input has MKL-DNN internal layout, we need reorder it to a temporal buffer with
    // default layout and copy from the temporal buffer back to output buffer which has the same
    // address with input buffer.
    // If the input has default layout, then nothing need to do.
    if (input.IsMKLDNNData()) {
      temp_ = std::make_shared<mkldnn::memory>(temp_pd, nullptr);
      prims_.push_back(mkldnn::reorder(*data_, *temp_));  // reorder to default
      prims_.push_back(mkldnn::reorder(*temp_, *out_));   // copy back
    }
  } else if (req == kWriteTo) {
    prims_.push_back(mkldnn::reorder(*data_, *out_));
  } else {
    LOG(FATAL) << "not supported req type: " << req;
  }
}

int MKLDNNReshapeFwd::GetWorkspaceSize() {
  return temp_ ? temp_->get_primitive_desc().get_size() : 0;
}

void MKLDNNReshapeFwd::SetNewMem(const NDArray &input,
                                 const NDArray &output,
                                 void* workspace) {
  this->data_->set_data_handle(input.GetMKLDNNData()->get_data_handle());
  this->out_->set_data_handle(output.GetMKLDNNData()->get_data_handle());
  if (workspace) {
    this->temp_->set_data_handle(workspace);
  }
}

void MKLDNNReshapeFwd::Execute(const NDArray &input,
                               const NDArray &output,
                               void* workspace) {
  if (this->prims_.size()) {
    // set memory handles
    SetNewMem(input, output, workspace);
    // register primitives
    auto stream = MKLDNNStream::Get();
    for (auto &v : this->prims_) {
      stream->RegisterPrim(v);
    }
    stream->Submit();
  }
  // invalidate mkldnn memory in output
  const_cast<NDArray &>(output).InvalidateMKLDNNData();
}

MKLDNNReshapeFwd &GetReshapeForward(const OpReqType &req,
                                    const NDArray &input,
                                    const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNReshapeSignature,
                                         MKLDNNReshapeFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNReshapeSignature,
                                            MKLDNNReshapeFwd, OpHash> fwds;
#endif
  MKLDNNReshapeSignature key;
  key.AddSign(req);
  key.AddSign(input);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNReshapeFwd fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNReshapeForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const NDArray &input,
                          const OpReqType &req,
                          const NDArray &output) {
  // For mkldnn non-supported input, it shouldn't hold mkldnn memory, so let's simply fallback to
  // naive implement.
  if (input.shape().ndim() > 4 || !SupportMKLDNNQuantize(input.dtype())) {
    if (req != kWriteInplace) {
      FallBackCompute(UnaryOp::IdentityCompute<cpu>, attrs, ctx, {input}, {req}, {output});
    }
    return;
  }
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";
  auto fwd = GetReshapeForward(req, input, output);
  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws = ctx.requested[0]
      .get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = static_cast<void*>(ws.dptr_);
  }
  fwd.Execute(input, output, ws_ptr);
}

}  // namespace op
}  // namespace mxnet
#endif

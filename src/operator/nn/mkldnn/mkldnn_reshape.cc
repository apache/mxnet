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

#if MXNET_USE_MKLDNN == 100

#include <mkldnn.hpp>
#include "mkldnn_reshape-inl.h"

namespace mxnet {
namespace op {

bool SupportMKLDNNReshape(const NDArray &in_data,
                          const NDArray &out_data) {
  auto in_ndim = in_data.shape().ndim();
  auto out_ndim = out_data.shape().ndim();

  if (in_ndim > 4 ||
      in_data.dtype() != mshadow::kFloat32 ||
      out_ndim > 4)
    return false;

  return true;
}

MKLDNNReshapeFwd::MKLDNNReshapeFwd(const OpReqType &req,
                                   const NDArray &input,
                                   const NDArray &output) {
  auto engine = CpuEngine::Get()->get_engine();

  // source
  auto in_mem = input.GetMKLDNNData();
  auto in_md = in_mem->get_desc();

  // temp_
  auto temp_md = GetDesc(in_md, GetDefaultFormat(in_md));
  temp_ = std::make_shared<mkldnn::memory>(temp_md, engine);

  // destination
  out_ = std::make_shared<mkldnn::memory>(temp_md, engine);

  if (req == kWriteInplace) {
    // If the input has MKL-DNN internal layout, we need reorder it to a temporal buffer with
    // default layout and copy from the temporal buffer back to output buffer which has the same
    // address with input buffer.
    // If the input has default layout, then nothing need to do.
    if (input.IsMKLDNNData()) {
      prims_.push_back(mkldnn::reorder(*in_mem, *temp_));  // reorder to default
      prims_.push_back(mkldnn::reorder(*temp_, *out_));   // copy back

      needInvalidateInput = true;
    }
  } else if (req == kWriteTo) {
    if (input.IsMKLDNNData()) {
      prims_.push_back(mkldnn::reorder(*in_mem, *temp_));   // reorder to default
      prims_.push_back(mkldnn::reorder(*temp_, *out_));     // copy to the output buffer

      needInvalidateInput = false;
    } else {
      prims_.push_back(mkldnn::reorder(*in_mem, *out_));    // copy directly from input to output

      needInvalidateInput = false;
    }
  } else {
    LOG(FATAL) << "not supported req type: " << req;
  }
}

int MKLDNNReshapeFwd::GetWorkspaceSize() {
  return temp_ ? temp_->get_desc().get_size() : 0;
}

void MKLDNNReshapeFwd::Execute(const NDArray &input,
                               const NDArray &output,
                               void* workspace) {
  auto stream = MKLDNNStream::Get();
  auto in_mem = input.GetMKLDNNData();
  auto in_md = in_mem->get_desc();
  // register primitives and arguments
  size_t prims_size = prims_.size();
  if (prims_size == 1) {
    args_map_.push_back({{MKLDNN_ARG_FROM, *in_mem},
                         {MKLDNN_ARG_TO, *output.GetMKLDNNData()}});
  } else if (prims_size == 2) {
    auto temp_md = GetDesc(in_md, GetDefaultFormat(in_md));
    temp_ = std::make_shared<mkldnn::memory>(temp_md, CpuEngine::Get()->get_engine(),
              workspace);
    args_map_.push_back({{MKLDNN_ARG_FROM, *in_mem},
                        {MKLDNN_ARG_TO, *temp_}});
    args_map_.push_back({{MKLDNN_ARG_FROM, *temp_},
                        {MKLDNN_ARG_TO, *output.GetMKLDNNData()}});
  }
  for (size_t i = 0; i < prims_size; i++) {
    stream->RegisterPrimArgs(prims_[i], args_map_[i]);
  }
  stream->Submit();
  // invalidate mkldnn memory in input
  if (needInvalidateInput) {
    const_cast<NDArray &>(input).InvalidateMKLDNNData();
  }
}

void MKLDNNReshapeForward(const nnvm::NodeAttrs& attrs,
                          const OpContext &ctx,
                          const NDArray &input,
                          const OpReqType &req,
                          const NDArray &output) {
  const ReshapeParam& param = nnvm::get<ReshapeParam>(attrs.parsed);
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";

  auto fwd = GetCachedForward<MKLDNNReshapeFwd, ReshapeParam,
                              MKLDNNReshapeSignature>(param, req, input, output);

  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws = ctx.requested[0]
      .get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = reinterpret_cast<void*>(ws.dptr_);
  }

  fwd.Execute(input, output, ws_ptr);
}
}  // namespace op
}  // namespace mxnet
#endif

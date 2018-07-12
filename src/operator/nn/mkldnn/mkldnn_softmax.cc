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
#include "../../tensor/broadcast_reduce_op.h"

#if MXNET_USE_MKLDNN == 1
namespace mxnet {
namespace op {

bool SupportMKLDNNSoftmax(const SoftmaxParam &param) {
  // MKLDNN does not support temperature argument in their softmax function
  // now. Need update this once they start to support it.
  if (param.temperature.has_value()) {
    return false;
  }
  return true;
}

void MKLDNNSoftmaxForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const NDArray &in_data, const OpReqType &req,
                          const NDArray &out_data) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  auto input_mem = in_data.GetMKLDNNData();
  mkldnn::memory::primitive_desc data_mpd = input_mem->get_primitive_desc();
  mkldnn::memory::desc data_md = data_mpd.desc();
  int axis = CheckAxis(param.axis, in_data.shape().ndim());

  auto cpu_engine = data_mpd.get_engine();
  auto prop = ctx.is_train
    ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  mkldnn::softmax_forward::desc desc = mkldnn::softmax_forward::desc(prop,
      data_md, axis);
  mkldnn::softmax_forward::primitive_desc pdesc(desc, cpu_engine);

  auto output_memory = out_data.GetMKLDNNData();
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::softmax_forward(pdesc, *input_mem, *output_memory));
  stream->Submit();
}

}   // namespace op
}   // namespace mxnet
#endif

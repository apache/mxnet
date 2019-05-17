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

static mkldnn::softmax_forward::primitive_desc GetSoftmaxFwdPd(const int axis,
                                                               const bool is_train,
                                                               const mkldnn::memory &input) {
  auto data_md = input.get_primitive_desc().desc();
  auto prop = is_train ? mkldnn::prop_kind::forward_training : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::softmax_forward::desc(prop, data_md, axis);
  auto pd = mkldnn::softmax_forward::primitive_desc(desc, CpuEngine::Get()->get_engine());
  return pd;
}

void MKLDNNSoftmaxForward(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &in_data,
                          const OpReqType &req,
                          const NDArray &out_data) {
  if (req == kNullOp) return;
  // same as the FCompute path, softmax only supports kWriteTo and kWriteInplace for now.
  CHECK_NE(req, kAddTo);
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
  const int axis = CheckAxis(param.axis, in_data.shape().ndim());

  NDArray data = in_data;
  if (in_data.IsView() && in_data.IsMKLDNNData()) {
    data = in_data.Reorder2Default();
  }

  auto data_mem = data.GetMKLDNNData();
  auto pd = GetSoftmaxFwdPd(axis, ctx.is_train, *data_mem);
  auto out_mem = CreateMKLDNNMem(out_data, pd.dst_primitive_desc(), req);
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrim(mkldnn::softmax_forward(pd, *data_mem, *out_mem.second));
  CommitOutput(out_data, out_mem);
  stream->Submit();
}
}   // namespace op
}   // namespace mxnet
#endif

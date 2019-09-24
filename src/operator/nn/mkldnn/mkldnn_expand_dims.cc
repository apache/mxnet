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
 * \file mkldnn_expand_dims.cc
 * \brief Implement expand_dims operator via MKL-DNN reorder primitive
 * \author Wuxun Zhang
*/

#if MXNET_USE_MKLDNN == 100

#include "mkldnn_reshape-inl.h"

namespace mxnet {
namespace op {

class MKLDNNExpandDimsFwd : public MKLDNNReshapeFwd {
 public:
  explicit MKLDNNExpandDimsFwd(const OpReqType &req,
                               const NDArray &input,
                               const NDArray &output)
    : MKLDNNReshapeFwd(req, input, output) {}
};

typedef ParamOpSign<ExpandDimParam> MKLDNNExpandDimsSignature;

void MKLDNNExpandDimsForward(const nnvm::NodeAttrs &attrs,
                             const OpContext &ctx,
                             const NDArray &input,
                             const OpReqType &req,
                             const NDArray &output) {
  const ExpandDimParam& param = nnvm::get<ExpandDimParam>(attrs.parsed);
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";

  auto fwd = GetCachedForward<MKLDNNExpandDimsFwd, ExpandDimParam,
                              MKLDNNExpandDimsSignature>(param, req, input, output);

  auto ws_size = fwd.GetWorkspaceSize();
  void* ws_ptr = nullptr;
  if (ws_size) {
    mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
    mshadow::Tensor<cpu, 1, char> ws = ctx.requested[0]
      .get_space_typed<cpu, 1, char>(mshadow::Shape1(ws_size), s);
    ws_ptr = reinterpret_cast<void*>(ws.dptr_);
  }

  fwd.Execute(input, output, req, ws_ptr);
}

}  // namespace op
}  // namespace mxnet

#endif

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
 * \file mkldnn_flatten.cc
 * \brief Implement flatten operator by using mkldnn reorder primitive
 * \author Wuxun Zhang
*/

#if MXNET_USE_MKLDNN == 1

#include "mkldnn_flatten-inl.h"

namespace mxnet {
namespace op {

static MKLDNNFlattenFwd &GetFlattenForward(const OpReqType &req,
                                           const NDArray &input,
                                           const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature,
                                         MKLDNNFlattenFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature,
                                            MKLDNNFlattenFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(req);
  key.AddSign(input);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNFlattenFwd fwd(req, input, output);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNFlattenForward(const nnvm::NodeAttrs &attrs,
                          const OpContext &ctx,
                          const NDArray &input,
                          const OpReqType &req,
                          const NDArray &output) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo) << "kAddTo is not supported yet";

  auto fwd = GetFlattenForward(req, input, output);
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

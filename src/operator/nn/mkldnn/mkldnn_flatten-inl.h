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

#include "mkldnn_reshape-inl.h"

namespace mxnet {
namespace op {

class MKLDNNFlattenFwd : public MKLDNNReshapeFwd {
 public:
  explicit MKLDNNFlattenFwd(const OpReqType &req, const NDArray &input, const NDArray &output)
      : MKLDNNReshapeFwd(req, input, output) {}
};

void MKLDNNFlattenForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx, const NDArray &input,
                          const OpReqType &req, const NDArray &output);

}  // namespace op
}  // namespace mxnet

#endif

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
#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_SUBGRAPH_BASE_INL_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_SUBGRAPH_BASE_INL_H_
#if MXNET_USE_MKLDNN == 1

#include "../subgraph_property.h"

namespace mxnet {
namespace op {

static inline bool SupportMKLDNNAttr(const std::shared_ptr<NodeAttr>& node_attr) {
  if (node_attr) {
    int ndim = node_attr->ishape[0].ndim();
    return (node_attr->dispatch_mode == DispatchMode::kFComputeEx) &&
           (node_attr->itype[0] == mshadow::kFloat32) && (ndim == 1 || ndim == 2 || ndim == 4);
  } else {
    return true;
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_SUBGRAPH_BASE_INL_H_

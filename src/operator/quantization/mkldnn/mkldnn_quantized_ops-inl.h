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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_quantized_ops-inl.h
 * \brief Common functions used by MKLDNN Quantized FullyConnected operator
 * \author Ciyong Chen
 */

#ifndef MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_OPS_INL_H_
#define MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_OPS_INL_H_

#if MXNET_USE_MKLDNN == 1

#include <mxnet/ndarray.h>
#include <vector>

namespace mxnet {
namespace op {

void MKLDNNQuantizedFullyConnectedForward(const nnvm::NodeAttrs &attrs,
                                          const OpContext &ctx,
                                          const std::vector<NDArray> &in_data,
                                          const std::vector<OpReqType> &req,
                                          const std::vector<NDArray> &out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_QUANTIZATION_MKLDNN_MKLDNN_QUANTIZED_OPS_INL_H_

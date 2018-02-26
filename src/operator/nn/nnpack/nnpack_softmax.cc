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
 * \file nnpack_softmax.cc
 * \brief
 * \author David Braude
*/

#include "../softmax-inl.h"
#include "./nnpack_ops-inl.h"


#if MXNET_USE_NNPACK == 1
namespace mxnet {
namespace op {

void NNPACKSoftmaxForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const NDArray &in_data, const OpReqType &req,
                          const NDArray &out_data) {
  const SoftmaxParam& param = nnvm::get<SoftmaxParam>(attrs.parsed);
//     enum nnp_status nnp_softmax_output(
//     size_t batch_size,
//     size_t channels,
//     const float input[],
//     float output[],
//     pthreadpool_t threadpool);
}

}   // namespace op
}   // namespace mxnet
#endif

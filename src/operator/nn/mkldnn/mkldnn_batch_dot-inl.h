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
 * \file mkldnn_batch_dot-inl.h
 */

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_DOT_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_DOT_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <numeric>
#include <utility>
#include <vector>
#include "../../tensor/dot-inl.h"
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

namespace mxnet {
namespace op {

using batch_dot_fwd_t = mkldnn::matmul;
using batch_dot_fwd_pd_t = mkldnn::matmul::primitive_desc;

typedef ParamOpSign<DotParam> BatchDotSignature;

class MKLDNNBatchDotFwd {
 public:
  static MKLDNNBatchDotFwd &GetCached(const DotParam &param,
                                      const std::vector<NDArray> &inputs,
                                      const std::vector<NDArray> &outputs);

  MKLDNNBatchDotFwd(const DotParam &param, const std::vector<NDArray> &inputs,
                    const std::vector<NDArray> &outputs);

  void Execute(const std::vector<NDArray> &inputs,
               const std::vector<OpReqType> &req,
               const std::vector<NDArray> &outputs);

 private:
  std::shared_ptr<batch_dot_fwd_t> fwd;
  std::shared_ptr<batch_dot_fwd_pd_t> fwd_pd;
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_MKLDNN_MKLDNN_BATCH_DOT_INL_H__

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
 * \file dnnl_batch_dot-inl.h
 * \author: Bartosz Kuncer, bartosz.kuncer@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <numeric>
#include <utility>
#include <vector>

#include "../../tensor/dot-inl.h"
#include "./dnnl_base-inl.h"
#include "./dnnl_ops-inl.h"

namespace mxnet {
namespace op {

using batch_dot_fwd_t    = dnnl::matmul;
using batch_dot_fwd_pd_t = dnnl::matmul::primitive_desc;

typedef ParamOpSign<DotParam> BatchDotSignature;

class DNNLBatchDotFwd {
 public:
  static DNNLBatchDotFwd& GetCached(const DotParam& param,
                                    const std::vector<NDArray>& inputs,
                                    const std::vector<NDArray>& outputs);

  DNNLBatchDotFwd(const DotParam& param,
                  const std::vector<NDArray>& inputs,
                  const std::vector<NDArray>& outputs);

  void Execute(const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  std::shared_ptr<batch_dot_fwd_t> fwd;
  std::shared_ptr<batch_dot_fwd_pd_t> fwd_pd;
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_BATCH_DOT_INL_H__

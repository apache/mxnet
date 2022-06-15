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
 * \file dnnl_sum-inl.h
 * \brief
 * \author Wolinski Piotr piotr.wolinski@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_SUM_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_SUM_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <vector>

#include <dnnl.hpp>
#include "operator/nn/dnnl/dnnl_base-inl.h"
#include "operator/operator_common.h"

namespace mxnet {
namespace op {

using sum_t    = dnnl::sum;
using sum_pd_t = dnnl::sum::primitive_desc;

class DNNLSumFwd {
 public:
  typedef OpSignature DNNLSumSignature;

  static DNNLSumFwd& GetCached(const std::vector<NDArray>& inputs,
                               const std::vector<NDArray>& outputs);

  explicit DNNLSumFwd(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs);

  void Execute(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs);

 private:
  std::shared_ptr<sum_t> fwd;
  std::shared_ptr<sum_pd_t> fwd_pd;
};

void DNNLSumForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_SUM_INL_H_

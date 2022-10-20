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
 * \file dnnl_dot-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_DOT_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_DOT_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <memory>
#include <vector>

#include "dnnl_base-inl.h"
#include "operator/tensor/dot-inl.h"

namespace mxnet {
namespace op {

using dot_fwd_t    = dnnl::matmul;
using dot_fwd_pd_t = dnnl::matmul::primitive_desc;

typedef ParamOpSign<DotParam> DotSignature;

class DNNLDotFwd {
 public:
  static DNNLDotFwd& GetCached(const DotParam& param,
                               const std::vector<NDArray>& inputs,
                               const std::vector<NDArray>& outputs,
                               const bool isNumpy);

  DNNLDotFwd(const DotParam& param,
             const std::vector<NDArray>& inputs,
             const std::vector<NDArray>& outputs,
             const bool isNumpy);

  void Execute(const OpContext& ctx,
               const std::vector<NDArray>& inputs,
               const std::vector<OpReqType>& req,
               const std::vector<NDArray>& outputs,
               const bool isNumpy);

 private:
  std::shared_ptr<dot_fwd_t> fwd;
  std::shared_ptr<dot_fwd_pd_t> fwd_pd;
};

template <bool isNumpy>
void DNNLDotForward(const nnvm::NodeAttrs& attrs,
                    const OpContext& ctx,
                    const std::vector<NDArray>& inputs,
                    const std::vector<OpReqType>& req,
                    const std::vector<NDArray>& outputs) {
  DotParam param;
  if (isNumpy) {
    // NumPy version of dot operator does not support transpose flags.
    param             = DotParam();
    param.transpose_a = false;
    param.transpose_b = false;
  } else {
    param = nnvm::get<DotParam>(attrs.parsed);
  }
  DNNLDotFwd& fwd = DNNLDotFwd::GetCached(param, inputs, outputs, isNumpy);
  fwd.Execute(ctx, inputs, req, outputs, isNumpy);
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_DOT_INL_H_

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
 * \file dnnl_split-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_SPLIT_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_SPLIT_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <vector>

#include "./dnnl_base-inl.h"

namespace mxnet {
namespace op {

using split_fwd_t    = dnnl::reorder;
using split_fwd_pd_t = dnnl::reorder::primitive_desc;

class DNNLSplitFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& input, const std::vector<NDArray>& outputs);

    const NDArray& input;
    const std::vector<NDArray>& outputs;
  };

  static DNNLSplitFwd& GetCached(const SplitParam& param,
                                 const Tensors& tensors,
                                 const TShape& split_pts,
                                 const int split_axis);

  DNNLSplitFwd(const Tensors& tensors, const TShape& split_pts, const int split_axis);

  void Execute(const Tensors& tensors,
               const TShape& split_pts,
               const int split_axis,
               const std::vector<OpReqType>& req) const;

 private:
  std::vector<split_fwd_t> split_fwds;
  std::vector<split_fwd_pd_t> split_pds;
  dnnl::memory::dims strides;
};

void DNNLSplitForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_SPLIT_INL_H_

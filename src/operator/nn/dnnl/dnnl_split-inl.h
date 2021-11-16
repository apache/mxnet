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
#include "./dnnl_ops-inl.h"

namespace mxnet {
namespace op {

using split_fwd_t    = dnnl::reorder;

class DNNLSplitFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& input, const std::vector<NDArray>& outputs);

    const NDArray& input;
    const std::vector<NDArray>& outputs;
  };

  static DNNLSplitFwd GetCached(const SplitParam& param,
                                const Tensors& tensors,
                                const bool is_train);

  // static split_fwd_pd_t GetSplitFwdPd(const dnnl::memory::desc& input_md,
  //                                     const dnnl::memory::desc& output_md);

  DNNLSplitFwd(const SplitParam& param, const Tensors& tensors, const bool is_train);

  void Execute(const Tensors& tensors) const;

 private:
  // std::shared_ptr<split_fwd_pd_t> split_pd;
  std::shared_ptr<split_fwd_t> split_fwd;
};


bool SupportDNNLSplit(const SplitParam& param, const NDArray& input);

}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_SPLIT_INL_H_
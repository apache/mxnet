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
 * \file dnnl_where-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_WHERE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_WHERE_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <memory>
#include <unordered_map>
#include <vector>
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

class DNNLWhereFwd {
 public:
  struct Tensors {
    Tensors(const std::vector<NDArray>& inputs, const std::vector<NDArray>& outputs);
    const NDArray& condition;
    const NDArray& left;
    const NDArray& right;
    const NDArray& output;
  };

  static DNNLWhereFwd GetCached(const Tensors& tensors);

  explicit DNNLWhereFwd(const Tensors& tensors);

  void Execute(const Tensors& tensors,
               const std::vector<OpReqType>& req,
               const OpContext& ctx) const;

 private:
  dnnl::binary::primitive_desc binary_eq_zero_pd;
  dnnl::binary::primitive_desc binary_ne_zero_pd;
  dnnl::binary::primitive_desc binary_mul_l_pd;
  dnnl::binary::primitive_desc binary_mul_r_pd;
  dnnl::binary::primitive_desc binary_sum_pd;
  dnnl::binary binary_eq_zero;
  dnnl::binary binary_ne_zero;
  dnnl::binary binary_mul_l;
  dnnl::binary binary_mul_r;
  dnnl::binary binary_sum;
};

bool SupportDNNLWhere(const std::vector<NDArray>& inputs);

void DNNLWhereForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const std::vector<NDArray>& inputs,
                      const std::vector<OpReqType>& req,
                      const std::vector<NDArray>& outputs);

}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_WHERE_INL_H_

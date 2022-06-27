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
 * \file dnnl_reduce-inl.h
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_REDUCE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_REDUCE_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <vector>

#include "./dnnl_base-inl.h"

namespace mxnet {
namespace op {

using reduce_fwd_t    = dnnl::reduction;
using reduce_fwd_pd_t = dnnl::reduction::primitive_desc;
struct NumpyReduceAxesParam;
struct ReduceAxesParam;
class DNNLReduceFwd {
 public:
  struct Tensors {
    Tensors(const NDArray& data, const NDArray& out);

    const NDArray& data;
    const NDArray& out;
  };

  static DNNLReduceFwd GetCached(const NumpyReduceAxesParam& param,
                                 const Tensors& tensors,
                                 const bool is_train,
                                 const dnnl::algorithm reduction_alg);

  static reduce_fwd_pd_t GetReduceFwdPd(const dnnl::memory::desc& input_md,
                                        const dnnl::memory::desc& output_md,
                                        const dnnl::algorithm reduction_alg);

  DNNLReduceFwd(const NumpyReduceAxesParam& param,
                const Tensors& tensors,
                const bool is_train,
                const dnnl::algorithm reduction_alg);
  void Execute(const Tensors& tensors) const;

 private:
  std::shared_ptr<reduce_fwd_pd_t> reduce_pd;
  std::shared_ptr<reduce_fwd_t> reduce_fwd;
};

template <class T>
NumpyReduceAxesParam ConvertReduceParamsToNumpy(const T& original_param,
                                                const NDArray& in_data,
                                                const NDArray& out_data);

void DNNLReduceForwardImpl(const NumpyReduceAxesParam& param,
                           const OpContext& ctx,
                           const NDArray& in_data,
                           const OpReqType& req,
                           const NDArray& out_data,
                           const dnnl::algorithm reduction_alg);

template <class ParamType, dnnl::algorithm reduction_alg>
void DNNLReduceForward(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const NDArray& in_data,
                       const OpReqType& req,
                       const NDArray& out_data) {
  const ParamType& org_param = nnvm::get<ParamType>(attrs.parsed);
  auto param                 = ConvertReduceParamsToNumpy<ParamType>(org_param, in_data, out_data);
  DNNLReduceForwardImpl(param, ctx, in_data, req, out_data, reduction_alg);
}

bool SupportDNNLReduceImpl(const NumpyReduceAxesParam& param,
                           const NDArray& in_data,
                           const NDArray& out_data);

template <class T>
bool SupportDNNLReduce(const nnvm::NodeAttrs& attrs,
                       const NDArray& in_data,
                       const NDArray& out_data) {
  const T& org_param = nnvm::get<T>(attrs.parsed);
  auto param         = ConvertReduceParamsToNumpy<T>(org_param, in_data, out_data);
  return SupportDNNLReduceImpl(param, in_data, out_data);
}

}  // namespace op
}  // namespace mxnet
#endif
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_REDUCE_INL_H_

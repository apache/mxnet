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
 * \file dnnl_transpose-inl.h
 * \author Rafal Litka
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_TRANSPOSE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_TRANSPOSE_INL_H_
#if MXNET_USE_ONEDNN == 1

#include "dnnl_base-inl.h"

#include "operator/numpy/np_matrix_op-inl.h"

namespace mxnet {
namespace op {

class DNNLTransposeFwd {
 public:
  std::shared_ptr<dnnl::memory> data_;
  std::shared_ptr<dnnl::memory> out_;
  std::shared_ptr<dnnl::memory::desc> dst_md_;
  std::shared_ptr<dnnl::reorder> transpose_;
  DNNLTransposeFwd(const NumpyTransposeParam& param, const NDArray& data);
  void SetNewMem(const NDArray& data, const NDArray& output);
  const dnnl::reorder& GetFwd() const;
  void Execute() const;
};

DNNLTransposeFwd& GetTransposeForward(const NumpyTransposeParam& param, const NDArray& data);

template <class ParamType>
NumpyTransposeParam ConvertTransposeParamsToNumpy(const ParamType& param);

template <class ParamType>
void DNNLTransposeForward(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const NDArray& data,
                          const OpReqType& req,
                          const NDArray& output) {
  const ParamType& org_param = nnvm::get<ParamType>(attrs.parsed);
  auto param                 = ConvertTransposeParamsToNumpy<ParamType>(org_param);
  auto fwd                   = GetTransposeForward(param, data);
  fwd.SetNewMem(data, output);
  fwd.Execute();
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_TRANSPOSE_INL_H_

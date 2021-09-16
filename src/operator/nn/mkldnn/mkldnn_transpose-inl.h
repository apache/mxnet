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
 * \file mkldnn_transpose-inl.h
 * \brief
 * \author Rafal Litka
 */

#ifndef MXNET_OPERATOR_NN_MKLDNN_MKLDNN_TRANSPOSE_INL_H_
#define MXNET_OPERATOR_NN_MKLDNN_MKLDNN_TRANSPOSE_INL_H_
#if MXNET_USE_ONEDNN == 1
#include "./mkldnn_base-inl.h"
#include "./mkldnn_ops-inl.h"

#include "../../numpy/np_matrix_op-inl.h"
#include "../../operator_common.h"
#include "../../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {
bool SupportMKLDNNTranspose(const NDArray& data);
class MKLDNNTransposeFwd {
 public:
  std::shared_ptr<mkldnn::memory> data_;
  std::shared_ptr<mkldnn::memory> out_;
  std::shared_ptr<mkldnn::memory::desc> dst_md_;
  std::shared_ptr<mkldnn::reorder> transpose_;
  MKLDNNTransposeFwd(const NumpyTransposeParam& param, const NDArray& data);
  void SetNewMem(const NDArray& data, const NDArray& output);
  const mkldnn::reorder& GetFwd() const;
  void Execute() const;
};

MKLDNNTransposeFwd& GetTransposeForward(const NumpyTransposeParam& param, const NDArray& data);

template <typename ParamType>
NumpyTransposeParam ProcessTransposeParam(const nnvm::NodeAttrs& attrs);

template <>
NumpyTransposeParam ProcessTransposeParam<NumpyTransposeParam>(const nnvm::NodeAttrs& attrs);

template <>
NumpyTransposeParam ProcessTransposeParam<TransposeParam>(const nnvm::NodeAttrs& attrs);

template <typename ParamType>
void MKLDNNTransposeForward(const nnvm::NodeAttrs& attrs,
                            const OpContext& ctx,
                            const NDArray& data,
                            const OpReqType& req,
                            const NDArray& output) {
  const NumpyTransposeParam param = ProcessTransposeParam<ParamType>(attrs);
  auto fwd                        = GetTransposeForward(param, data);
  fwd.SetNewMem(data, output);
  fwd.Execute();
}

}  // namespace op
}  // namespace mxnet
#endif
#endif
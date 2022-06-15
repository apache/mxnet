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
 * \file dnnl_reshape-inl.h
 * \brief Function definition of dnnl reshape operator
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_RESHAPE_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_RESHAPE_INL_H_

#if MXNET_USE_ONEDNN == 1
#include <vector>

#include "operator/tensor/matrix_op-inl.h"
#include "dnnl_base-inl.h"

namespace mxnet {
namespace op {

class DNNLReshapeFwd {
 protected:
  std::shared_ptr<dnnl::memory> out_;
  std::shared_ptr<dnnl::memory> temp_;
  std::vector<dnnl::primitive> prims_;

 public:
  DNNLReshapeFwd(const OpReqType& req, const NDArray& input, const NDArray& output);
  int GetWorkspaceSize();
  void Execute(const NDArray& input,
               const NDArray& output,
               const OpReqType& req,
               void* workspace = nullptr);
};

typedef OpSignature DNNLReshapeSignature;
DNNLReshapeFwd& GetReshapeForward(const OpReqType& req,
                                  const NDArray& input,
                                  const NDArray& output);

void DNNLReshapeForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const NDArray& input,
                        const OpReqType& req,
                        const NDArray& output);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_RESHAPE_INL_H_

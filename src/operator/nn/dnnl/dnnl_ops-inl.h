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
 * \file dnnl_ops-inl.h
 * \brief
 * \author Da Zheng
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_OPS_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_OPS_INL_H_

#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <mxnet/base.h>
#include <mxnet/io.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>

#include <vector>

#if MXNET_USE_ONEDNN == 1
#include <dnnl.hpp>

namespace mxnet {
namespace op {

// void DNNLSumForward(const nnvm::NodeAttrs& attrs,
//                     const OpContext& ctx,
//                     const std::vector<NDArray>& inputs,
//                     const std::vector<OpReqType>& req,
//                     const std::vector<NDArray>& outputs);

// void DNNLCopy(const nnvm::NodeAttrs& attrs,
//               const OpContext& ctx,
//               const NDArray& in_data,
//               const OpReqType& req,
//               const NDArray& out_data);

// void DNNLSum(const dnnl::memory& arr1, const dnnl::memory& arr2, const dnnl::memory& out);

// void DNNLStackForward(const nnvm::NodeAttrs& attrs,
//                       const OpContext& ctx,
//                       const std::vector<NDArray>& in_data,
//                       const std::vector<OpReqType>& req,
//                       const std::vector<NDArray>& out_data);

void DNNLPowerForward(const nnvm::NodeAttrs& attrs,
                      const OpContext& ctx,
                      const NDArray& input,
                      const OpReqType& req,
                      const NDArray& output);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_OPS_INL_H_

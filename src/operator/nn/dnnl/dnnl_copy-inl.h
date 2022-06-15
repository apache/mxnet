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
 * \file dnnl_copy-inl.h
 * \brief
 * \author Wolinski Piotr piotr.wolinski@intel.com
 */

#ifndef MXNET_OPERATOR_NN_DNNL_DNNL_COPY_INL_H_
#define MXNET_OPERATOR_NN_DNNL_DNNL_COPY_INL_H_

#if MXNET_USE_ONEDNN == 1

#include <dnnl.hpp>

namespace mxnet {
namespace op {

void DNNLCopy(const nnvm::NodeAttrs& attrs,
              const OpContext& ctx,
              const NDArray& in_data,
              const OpReqType& req,
              const NDArray& out_data);

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_NN_DNNL_DNNL_COPY_INL_H_

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
 * \file cudnn_batch_norm.h
 * \brief
 * \author Junyuan Xie
 */

#ifndef MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_H_
#define MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_H_

#include <mxnet/base.h>
#include <vector>
#include "../batch_norm-inl.h"

namespace mxnet {
namespace op {

#if MXNET_USE_CUDNN == 1

STATIC_ASSERT_CUDNN_VERSION_GE(7401);

bool CudnnBatchNormSupports(const BatchNormParam& param, const TBlob& x);

void CudnnBatchNormForward(const BatchNormParam& param,
                           const OpContext& ctx,
                           const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs);

void CudnnBatchNormBackward(const BatchNormParam& param,
                            const OpContext& ctx,
                            const std::vector<TBlob>& inputs,
                            const std::vector<OpReqType>& req,
                            const std::vector<TBlob>& outputs);

#endif  // MXNET_USE_CUDNN == 1

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CUDNN_CUDNN_BATCH_NORM_H_
